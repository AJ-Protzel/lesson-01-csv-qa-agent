# csv-qa-agent

Ask a CSV questions in plain English. Claude writes the pandas, the code runs
locally, and the value it produces is the answer.

```console
$ python csv_agent.py data/sales.csv "which category had the highest sales in March"
Electronics

$ python csv_agent.py data/sales.csv "average revenue per order in the West region" --show-code
--- attempt 1 ---
result = df[df["region"] == "West"]["revenue"].mean()

1,594.89
```

The interesting part isn't the wrapper — it's that an LLM writing code is only
useful if you can tell when it's wrong. So the repo is two files: the agent, and
a fixed regression suite that grades it.

## Install

```bash
pip install -r requirements.txt
```

Two backends reach the same model with the same prompt, retry loop, and output:

| backend | how it calls Claude | needs |
|---|---|---|
| `--backend api` (default) | Anthropic API via the Python SDK | `ANTHROPIC_API_KEY` set; bills API credits |
| `--backend cli` | `claude -p`, Claude Code's scripting mode | Claude Code installed and logged in; uses the subscription |

The `cli` backend runs with `--tools ""`, so Claude Code has no tools and can
only reply with text. It also strips `ANTHROPIC_API_KEY` from the child
environment, since a key there would make Claude Code bill the API instead.

## How the agent works

`csv_agent.py`, one loop:

1. **Describe the file.** Columns, dtypes, date ranges, and the full value list
   for any low-cardinality text column. That last part matters most — showing the
   model that `category` contains exactly `Apparel, Electronics, Furniture,
   Grocery` is what stops it filtering on `"electronics"` and silently returning
   an empty frame.
2. **Ask for code, not an answer.** The model gets the schema and the question
   and must reply with a single fenced python block that assigns to `result`.
   It sees 3 sample rows and the value lists, never the other 397, so it can't
   "answer" from the data — the arithmetic has to happen in pandas. That's also
   what lets the same design work on a file too large to fit in a prompt.
3. **Run it locally** against a copy of the DataFrame.
4. **Repair on failure.** Exception, banned construct, or a missing `result`
   all get handed back to the model as a repair prompt, up to 3 attempts total.
   This is the only place the loop actually loops.

Uses `claude-opus-5` with adaptive thinking at `effort=low` — these are one-line
groupbys, and low effort keeps a full test run cheap. `--effort high` if you're
pointing it at something gnarlier. `--model claude-sonnet-5` runs the same code on
a cheaper model; Haiku 4.5 won't work on the `api` backend, because it rejects
the adaptive-thinking and effort settings.

## The test suite

Ten fixed questions with known answers. `python test_agent.py` runs each one
through the real agent and prints a pass/fail table.

### Results

Two full runs, 2026-09-19, `--backend cli` at `effort=low`:

| model | passed | cases needing a repair attempt |
|---|---|---|
| `claude-opus-5` | 10/10 | 0 |
| `claude-sonnet-5` | 10/10 | 0 |

Both models answered every case correctly on the first attempt, so these runs
never exercised the repair loop — it's there for the failure path, not proven by
this table.

The two models formatted `biggest-order` differently, one returning a dict and
the other a tuple, and both passed. That's exactly why the graders match on
substrings and numeric tolerance rather than exact equality: the question was
answered correctly either way.

### Limits of this eval

Worth stating plainly, because 10/10 reads stronger than it is:

- The data, the questions and the prompt were written together, so nothing here
  independently checks the assumptions behind them.
- Every case passed on both models. A suite nothing fails can't separate a good
  build from a worse one — a useful eval has cases near the edge.
- The repair loop has never fired in a recorded run, so that path is unproven.
- The data is synthetic and clean: no missing values, no mixed date formats, no
  duplicate spellings of the same label, and no questions the tool should refuse.

The next version should use a real, messy export, with questions written by
someone other than whoever built the agent.

### How each case is defined

Each case declares four things:

| field | purpose |
|---|---|
| `question` | asked verbatim, with no hints about which columns to use |
| `expected` | the human-readable answer, shown in the report |
| `grader` | how the answer is judged — numeric tolerance, or required substrings |
| `reference` | pandas that recomputes the truth directly from the CSV |

`reference` is the part that keeps the suite honest. Hardcoded expectations rot
the moment the dataset changes, so:

```bash
python test_agent.py --self-check    # no API calls, no cost
```

recomputes every expected value from `data/sales.csv` and runs it through that
case's own grader. If the data and the expectations ever drift apart, this fails
loudly instead of the suite quietly grading against fiction. (It caught a real
mismatch while this repo was being written: `best-month`'s reference returned
the integer `5` while its grader wanted the word `May`. The question now asks
for a month name and the reference returns one.)

`python test_agent.py --list` prints the whole suite — question, expectation,
pass condition, and what each case is probing — without calling the API.

### What the cases cover

Deliberately not ten flavours of the same `groupby`:

- **`row-count`** — a baseline. If this fails, something is badly broken.
- **`total-revenue`, `avg-order-value`** — whole-column aggregates. The pair
  catches a mean/sum mixup, the most common wrong-aggregate slip.
- **`march-top-category`, `top-region`, `best-month`** — filter, group, rank.
- **`laptop-units`, `electronics-west-units`** — filtering on one and two
  conditions, aggregating a column that isn't `revenue`.
- **`returned-count`** — a boolean column, which invites `== "True"` string
  comparisons.
- **`biggest-order`** — locate a *row* rather than compute an aggregate, and
  return two facts at once.
- **`best-month`** is deliberately tight: May beats March by 0.6%. Any dropped
  row or stray filter flips it.

Numbers are graded within a tolerance (0.01% for totals, exact for counts) so
formatting differences pass and arithmetic errors don't. Text answers are graded
on required substrings.

### The ambiguity the questions handle explicitly

5% of rows have `returned = True`, and "total revenue" could reasonably mean
either with or without them. Rather than let the grader depend on a coin flip,
the questions that hinge on it say which they want ("...including returned
orders"), and the system prompt states the default: if the question doesn't
mention returns, count every row. The categorical answers (`Electronics`,
`North`, `May`, the biggest order) were checked to be stable under *both*
readings, so those cases can't be won or lost on that interpretation.

## Regenerating the data

`data/sales.csv` is committed so the suite is reproducible. It's synthetic —
400 orders across six months, four regions, twelve products, seeded at 7:

```bash
python data/make_data.py
python test_agent.py --self-check    # confirm expectations still hold
```

## Security note

This tool executes model-generated code on your machine with `exec`. There's a
regex guard rejecting `import`, `open`, `eval`, `getattr` and friends, and the
namespace is limited to `pd`, `np`, and `df` — but a regex blocklist is a
guardrail, not a sandbox, and it should not be treated as one. For anything
handling untrusted input, run it in a container or use the Claude API's
server-side code execution tool instead.

## Notes

What I took from building this:

- The loop is the easy half. The eval is what makes the thing trustworthy, and
  it's where the judgment calls are.
- Expected answers have to be produced independently of the model being tested,
  or the suite just grades the model against itself. That's what `--self-check`
  is for, and it caught a real mismatch during the build.
- A question with two defensible answers can't be graded at all. "Total revenue"
  is one number with returned orders and another without, so the question has to
  say which it wants.
- Graders should absorb formatting differences and still reject arithmetic
  errors. Opus returned one answer as a dict and Sonnet as a tuple, and both were
  right.
- A pass rate is only as good as whoever wrote the questions — see Limits above.

## Files

```
csv_agent.py        the agent: describe -> generate -> execute -> repair
test_agent.py       10 fixed cases, graders, and the self-check
data/make_data.py   seeded generator for the sample CSV
data/sales.csv      400 synthetic orders
requirements.txt    anthropic, pandas, numpy
```

MIT licensed.
