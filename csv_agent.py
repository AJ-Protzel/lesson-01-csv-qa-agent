"""Ask questions about a CSV in plain English.

Claude writes pandas code against the file, the code runs locally, and the value
it produces is the answer.

    python csv_agent.py data/sales.csv "which category had the highest sales in March"

Two backends reach the same model with the same prompt:

    --backend api   Anthropic API; needs ANTHROPIC_API_KEY (default)
    --backend cli   `claude -p`; runs on a logged-in Claude Code subscription
"""
from __future__ import annotations

import argparse
import io
import os
import re
import shutil
import subprocess
import sys
import traceback
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

MODEL = "claude-opus-5"
MAX_ATTEMPTS = 3  # 1 first try + 2 repair attempts

SYSTEM = """You write pandas code to answer questions about a CSV file.

A DataFrame named `df` is already loaded. Reply with exactly one fenced python
code block and nothing else -- no prose before or after it.

Rules:
- Assign the final answer to a variable named `result`.
- Keep `result` small and literal: a number, a string, or a short Series/dict.
  Never assign the whole DataFrame.
- Only `pd` (pandas), `np` (numpy) and `df` are available. Do not import
  anything, read or write files, or touch the network.
- The question is the spec. If it does not mention returns, count every row.
- Date columns are already parsed as datetime64.

Example reply:

```python
result = df.groupby("region")["revenue"].sum().idxmax()
```
"""

# Guardrail, not a sandbox -- see the security note in README.md.
BANNED = re.compile(
    r"\b(import|__import__|eval|exec|compile|open|globals|locals|getattr|setattr)\b"
)


@dataclass
class AgentResult:
    """One question, and everything it took to answer it."""

    question: str
    answer: str | None = None
    code: str | None = None
    error: str | None = None
    attempts: int = 0
    transcript: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.error is None


def load_csv(path: str) -> pd.DataFrame:
    """Read the CSV, parsing anything that looks like a date column."""
    df = pd.read_csv(path)
    for col in df.columns:
        if "date" in col.lower() or "time" in col.lower():
            try:
                df[col] = pd.to_datetime(df[col])
            except (ValueError, TypeError):
                pass  # column just happens to be named "date"; leave it alone
    return df


def describe(df: pd.DataFrame, max_uniques: int = 12) -> str:
    """A compact schema blurb: columns, dtypes, category values, sample rows.

    Listing the actual values of low-cardinality columns is what stops the model
    guessing at labels like "elec" when the data says "Electronics".
    """
    lines = [f"{len(df)} rows x {len(df.columns)} columns", "", "Columns:"]
    for col in df.columns:
        line = f"  {col} ({df[col].dtype})"
        if df[col].dtype == object and df[col].nunique() <= max_uniques:
            line += " values: " + ", ".join(map(str, sorted(df[col].unique())))
        elif pd.api.types.is_datetime64_any_dtype(df[col]):
            line += f" range: {df[col].min():%Y-%m-%d} to {df[col].max():%Y-%m-%d}"
        lines.append(line)

    buf = io.StringIO()
    df.head(3).to_csv(buf, index=False)
    lines += ["", "First 3 rows:", buf.getvalue().strip()]
    return "\n".join(lines)


def extract_code(text: str) -> str:
    """Pull the python out of the model's fenced block."""
    match = re.search(r"```(?:python)?\s*\n(.*?)```", text, re.DOTALL)
    code = match.group(1) if match else text
    return code.strip()


def run_code(code: str, df: pd.DataFrame) -> str:
    """Execute generated code against a copy of the frame; return `result`.

    Raises on banned constructs, on any runtime error, or if `result` is unset --
    all three come back to the model as a repair prompt.
    """
    if hit := BANNED.search(code):
        raise ValueError(f"generated code used a disallowed name: {hit.group(0)}")

    namespace: dict = {"pd": pd, "np": np, "df": df.copy()}
    exec(code, namespace)  # noqa: S102 -- see README security note

    if "result" not in namespace:
        raise ValueError("code ran but never assigned `result`")
    return format_answer(namespace["result"])


def format_answer(value) -> str:
    """Render a result value as a stable one-liner."""
    if isinstance(value, (np.integer, np.floating)):
        value = value.item()
    if isinstance(value, float):
        return f"{value:,.2f}"
    if isinstance(value, (pd.Series, pd.DataFrame)):
        return value.to_string()
    return str(value)


def call_api(messages: list[dict], model: str, effort: str) -> str:
    """One turn through the Anthropic API. Bills API credits."""
    import anthropic

    response = anthropic.Anthropic().messages.create(
        model=model,
        max_tokens=4000,
        system=SYSTEM,
        messages=messages,
        thinking={"type": "adaptive"},
        output_config={"effort": effort},
    )
    if response.stop_reason == "refusal":
        raise RuntimeError("the model declined to answer this request")
    return "".join(b.text for b in response.content if b.type == "text")


def find_claude() -> str:
    """Locate the Claude Code executable, which the installer may leave off PATH."""
    found = shutil.which("claude")
    if found:
        return found
    fallback = Path.home() / ".local" / "bin" / ("claude.exe" if os.name == "nt" else "claude")
    if fallback.exists():
        return str(fallback)
    raise RuntimeError("Claude Code not found; install it or use --backend api")


def call_cli(messages: list[dict], model: str, effort: str) -> str:
    """One turn through `claude -p`. Runs on the Claude Code subscription.

    Each `claude -p` call starts fresh, so a repair turn replays the earlier
    exchange as a single prompt. `--tools ""` leaves the model no tools at all:
    it can only reply with text, never touch files.
    """
    if len(messages) == 1:
        prompt = messages[0]["content"]
    else:
        prompt = "\n\n".join(f"[{m['role']}]\n{m['content']}" for m in messages)

    # A key in the environment would make Claude Code bill the API instead.
    env = {k: v for k, v in os.environ.items() if k != "ANTHROPIC_API_KEY"}
    proc = subprocess.run(
        [
            find_claude(), "-p",
            "--system-prompt", SYSTEM,
            "--model", model,
            "--effort", effort,
            "--tools", "",
            "--no-session-persistence",
        ],
        input=prompt,
        capture_output=True,
        text=True,
        encoding="utf-8",
        env=env,
        timeout=300,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"claude -p exited {proc.returncode}: {proc.stderr.strip()[-300:]}")
    return proc.stdout


BACKENDS = {"api": call_api, "cli": call_cli}


def ask(
    df: pd.DataFrame,
    question: str,
    backend: str = "api",
    model: str = MODEL,
    effort: str = "low",
    verbose: bool = False,
) -> AgentResult:
    """Answer one question. Retries with the traceback when the code fails."""
    call = BACKENDS[backend]
    out = AgentResult(question=question)
    messages: list[dict] = [
        {
            "role": "user",
            "content": f"CSV summary:\n\n{describe(df)}\n\nQuestion: {question}",
        }
    ]

    for attempt in range(1, MAX_ATTEMPTS + 1):
        out.attempts = attempt
        try:
            reply = call(messages, model, effort)
        except Exception as exc:
            out.error = str(exc)
            return out

        code = extract_code(reply)
        out.code = code
        out.transcript.append(code)
        if verbose:
            print(f"--- attempt {attempt} ---\n{code}\n", file=sys.stderr)

        try:
            out.answer = run_code(code, df)
            out.error = None
            return out
        except Exception:
            err = traceback.format_exc(limit=1).strip()
            out.error = err
            if verbose:
                print(f"{err}\n", file=sys.stderr)
            if attempt == MAX_ATTEMPTS:
                break
            messages += [
                {"role": "assistant", "content": reply},
                {
                    "role": "user",
                    "content": f"That code failed:\n\n{err}\n\nFix it and reply with the corrected block only.",
                },
            ]

    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("csv", help="path to the CSV file")
    parser.add_argument("question", help="question to ask, in plain English")
    parser.add_argument("--backend", default="api", choices=list(BACKENDS),
                        help="api: API credits (default); cli: Claude Code subscription")
    parser.add_argument("--model", default=MODEL)
    parser.add_argument(
        "--effort",
        default="low",
        choices=["low", "medium", "high", "xhigh", "max"],
        help="how hard the model thinks (default: low -- these queries are small)",
    )
    parser.add_argument("--show-code", action="store_true", help="print the generated pandas")
    args = parser.parse_args()

    df = load_csv(args.csv)
    result = ask(df, args.question, backend=args.backend, model=args.model,
                 effort=args.effort, verbose=args.show_code)

    if not result.ok:
        print(f"failed after {result.attempts} attempt(s): {result.error}", file=sys.stderr)
        return 1
    print(result.answer)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
