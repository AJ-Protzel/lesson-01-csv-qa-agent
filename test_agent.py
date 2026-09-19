"""Fixed Q&A regression suite for the CSV agent.

Ten questions with known answers. Each case carries three things:

  question   what the agent is asked (plain English, no hints)
  grader     how the agent's answer is judged
  reference  pandas that recomputes the truth straight from the CSV

The `reference` column is what keeps the suite honest: `--self-check` runs it
against data/sales.csv and confirms every expected value still matches, so a
regenerated dataset can never leave stale expectations sitting in this file.

    python test_agent.py --self-check   # verify expectations, no API calls
    python test_agent.py --list         # show the suite, no API calls
    python test_agent.py                # run the agent against all 10
    python test_agent.py -k month       # run matching cases only
"""
from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from typing import Callable

import pandas as pd

import csv_agent

CSV_PATH = "data/sales.csv"


# --------------------------------------------------------------------------
# graders -- a grader takes the agent's answer string and returns True/False
# --------------------------------------------------------------------------

def number(expected: float, tol_pct: float = 0.1) -> Callable[[str], bool]:
    """Pass if any number in the answer is within tol_pct of `expected`.

    Tolerant of formatting ("647,900.57", "647900.5723") but not of arithmetic.
    """
    def grade(answer: str) -> bool:
        found = [float(m) for m in re.findall(r"-?\d+(?:\.\d+)?", answer.replace(",", ""))]
        margin = abs(expected) * tol_pct / 100
        return any(abs(f - expected) <= margin for f in found)

    grade.__doc__ = f"a number within {tol_pct}% of {expected:,.2f}"
    return grade


def mentions(*tokens: str) -> Callable[[str], bool]:
    """Pass if every token appears in the answer (case-insensitive)."""
    def grade(answer: str) -> bool:
        haystack = answer.lower().replace(",", "")
        return all(t.lower() in haystack for t in tokens)

    grade.__doc__ = "mentions " + " and ".join(f"'{t}'" for t in tokens)
    return grade


def any_of(*tokens: str) -> Callable[[str], bool]:
    """Pass if at least one token appears (for answers with several spellings)."""
    def grade(answer: str) -> bool:
        haystack = answer.lower().replace(",", "")
        return any(t.lower() in haystack for t in tokens)

    grade.__doc__ = "mentions " + " or ".join(f"'{t}'" for t in tokens)
    return grade


# --------------------------------------------------------------------------
# the suite
# --------------------------------------------------------------------------

@dataclass
class Case:
    id: str
    question: str
    expected: str                                  # human-readable, shown in the report
    grader: Callable[[str], bool]
    reference: Callable[[pd.DataFrame], object]    # recomputes `expected` from the CSV
    probes: str                                    # what this case actually tests


CASES: list[Case] = [
    Case(
        id="march-top-category",
        question="Which product category had the highest total revenue in March 2024?",
        expected="Electronics",
        grader=mentions("electronics"),
        reference=lambda d: d[d.date.dt.month == 3].groupby("category").revenue.sum().idxmax(),
        probes="filter by month, then group and rank",
    ),
    Case(
        id="total-revenue",
        question="What is the total revenue across every row in the file, including returned orders?",
        expected="647,900.57",
        grader=number(647900.57, tol_pct=0.01),
        reference=lambda d: round(d.revenue.sum(), 2),
        probes="whole-column aggregate, no filtering",
    ),
    Case(
        id="row-count",
        question="How many orders are in this dataset?",
        expected="400",
        grader=number(400, tol_pct=0),
        reference=lambda d: len(d),
        probes="baseline -- a wrong answer here means something is badly broken",
    ),
    Case(
        id="top-region",
        question="Which region generated the most total revenue overall?",
        expected="North",
        grader=mentions("north"),
        reference=lambda d: d.groupby("region").revenue.sum().idxmax(),
        probes="group and rank on a plain categorical",
    ),
    Case(
        id="laptop-units",
        question="How many Laptop units were sold in total, counting returned orders too?",
        expected="280",
        grader=number(280, tol_pct=0),
        reference=lambda d: int(d[d["product"] == "Laptop"].units.sum()),
        probes="row filter, then sum a column other than revenue",
    ),
    Case(
        id="avg-order-value",
        question="What is the average revenue per order across the whole file?",
        expected="1,619.75",
        grader=number(1619.75, tol_pct=0.1),
        reference=lambda d: round(d.revenue.mean(), 2),
        probes="mean vs sum -- catches the classic wrong-aggregate slip",
    ),
    Case(
        id="returned-count",
        question="How many orders were returned?",
        expected="22",
        grader=number(22, tol_pct=0),
        reference=lambda d: int(d.returned.sum()),
        probes="boolean column handled as a condition, not a string",
    ),
    Case(
        id="electronics-west-units",
        question="How many units of Electronics were sold in the West region?",
        expected="118",
        grader=number(118, tol_pct=0),
        reference=lambda d: int(d[(d.category == "Electronics") & (d.region == "West")].units.sum()),
        probes="two conditions combined correctly",
    ),
    Case(
        id="best-month",
        question="Which calendar month had the highest total revenue? Give the month name.",
        expected="May (131,822.86 -- 0.6% ahead of March)",
        grader=any_of("may", "2024-05"),
        reference=lambda d: d.groupby(d.date.dt.strftime("%B")).revenue.sum().idxmax(),
        probes="tight margin -- a dropped row or wrong filter flips this to March",
    ),
    Case(
        id="biggest-order",
        question="What was the largest single order by revenue, and which product was it for?",
        expected="ORD-00195, 14,520.00, Laptop",
        grader=mentions("laptop", "14520"),
        reference=lambda d: d.loc[d.revenue.idxmax(), ["order_id", "revenue", "product"]].to_dict(),
        probes="locate a row, not just an aggregate; return two facts at once",
    ),
]


# --------------------------------------------------------------------------
# runners
# --------------------------------------------------------------------------

def self_check(df: pd.DataFrame) -> int:
    """Confirm each hardcoded `expected` still matches the CSV. No API calls."""
    print(f"self-check against {CSV_PATH} ({len(df)} rows)\n")
    failures = 0
    for case in CASES:
        truth = csv_agent.format_answer(case.reference(df))
        ok = case.grader(truth)
        failures += not ok
        print(f"  [{'ok   ' if ok else 'STALE'}] {case.id:<24} reference says: {truth}")
    print()
    if failures:
        print(f"{failures} expected value(s) no longer match the data -- update CASES.")
    else:
        print(f"all {len(CASES)} expected values verified against the CSV.")
    return 1 if failures else 0


def list_cases() -> int:
    for i, case in enumerate(CASES, 1):
        print(f"{i:>2}. {case.id}")
        print(f"    ask:      {case.question}")
        print(f"    expect:   {case.expected}")
        print(f"    pass if:  {case.grader.__doc__}")
        print(f"    probes:   {case.probes}\n")
    return 0


def run(df: pd.DataFrame, cases: list[Case], effort: str, show_code: bool) -> int:
    passed = 0
    rows = []

    for i, case in enumerate(cases, 1):
        print(f"[{i}/{len(cases)}] {case.id} ... ", end="", flush=True)
        result = csv_agent.ask(df, case.question, effort=effort, verbose=show_code)

        if not result.ok:
            verdict, got = "ERROR", result.error.splitlines()[-1][:60]
        else:
            got = " ".join(result.answer.split())[:60]
            verdict = "PASS" if case.grader(result.answer) else "FAIL"
        passed += verdict == "PASS"

        print(f"{verdict} ({result.attempts} attempt{'s' if result.attempts > 1 else ''})")
        rows.append((case.id, verdict, case.expected, got, result.attempts))

    print(f"\n{'case':<24} {'result':<7} {'expected':<28} {'agent said':<32} tries")
    print("-" * 100)
    for cid, verdict, expected, got, attempts in rows:
        print(f"{cid:<24} {verdict:<7} {expected[:27]:<28} {got[:31]:<32} {attempts}")

    print(f"\n{passed}/{len(cases)} passed")
    return 0 if passed == len(cases) else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--csv", default=CSV_PATH)
    parser.add_argument("--self-check", action="store_true",
                        help="verify expected values against the CSV, no API calls")
    parser.add_argument("--list", action="store_true", help="print the suite and exit")
    parser.add_argument("-k", metavar="SUBSTR", help="only run cases whose id matches")
    parser.add_argument("--effort", default="low",
                        choices=["low", "medium", "high", "xhigh", "max"])
    parser.add_argument("--show-code", action="store_true", help="print generated pandas")
    args = parser.parse_args()

    if args.list:
        return list_cases()

    df = csv_agent.load_csv(args.csv)
    if args.self_check:
        return self_check(df)

    cases = [c for c in CASES if not args.k or args.k in c.id]
    if not cases:
        print(f"no cases match {args.k!r}", file=sys.stderr)
        return 1
    return run(df, cases, args.effort, args.show_code)


if __name__ == "__main__":
    raise SystemExit(main())
