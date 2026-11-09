#!/usr/bin/env python3
"""run_study4.py: entry point for Study 4 (improver arm, pilot: A0 vs A1).

What it does:
  * --check verifies the Study 4 pre-registration (study4/guard.py) and exits: 0 if a valid LOCKED
    pre-registration matches the code and kernel, 3 otherwise. Reads no API key, writes no file.
  * --run verifies the SAME way first and is refused (exit 3) on any failure, BEFORE touching the
    live loop at all. Only once verification passes does it import study4/run_live.py, read the
    model API key from the environment (same handling as Study 3's kernel/llm_meta.py -- never
    written to a file, log, or error message), and run the loop.

    python run_study4.py --check                        # report whether a locked pre-registration is in place
    python run_study4.py --run --arm A1 --cycles 3       # Study 4 pilot, arm A1 (improver allowed)
    python run_study4.py --run --arm A0 --cycles 3       # Study 4 pilot, arm A0 (solver-only baseline)
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from kernel.llm_meta import GROQ_BASE_URL, GROQ_MODEL, LLMError  # noqa: E402
from study4.guard import PREREG_PATH, PreregError, verify


def parse_args(argv=None, prereg_path: Path = PREREG_PATH) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--check", action="store_true", help="verify the pre-registration and exit")
    g.add_argument("--run", action="store_true", help="verify the pre-registration, then run the live loop")
    ap.add_argument("--prereg", type=Path, default=prereg_path, help=argparse.SUPPRESS)
    # --run options (ignored by --check; defaults match run_study3_live.py where they overlap)
    ap.add_argument("--out", default="runs/study4_live")
    ap.add_argument("--arm", default="A1", choices=["A0", "A1"], help="A0: solver-only baseline; A1: improver allowed")
    ap.add_argument("--backend", choices=["groq", "openai-compatible"], default="groq")
    ap.add_argument("--model", default=GROQ_MODEL)
    ap.add_argument("--base-url", default="", help="default: Groq's OpenAI-compatible endpoint")
    ap.add_argument("--api-key-env", default="", help="name of the env var holding the key (default: OPENAI_API_KEY, then GROQ_API_KEY)")
    ap.add_argument("--cycles", type=int, default=3)
    ap.add_argument("--n", type=int, default=48, help="problems per selection / confirm batch")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--anchor-every", type=int, default=5)
    ap.add_argument("--heldout-every", type=int, default=10)
    ap.add_argument("--token-budget", type=int, default=20_000, help="LLM tokens per cycle")
    ap.add_argument("--run-token-cap", type=int, default=200_000, help="LLM tokens for the whole run (0 = no cap)")
    ap.add_argument("--max-calls-per-cycle", type=int, default=3)
    ap.add_argument("--min-interval", type=float, default=3.0, help="seconds between LLM calls (rate limiting)")
    ap.add_argument("--max-context-chars", type=int, default=12_000, help="repo text sent per call")
    ap.add_argument("--max-tokens", type=int, default=1500, help="reply length cap per call")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--container-image", default="")
    ap.add_argument("--yield-profile", default="smoke", choices=["tiny", "smoke", "full"], help="arm A1 only")
    ap.add_argument("--yield-starts", type=int, default=2, help="arm A1 only")
    ap.add_argument("--yield-seeds", type=int, default=2, help="arm A1 only")
    ap.add_argument("--yield-k", type=int, default=4, help="arm A1 only: cycles per yield measurement")
    args = ap.parse_args(argv)
    if not args.base_url:
        args.base_url = GROQ_BASE_URL
    return args


def main(argv: list[str] | None = None, prereg_path: Path = PREREG_PATH) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = parse_args(argv, prereg_path)
    try:
        verify(args.prereg)
    except PreregError as e:
        print(f"REFUSED: {e}", file=sys.stderr)
        return 3
    print(f"pre-registration OK: {args.prereg}")
    if args.check:
        return 0
    from study4.run_live import run as run_live        # imported only once verification has passed
    try:
        return run_live(args)
    except KeyboardInterrupt:
        print(f"\nstopped (Ctrl+C). Partial results are in {args.out}; the ledger keeps every finished cycle.",
              file=sys.stderr)
        return 130
    except LLMError as e:
        print(f"provider error: {e}", file=sys.stderr)
        return 3


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
