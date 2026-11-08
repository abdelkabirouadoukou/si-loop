#!/usr/bin/env python3
"""run_study4.py: entry point for Study 4 (improver arm). NOT IMPLEMENTED YET.

What it does today:
  * checks the Study 4 pre-registration (study4/guard.py) and reports the result;
  * exits 3 if no valid locked pre-registration exists;
  * exits 2 even when one is valid, because the improver loop itself is not built yet.

It never reads API keys and never writes results.

    python run_study4.py --check              # report whether a locked pre-registration is in place
    python run_study4.py --run                # refused until the runner exists (exit 2)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from study4.guard import PREREG_PATH, PreregError, verify

NOT_IMPLEMENTED = ("Study 4 runner is not implemented. Design status: study4/DESIGN_DRAFT.md. "
                   "Nothing has been run for Study 4.")


def main(argv: list[str] | None = None, prereg_path: Path = PREREG_PATH) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--check", action="store_true", help="verify the pre-registration and exit")
    g.add_argument("--run", action="store_true", help="start a run (refused until the runner exists)")
    ap.add_argument("--prereg", type=Path, default=prereg_path, help=argparse.SUPPRESS)
    args = ap.parse_args(argv)
    try:
        verify(args.prereg)
    except PreregError as e:
        print(f"REFUSED: {e}", file=sys.stderr)
        return 3
    print(f"pre-registration OK: {args.prereg}")
    if args.check:
        return 0
    print(NOT_IMPLEMENTED, file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
