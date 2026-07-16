#!/usr/bin/env python3
"""Independent GT-label replay entrypoint.

Labels require candidate exports and verified GT geometry. Those artifacts were
not produced because official source acquisition failed.
"""

from __future__ import annotations


def main() -> int:
    raise SystemExit(
        "NOT-RUN: candidate exports and verified GT geometry are unavailable; "
        "GT label replay cannot be performed."
    )


if __name__ == "__main__":
    main()
