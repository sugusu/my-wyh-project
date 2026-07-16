#!/usr/bin/env python3
"""Independent candidate export replay entrypoint.

This run is intentionally blocked because the official baseline source and
checkpoints were not acquired. The script exists so the Stage 0 project
contains the required replay entrypoint, but it refuses to fabricate results.
"""

from __future__ import annotations


def main() -> int:
    raise SystemExit(
        "NOT-RUN: official baseline source/checkpoints are unavailable; "
        "candidate export replay cannot be performed."
    )


if __name__ == "__main__":
    main()
