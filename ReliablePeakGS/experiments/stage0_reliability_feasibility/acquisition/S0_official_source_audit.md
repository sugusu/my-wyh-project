# S0 Official Source Audit

## Scope

This audit searched for official publication/project/source locations for TSPE-GS, TSGS, TransLab, and officially referenced datasets. Only official author/project/repository sources were considered acceptable.

## Primary Candidate

TSPE-GS is the selected candidate baseline because its official project and paper describe robust multi-view transparent surface reconstruction with a transparent surface-points extractor and multi-layer transparent surface handling. The official repository is:

https://github.com/nortonii/TSPE-GS

Official project page:

https://nortonii.github.io/TSPE-GS/

Paper page:

https://arxiv.org/abs/2511.09944

The GitHub web page showed default branch `main` and short commit `cac432c` at the time of audit. A full local commit lock could not be made because both `git` and `curl` failed against GitHub over TLS from this server.

## Related Sources

TSGS official project:

https://longxiang-ai.github.io/TSGS/

TSGS official repository:

https://github.com/longxiang-ai/TSGS

TSGS was not selected as the primary Stage 0 baseline because the Stage 0 protocol requires a transparent Gaussian multi-depth or multi-peak depth candidate baseline. TSGS is relevant prior work and may be useful for context, but the primary Stage 0 candidate export must come from a method that exposes multi-depth hypotheses.

## Local Acquisition Attempts

All commands were run under `/data/wyh/ReliablePeakGS/external`.

- `git clone https://github.com/nortonii/TSPE-GS.git TSPE-GS`
  - result: failed, `gnutls_handshake() failed: The TLS connection was non-properly terminated`
- `git clone --depth 1 https://github.com/nortonii/TSPE-GS.git TSPE-GS`
  - result: failed, same TLS error
- `git ls-remote https://github.com/nortonii/TSPE-GS.git HEAD`
  - result: failed, same TLS error
- `curl -L --retry 3 --retry-delay 2 -o TSPE-GS-main.zip https://github.com/nortonii/TSPE-GS/archive/refs/heads/main.zip`
  - result: failed, TLS unexpected EOF
- `git ls-remote https://github.com/longxiang-ai/TSGS.git HEAD`
  - result: failed, same TLS error
- `curl -L --retry 3 --retry-delay 2 -o TSGS-main.zip https://github.com/longxiang-ai/TSGS/archive/refs/heads/main.zip`
  - result: failed, TLS unexpected EOF

## S0-G0 Decision

S0-G0 requires at least one official transparent Gaussian multi-depth or multi-peak baseline repository and at least one compatible public dataset containing camera calibration and scale-consistent GT geometry.

Result: FAIL_BLOCKED_LOCAL_SOURCE_ACQUISITION.

Reason: an official candidate repository exists, but the protocol also requires cloning and locking exact source commits before data download, baseline reproduction, candidate export, and evaluation. This server could not acquire the official repository. The project must stop rather than fabricate source lock, candidate exports, labels, or metrics.
