#!/usr/bin/env bash
set -euo pipefail

out="${1:-/data/wyh/ReliablePeakGS/data/raw/alphasurf/data.zip}"
mkdir -p "$(dirname "$out")"

# Google Drive is not reachable by direct IPv4/IPv6 from this server, while the
# local proxy can at least reach the Drive confirmation endpoint. This script
# deliberately uses the ambient proxy variables.
curl --http1.1 -L \
  --retry 20 --retry-delay 5 --retry-all-errors \
  --continue-at - \
  -o "$out" \
  'https://drive.usercontent.google.com/download?id=10OvhfGj9P4t5esdy9NTriBWm-jKE2n8w&export=download&confirm=t&uuid=56b93a3f-c4c3-4749-bf6f-152db7c4255e'
