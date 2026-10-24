#!/usr/bin/env bash
# RunPod pod controller for Niti runs.
#
#   export RUNPOD_API_KEY=...
#   bash jobs/runpod.sh create     # A100 SXM 80GB, 60 GB disk, SSH only
#   bash jobs/runpod.sh wait       # poll until RUNNING, print ssh details
#   bash jobs/runpod.sh delete     # terminate (stops billing)
#
# Pod id is cached in results/.pod_id so later calls need no arguments.
set -euo pipefail

API="https://rest.runpod.io/v1"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ID_FILE="$ROOT/results/.pod_id"

GPU_TYPE="${GPU_TYPE:-NVIDIA A100-SXM4-80GB}"
CLOUD="${CLOUD:-SECURE}"
IMAGE="${IMAGE:-runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04}"

need_key() {
    : "${RUNPOD_API_KEY:?RUNPOD_API_KEY is not set}"
}

cmd_create() {
    need_key
    curl -sS -X POST "$API/pods" \
        -H "Authorization: Bearer $RUNPOD_API_KEY" \
        -H "Content-Type: application/json" \
        -d "$(cat <<JSON
{
  "name": "niti-a100",
  "imageName": "$IMAGE",
  "gpuTypeIds": ["$GPU_TYPE"],
  "gpuCount": 1,
  "containerDiskInGb": 60,
  "volumeInGb": 0,
  "ports": ["22/tcp"],
  "supportPublicIp": true,
  "cloudType": "$CLOUD"
}
JSON
)" | python3 -c "
import json, sys, pathlib
d = json.load(sys.stdin)
if 'id' not in d:
    print('CREATE FAILED:', json.dumps(d)[:400]); raise SystemExit(1)
pathlib.Path('$ID_FILE').write_text(d['id'])
print('pod', d['id'], '|', d.get('desiredStatus'), '|', d.get('costPerHr'), '/hr')
"
}

cmd_wait() {
    need_key
    local id; id="$(cat "$ID_FILE")"
    for _ in $(seq 1 60); do
        curl -sS -H "Authorization: Bearer $RUNPOD_API_KEY" "$API/pods/$id" | python3 -c "
import json, sys
d = json.load(sys.stdin)
print(d.get('desiredStatus'), '|', d.get('publicIp') or 'no-ip', '|', d.get('portMappings') or {})
" || true
        if curl -sS -H "Authorization: Bearer $RUNPOD_API_KEY" "$API/pods/$id" \
            | grep -q '"desiredStatus":"RUNNING"'; then break; fi
        sleep 10
    done
    curl -sS -H "Authorization: Bearer $RUNPOD_API_KEY" "$API/pods/$id" \
        | python3 -c "
import json, sys
d = json.load(sys.stdin)
ip, pm = d.get('publicIp'), (d.get('portMappings') or {})
port = pm.get('22') or pm.get(22)
print(f\"ssh -i ~/.ssh/id_ed25519 root@{ip} -p {port}\" if ip and port else 'not ready')
"
}

cmd_delete() {
    need_key
    local id; id="$(cat "$ID_FILE")"
    curl -sS -X DELETE -H "Authorization: Bearer $RUNPOD_API_KEY" "$API/pods/$id" -o /dev/null -w "delete HTTP %{http_code}\n"
    rm -f "$ID_FILE"
}

case "${1:-}" in
    create) cmd_create ;;
    wait)   cmd_wait ;;
    delete) cmd_delete ;;
    *) echo "usage: bash jobs/runpod.sh {create|wait|delete}"; exit 2 ;;
esac
