#!/usr/bin/env bash
# Drive jobs/final.sh from this Mac and pull every finished run back immediately,
# so nothing exists only on the pod.
#
#   bash jobs/final_local.sh <ip> <port>
set -uo pipefail
IP=${1:?ip}; PORT=${2:?port}
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SSH="ssh -o StrictHostKeyChecking=accept-new -o ServerAliveInterval=30 -i $HOME/.ssh/id_ed25519 -p $PORT root@$IP"
cd "$ROOT"

if [ -z "${SKIP_SETUP:-}" ]; then
echo "== upload code $(date +%T)"
$SSH 'mkdir -p /root/niti'
COPYFILE_DISABLE=1 tar czf - --no-xattrs train bench jobs data | $SSH 'tar xzf - -C /root/niti'

echo "== setup $(date +%T)"
$SSH 'cd /root/niti && bash jobs/pod_setup_pinned.sh' 2>&1 | tail -12
fi

if [ -z "${SKIP_LAUNCH:-}" ]; then
echo "== launch $(date +%T)"
$SSH 'cd /root/niti && nohup bash jobs/final.sh > final.out 2>&1 &'
fi

mkdir -p results/final
pending="${RUNS:-large lowlr perm small}"
while [ -n "$pending" ]; do
    sleep 20
    left=""
    for r in $pending; do
        if $SSH "test -f /root/niti/final/$r/DONE"; then
            $SSH "tar cf - --exclude=$r/checkpoints -C /root/niti/final $r $r.log" | tar xf - -C results/final \
              && echo "== pulled $r $(date +%T)" \
              && grep -h "holdout KL" results/final/$r.log \
              && uv run -q --with pyarrow python bench/score.py results/final/$r/preds_temper.jsonl \
                   --json results/final/$r/score.json > /dev/null
        elif $SSH "grep -q Traceback /root/niti/final/$r.log 2>/dev/null"; then
            echo "== $r FAILED:"; $SSH "grep -A3 -m1 Traceback /root/niti/final/$r.log; tail -3 /root/niti/final/$r.log"
        else
            left="$left $r"
        fi
    done
    pending=$left
done
echo "== all pulled $(date +%T); safe to terminate the pod"

# Pre-registered selection: lowest holdout KL among the 2B variants. Test is printed
# only after the choice is made.
best=$(for r in ${RUNS:-large lowlr perm}; do
         [ "$r" = small ] && continue
         k=$(grep -h "holdout KL" results/final/$r.log 2>/dev/null | sed 's/.*soft-fit //')
         [ -n "$k" ] && echo "$k $r"
       done | sort -n | head -1 | cut -d' ' -f2)
echo "== selected 2B variant by holdout KL: $best"
for r in $best small; do
    echo -n "$r test: "
    uv run -q --with pyarrow python bench/score.py results/final/$r/preds_temper.jsonl | sed -n 4p
done
