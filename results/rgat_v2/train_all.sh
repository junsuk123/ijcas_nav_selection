#!/bin/bash
# v2 training (design 13.3): main-question models first.
cd "$(dirname "$0")/../.."
export IJCAS_RGAT_VERSION=v2
for V in rgat_dyn gru_flat cr_plain,cr_e0,cr_gstar,cr_gstar_kd rgat_static,rgcn_dyn,rgat_nomem,rgat_dense; do
  echo "=== $(date '+%F %T') start $V"
  python3 tools/rgat_run.py train --variants "$V" --seeds 0,1,2
  echo "=== $(date '+%F %T') end $V rc=$?"
done
echo "=== ALL DONE $(date '+%F %T')"
