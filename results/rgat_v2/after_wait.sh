#!/bin/bash
# v2 follow-up that needs no judgement (design 13): val summary after training; test2 tensors (v2 and v1
# code) after collection.  Freeze and the one-time test2 evaluation are NOT run here (reviewed step).
cd "$(dirname "$0")/../.."
D=$HOME/ijcas_data
until grep -q "ALL DONE" results/rgat_v2/train_stdout.log 2>/dev/null; do sleep 120; done
echo "$(date '+%F %T') training finished; reports: $(ls results/rgat_v2/runs/*/seed*/report.json 2>/dev/null | wc -l)/30"
IJCAS_RGAT_VERSION=v2 python3 tools/rgat_run.py val-summary > results/rgat_v2/val_summary_stdout.log 2>&1
echo "$(date '+%F %T') val-summary rc=$?"
while pgrep -f "run_batch.py --set test2" > /dev/null; do sleep 300; done
echo "$(date '+%F %T') collection finished: DONE $(ls $D/runs/test2/*/DONE 2>/dev/null | wc -l)/330, FAILED $(ls $D/runs/test2/*/FAILED 2>/dev/null | wc -l)"
python3 tools/rgat_build_dataset.py --set test2 --thresholds config/selector/thresholds_validated.yaml \
  --out-root $D/rgat/dataset_v2 --workers 6 > results/rgat_v2/build_test2.log 2>&1
echo "$(date '+%F %T') v2 test2 tensors: $(ls $D/rgat/dataset_v2/test2/*.npz 2>/dev/null | wc -l)"
python3 tools/rgat_test2.py build-v1 --workers 6 > results/rgat_v2/build_test2_v1.log 2>&1
echo "$(date '+%F %T') v1 test2 tensors: $(ls $D/rgat/dataset_v1/test2/*.npz 2>/dev/null | wc -l)"
echo "READY for review -> freeze -> tools/rgat_test2.py run (once)"
