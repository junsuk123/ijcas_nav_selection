#!/bin/bash
# fusion_dev follow-up (docs/FUSION_DESIGN_KO.md): build -> train -> validation eval -> MATLAB R2025a figures.
# Validation split only (seeds 1, 3, 6, 8); no test data is used.  Logs: results/fusion/after_collect.log
cd "$(dirname "$0")/../.."
while pgrep -f "run_batch.py --set fusion_dev" > /dev/null; do sleep 300; done
echo "$(date '+%F %T') collection finished: DONE $(ls $HOME/ijcas_data/runs/fusion_dev/*/DONE 2>/dev/null | wc -l)/100"
python3 tools/fusion_build_dataset.py --set fusion_dev --workers 6 > results/fusion/build_fusion_dev.log 2>&1
echo "$(date '+%F %T') datasets: $(ls $HOME/ijcas_data/fusion/dataset/fusion_dev/*.npz | wc -l), failed $(ls $HOME/ijcas_data/fusion/dataset/fusion_dev/*.FAILED 2>/dev/null | wc -l)"
for V in rgat gru; do
  python3 tools/fusion_run.py train --set fusion_dev --variants $V --seeds 0,1,2 --batch 60 --device cuda >> results/fusion/train_fusion_dev.log 2>&1
  echo "$(date '+%F %T') trained $V rc=$?"
done
python3 tools/fusion_run.py eval --set fusion_dev --split val --device cuda > results/fusion/eval_val_1.log 2>&1
echo "$(date '+%F %T') eval (rgat, gru) rc=$?"
for V in rgat_A rgat_noonto rgat_block; do
  python3 tools/fusion_run.py train --set fusion_dev --variants $V --seeds 0,1,2 --batch 60 --device cuda >> results/fusion/train_fusion_dev.log 2>&1
  echo "$(date '+%F %T') trained $V rc=$?"
done
python3 tools/fusion_run.py eval --set fusion_dev --split val --device cuda > results/fusion/eval_val.log 2>&1
echo "$(date '+%F %T') eval (all) rc=$?"
cd matlab && /usr/local/MATLAB/R2025a/bin/matlab -batch "cfg = ijcas_config(); for r = [\"S10_003\" \"S02_008\" \"S04_006\"], ijcas_plot(cfg, Run=r, Variant=\"rgat\", Seed=0, Split=\"val\", Device=\"cuda\"); end" > ../results/fusion/plot.log 2>&1
echo "$(date '+%F %T') figures rc=$? -> results/fusion/figures"
echo "DONE"
