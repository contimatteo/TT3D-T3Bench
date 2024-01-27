###

exit 0

###
### QUALITY EVALUATION
###

CUDA_VISIBLE_DEVICES=2 python3 tt3d_eval_quality.py \
  --model "fantasia3d" \
  --prompt-file /media/data2/mconti/TT3D/prompts/test.v1.n2.txt \
  --source-path /media/data2/mconti/TT3D/models/Threestudio-Fantiasia3D/outputs \
  --out-path /media/data2/mconti/TT3D/metrics/T3Bench/outputs/fantasia3d
