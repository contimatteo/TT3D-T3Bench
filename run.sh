###

exit 0

###
### QUALITY EVALUATION
###

CUDA_VISIBLE_DEVICES=3 python3 tt3d_eval_quality.py \
  --model "fantasia3d" \
  --prompt-file /media/data2/mconti/TT3D/prompts/test.v1.n2.txt \
  --source-path /media/data2/mconti/TT3D/models/Threestudio-Fantasia3D/outputs \
  --out-path /media/data2/mconti/TT3D/metrics/T3Bench/fantasia3d \
  --skip-existing-renderings \
  --skip-existing-scores

###
### ALIGNMENT EVALUATION
###

CUDA_VISIBLE_DEVICES=3 python3 tt3d_eval_alignment.py \
  --model "fantasia3d" \
  --prompt-file /media/data2/mconti/TT3D/prompts/test.v1.n2.txt \
  --source-path /media/data2/mconti/TT3D/models/Threestudio-Fantasia3D/outputs \
  --out-path /media/data2/mconti/TT3D/metrics/T3Bench/fantasia3d \
  --skip-existing-renderings \
  --skip-existing-captions \
  --skip-existing-scores
