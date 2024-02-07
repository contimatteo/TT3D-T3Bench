###

# exit 0

GPU=0
PROMPT="test_t3bench_n1"

ROOT_DIR="/media/data2/mconti/TT3D"
PROMPT_DIR="${ROOT_DIR}/prompts"
SOURCE_DIR="${ROOT_DIR}/outputs/${PROMPT}"
PROMPT_FILE="${PROMPT_DIR}/${PROMPT}.txt"
OUT_DIR="${ROOT_DIR}/metrics/T3Bench/${PROMPT}"


###
### QUALITY EVALUATION
###

### Fantasia3D
# CUDA_VISIBLE_DEVICES=${GPU} python3 tt3d_eval_quality.py \
#   --model "fantasia3d" \
#   --prompt-file $PROMPT_FILE \
#   --source-path "${SOURCE_DIR}/Threestudio-Fantasia3D/" \
#   --out-path "${OUT_DIR}"

### ShapE
CUDA_VISIBLE_DEVICES=${GPU} python3 tt3d_eval_quality.py \
  --model "shap-e" \
  --prompt-file $PROMPT_FILE \
  --source-path "${SOURCE_DIR}/OpenAI-ShapE/" \
  --out-path "${OUT_DIR}" \
  --skip-existing-renderings \
  --skip-existing-scores

### PointE
CUDA_VISIBLE_DEVICES=${GPU} python3 tt3d_eval_quality.py \
  --model "point-e" \
  --prompt-file $PROMPT_FILE \
  --source-path "${SOURCE_DIR}/OpenAI-PointE/" \
  --out-path "${OUT_DIR}" \
  --skip-existing-renderings \
  --skip-existing-captions \
  --skip-existing-scores

###
### ALIGNMENT EVALUATION
###

### Fantasia3D
# CUDA_VISIBLE_DEVICES=${GPU} python3 tt3d_eval_alignment.py \
#   --model "fantasia3d" \
#   --prompt-file $PROMPT_FILE \
#   --source-path "${SOURCE_DIR}/Threestudio-Fantasia3D/" \
#   --out-path "${OUT_DIR}"

### ShapE
CUDA_VISIBLE_DEVICES=${GPU} python3 tt3d_eval_alignment.py \
  --model "shap-e" \
  --prompt-file $PROMPT_FILE \
  --source-path "${SOURCE_DIR}/OpenAI-ShapE/" \
  --out-path "${OUT_DIR}" \
  --skip-existing-renderings \
  --skip-existing-scores

### PointE
CUDA_VISIBLE_DEVICES=${GPU} python3 tt3d_eval_alignment.py \
  --model "point-e" \
  --prompt-file $PROMPT_FILE \
  --source-path "${SOURCE_DIR}/OpenAI-PointE/" \
  --out-path "${OUT_DIR}" \
  --skip-existing-renderings \
  --skip-existing-captions \
  --skip-existing-scores
