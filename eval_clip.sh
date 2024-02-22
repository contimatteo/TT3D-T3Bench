###

exit 1


GPU=3
ENV="report"
PROMPT="n100"
EXPERIMENT_PREFIX="t3bench/single"

# ROOT_DIR="/media/data2/mconti/TT3D"
ROOT_DIR="mconti/TT3D"
PROMPT_FILE="/media/data2/${ROOT_DIR}/prompts/${EXPERIMENT_PREFIX}/${PROMPT}.txt"
# SOURCE_DIR="/media/data2/${ROOT_DIR}/outputs/${ENV}/${EXPERIMENT_PREFIX}/${PROMPT}"
SOURCE_DIR="/media/data2/${ROOT_DIR}/metrics/${ENV}/${EXPERIMENT_PREFIX}/${PROMPT}"
OUT_DIR="/media/data3/${ROOT_DIR}/metrics/${ENV}/${EXPERIMENT_PREFIX}/${PROMPT}"

export TRANSFORMERS_OFFLINE=1
export DIFFUSERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1


###


echo ">"
echo "> [quality] ShapE"
echo ">"

### OpenAI-ShapE
CUDA_VISIBLE_DEVICES=${GPU} python3 tt3d_eval_clip.py \
  --model "shap-e" \
  --prompt-file $PROMPT_FILE \
  --source-path "${SOURCE_DIR}/" \
  --out-path "${OUT_DIR}" \
  --skip-existing
