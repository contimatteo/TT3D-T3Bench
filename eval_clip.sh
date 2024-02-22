###

exit 1


GPU=3
ENV="report"
PROMPT="n100"
EXPERIMENT_PREFIX="t3bench/single"

# ROOT_DIR="/media/data2/mconti/TT3D"
ROOT_DIR="mconti/TT3D"
PROMPT_FILE="/media/data2/${ROOT_DIR}/prompts/${EXPERIMENT_PREFIX}/${PROMPT}.txt"
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

# echo ">"
# echo "> [quality] ShapE"
# echo ">"

# ### OpenAI-ShapE
# CUDA_VISIBLE_DEVICES=${GPU} python3 tt3d_eval_clip.py \
#   --model "shap-e" \
#   --prompt-file $PROMPT_FILE \
#   --source-path "${SOURCE_DIR}/" \
#   --out-path "${OUT_DIR}" \
#   --skip-existing

# ### Cap3D-ShapE
# CUDA_VISIBLE_DEVICES=${GPU} python3 tt3d_eval_clip.py \
#   --model "cap3d-shap-e" \
#   --prompt-file $PROMPT_FILE \
#   --source-path "${SOURCE_DIR}/" \
#   --out-path "${OUT_DIR}" \
#   --skip-existing

# echo ">"
# echo "> [quality] PointE"
# echo ">"

# ### OpenAI-PointE
# CUDA_VISIBLE_DEVICES=${GPU} python3 tt3d_eval_clip.py \
#   --model "point-e" \
#   --prompt-file $PROMPT_FILE \
#   --source-path "${SOURCE_DIR}/" \
#   --out-path "${OUT_DIR}" \
#   --skip-existing

# ### Cap3D-PointE
# CUDA_VISIBLE_DEVICES=${GPU} python3 tt3d_eval_clip.py \
#   --model "cap3d-point-e" \
#   --prompt-file $PROMPT_FILE \
#   --source-path "${SOURCE_DIR}/" \
#   --out-path "${OUT_DIR}" \
#   --skip-existing

# echo ">"
# echo "> [quality] Threestudio-DreamFusion"
# echo ">"

# ### Threestudio-DreamFusion(sd)
# CUDA_VISIBLE_DEVICES=${GPU} python3 tt3d_eval_clip.py \
#   --model "dreamfusion-sd" \
#   --prompt-file $PROMPT_FILE \
#   --source-path "${SOURCE_DIR}/" \
#   --out-path "${OUT_DIR}" \
#   --skip-existing

# ### Threestudio-DreamFusion(if)
# CUDA_VISIBLE_DEVICES=${GPU} python3 tt3d_eval_clip.py \
#   --model "dreamfusion-if" \
#   --prompt-file $PROMPT_FILE \
#   --source-path "${SOURCE_DIR}/" \
#   --out-path "${OUT_DIR}" \
#   --skip-existing

# echo ">"
# echo "> [quality] Threestudio-Fantasia3D"
# echo ">"

# ### Threestudio-Fantasia3D
# CUDA_VISIBLE_DEVICES=${GPU} python3 tt3d_eval_clip.py \
#   --model "fantasia3d" \
#   --prompt-file $PROMPT_FILE \
#   --source-path "${SOURCE_DIR}/" \
#   --out-path "${OUT_DIR}" \
#   --skip-existing

# echo ">"
# echo "> [quality] Threestudio-ProlificDreamer"
# echo ">"

# ### Threestudio-ProlificDreamer
# CUDA_VISIBLE_DEVICES=${GPU} python3 tt3d_eval_clip.py \
#   --model "prolificdreamer" \
#   --prompt-file $PROMPT_FILE \
#   --source-path "${SOURCE_DIR}/" \
#   --out-path "${OUT_DIR}" \
#   --skip-existing

# echo ">"
# echo "> [quality] Threestudio-Magic3D"
# echo ">"

# ### Threestudio-Magic3D(sd)
# CUDA_VISIBLE_DEVICES=${GPU} python3 tt3d_eval_clip.py \
#   --model "magic3d-sd" \
#   --prompt-file $PROMPT_FILE \
#   --source-path "${SOURCE_DIR}/" \
#   --out-path "${OUT_DIR}" \
#   --skip-existing

# ### Threestudio-Magic3D(if)
# CUDA_VISIBLE_DEVICES=${GPU} python3 tt3d_eval_clip.py \
#   --model "magic3d-if" \
#   --prompt-file $PROMPT_FILE \
#   --source-path "${SOURCE_DIR}/" \
#   --out-path "${OUT_DIR}" \
#   --skip-existing

# echo ">"
# echo "> [quality] Threestudio-TextMesh"
# echo ">"

# ### Threestudio-TextMesh(sd)
# CUDA_VISIBLE_DEVICES=${GPU} python3 tt3d_eval_clip.py \
#   --model "textmesh-sd" \
#   --prompt-file $PROMPT_FILE \
#   --source-path "${SOURCE_DIR}/" \
#   --out-path "${OUT_DIR}" \
#   --skip-existing

# ### Threestudio-TextMesh(if)
# CUDA_VISIBLE_DEVICES=${GPU} python3 tt3d_eval_clip.py \
#   --model "textmesh-if" \
#   --prompt-file $PROMPT_FILE \
#   --source-path "${SOURCE_DIR}/" \
#   --out-path "${OUT_DIR}" \
#   --skip-existing

# ### Threestudio-TextMesh(if)
### CUDA_VISIBLE_DEVICES=${GPU} python3 tt3d_eval_clip.py \
###   --model "textmesh-if" \
###   --prompt-file $PROMPT_FILE \
###   --source-path "${SOURCE_DIR}/" \
###   --out-path "${OUT_DIR}" \
###   --skip-existing

# echo ">"
# echo "> [quality] Threestudio-HiFA"
# echo ">"

# ### Threestudio-HiFA
# CUDA_VISIBLE_DEVICES=${GPU} python3 tt3d_eval_clip.py \
#   --model "hifa" \
#   --prompt-file $PROMPT_FILE \
#   --source-path "${SOURCE_DIR}/" \
#   --out-path "${OUT_DIR}" \
#   --skip-existing

# echo ">"
# echo "> [quality] LucidDreamer"
# echo ">"

# ### LucidDreamer
# CUDA_VISIBLE_DEVICES=${GPU} python3 tt3d_eval_clip.py \
#   --model "luciddreamer" \
#   --prompt-file $PROMPT_FILE \
#   --source-path "/media/data3/${ROOT_DIR}/metrics/${ENV}/${EXPERIMENT_PREFIX}/${PROMPT}" \
#   --out-path "${OUT_DIR}" \
#   --skip-existing
