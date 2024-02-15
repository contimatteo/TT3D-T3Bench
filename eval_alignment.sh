###

exit 1


GPU=3
ENV="test"
PROMPT="n0_n1"
EXPERIMENT_PREFIX="t3bench/single"

ROOT_DIR="/media/data2/mconti/TT3D"
SOURCE_DIR="${ROOT_DIR}/outputs/${ENV}/${EXPERIMENT_PREFIX}/${PROMPT}"
OUT_DIR="${ROOT_DIR}/metrics/${ENV}/${EXPERIMENT_PREFIX}/${PROMPT}"
PROMPT_FILE="${ROOT_DIR}/prompts/${EXPERIMENT_PREFIX}/${PROMPT}.txt"


###


echo ">"
echo "> [quality] ShapE"
echo ">"

### OpenAI-ShapE
CUDA_VISIBLE_DEVICES=${GPU} python3 tt3d_eval_alignment.py \
  --model "shap-e" \
  --prompt-file $PROMPT_FILE \
  --source-path "${SOURCE_DIR}/OpenAI-ShapE/" \
  --out-path "${OUT_DIR}" \
  --skip-existing-renderings \
  --skip-existing-captions \
  --skip-existing-scores

### Cap3D-ShapE
CUDA_VISIBLE_DEVICES=${GPU} python3 tt3d_eval_alignment.py \
  --model "shap-e" \
  --prompt-file $PROMPT_FILE \
  --source-path "${SOURCE_DIR}/Cap3D-ShapE/" \
  --out-path "${OUT_DIR}" \
  --skip-existing-renderings \
  --skip-existing-captions \
  --skip-existing-scores

echo ">"
echo "> [quality] PointE"
echo ">"

### OpenAI-PointE
CUDA_VISIBLE_DEVICES=${GPU} python3 tt3d_eval_alignment.py \
  --model "point-e" \
  --prompt-file $PROMPT_FILE \
  --source-path "${SOURCE_DIR}/OpenAI-PointE/" \
  --out-path "${OUT_DIR}" \
  --skip-existing-renderings \
  --skip-existing-captions \
  --skip-existing-scores

### Cap3D-PointE
CUDA_VISIBLE_DEVICES=${GPU} python3 tt3d_eval_alignment.py \
  --model "point-e" \
  --prompt-file $PROMPT_FILE \
  --source-path "${SOURCE_DIR}/Cap3D-PointE/" \
  --out-path "${OUT_DIR}" \
  --skip-existing-renderings \
  --skip-existing-captions \
  --skip-existing-scores

echo ">"
echo "> [quality] Threestudio-DreamFusion"
echo ">"

### Threestudio-DreamFusion(sd)
CUDA_VISIBLE_DEVICES=${GPU} python3 tt3d_eval_alignment.py \
  --model "dreamfusion-sd" \
  --prompt-file $PROMPT_FILE \
  --source-path "${SOURCE_DIR}/Threestudio-DreamFusion/" \
  --out-path "${OUT_DIR}" \
  --skip-existing-renderings \
  --skip-existing-captions \
  --skip-existing-scores

### Threestudio-DreamFusion(if)
CUDA_VISIBLE_DEVICES=${GPU} python3 tt3d_eval_alignment.py \
  --model "dreamfusion-if" \
  --prompt-file $PROMPT_FILE \
  --source-path "${SOURCE_DIR}/Threestudio-DreamFusion/" \
  --out-path "${OUT_DIR}" \
  --skip-existing-renderings \
  --skip-existing-captions \
  --skip-existing-scores

echo ">"
echo "> [quality] Threestudio-Fantasia3D"
echo ">"

### Threestudio-Fantasia3D
CUDA_VISIBLE_DEVICES=${GPU} python3 tt3d_eval_alignment.py \
  --model "fantasia3d" \
  --prompt-file $PROMPT_FILE \
  --source-path "${SOURCE_DIR}/Threestudio-Fantasia3D/" \
  --out-path "${OUT_DIR}" \
  --skip-existing-renderings \
  --skip-existing-captions \
  --skip-existing-scores

echo ">"
echo "> [quality] Threestudio-ProlificDreamer"
echo ">"

### Threestudio-ProlificDreamer
CUDA_VISIBLE_DEVICES=${GPU} python3 tt3d_eval_alignment.py \
  --model "prolificdreamer" \
  --prompt-file $PROMPT_FILE \
  --source-path "${SOURCE_DIR}/Threestudio-ProlificDreamer/" \
  --out-path "${OUT_DIR}" \
  --skip-existing-renderings \
  --skip-existing-captions \
  --skip-existing-scores

echo ">"
echo "> [quality] Threestudio-Magic3D"
echo ">"

### Threestudio-Magic3D(sd)
CUDA_VISIBLE_DEVICES=${GPU} python3 tt3d_eval_alignment.py \
  --model "magic3d-sd" \
  --prompt-file $PROMPT_FILE \
  --source-path "${SOURCE_DIR}/Threestudio-Magic3D/" \
  --out-path "${OUT_DIR}" \
  --skip-existing-renderings \
  --skip-existing-captions \
  --skip-existing-scores

### Threestudio-Magic3D(if)
CUDA_VISIBLE_DEVICES=${GPU} python3 tt3d_eval_alignment.py \
  --model "magic3d-if" \
  --prompt-file $PROMPT_FILE \
  --source-path "${SOURCE_DIR}/Threestudio-Magic3D/" \
  --out-path "${OUT_DIR}" \
  --skip-existing-renderings \
  --skip-existing-captions \
  --skip-existing-scores

echo ">"
echo "> [quality] Threestudio-TextMesh"
echo ">"

### Threestudio-TextMesh(sd)
CUDA_VISIBLE_DEVICES=${GPU} python3 tt3d_eval_alignment.py \
  --model "textmesh-sd" \
  --prompt-file $PROMPT_FILE \
  --source-path "${SOURCE_DIR}/Threestudio-TextMesh/" \
  --out-path "${OUT_DIR}" \
  --skip-existing-renderings \
  --skip-existing-captions \
  --skip-existing-scores

### Threestudio-TextMesh(if)
CUDA_VISIBLE_DEVICES=${GPU} python3 tt3d_eval_alignment.py \
  --model "textmesh-if" \
  --prompt-file $PROMPT_FILE \
  --source-path "${SOURCE_DIR}/Threestudio-TextMesh/" \
  --out-path "${OUT_DIR}" \
  --skip-existing-renderings \
  --skip-existing-captions \
  --skip-existing-scores

echo ">"
echo "> [quality] Threestudio-HiFA"
echo ">"

### Threestudio-HiFA
CUDA_VISIBLE_DEVICES=${GPU} python3 tt3d_eval_alignment.py \
  --model "hifa" \
  --prompt-file $PROMPT_FILE \
  --source-path "${SOURCE_DIR}/Threestudio-HiFA/" \
  --out-path "${OUT_DIR}" \
  --skip-existing-renderings \
  --skip-existing-captions \
  --skip-existing-scores
