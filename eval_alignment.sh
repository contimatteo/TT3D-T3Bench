###

exit 0

GPU=0
PROMPT="test_t3bench_n1"

ROOT_DIR="/media/data2/mconti/TT3D"
SOURCE_DIR="${ROOT_DIR}/outputs/${PROMPT}"
PROMPT_FILE="${ROOT_DIR}/prompts/${PROMPT}.txt"
OUT_DIR="${ROOT_DIR}/metrics/T3Bench/${PROMPT}"


###


echo ">"
echo "> [alignment] OpenAI-ShapE"
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

echo ">"
echo "> [alignment] OpenAI-PointE"
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

echo ">"
echo "> [alignment] Threestudio-DreamFusion"
echo ">"

### Threestudio-DreamFusion
CUDA_VISIBLE_DEVICES=${GPU} python3 tt3d_eval_alignment.py \
  --model "dreamfusion-sd" \
  --prompt-file $PROMPT_FILE \
  --source-path "${SOURCE_DIR}/Threestudio-DreamFusion/" \
  --out-path "${OUT_DIR}" \
  --skip-existing-renderings \
  --skip-existing-captions \
  --skip-existing-scores

echo ">"
echo "> [alignment] Threestudio-Fantasia3D"
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
echo "> [alignment] Threestudio-ProlificDreamer"
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
echo "> [alignment] Threestudio-Magic3D"
echo ">"

### Threestudio-Magic3D
CUDA_VISIBLE_DEVICES=${GPU} python3 tt3d_eval_alignment.py \
  --model "magic3d" \
  --prompt-file $PROMPT_FILE \
  --source-path "${SOURCE_DIR}/Threestudio-Magic3D/" \
  --out-path "${OUT_DIR}" \
  --skip-existing-renderings \
  --skip-existing-captions \
  --skip-existing-scores

echo ">"
echo "> [alignment] Threestudio-TextMesh"
echo ">"

### Threestudio-TextMesh
CUDA_VISIBLE_DEVICES=${GPU} python3 tt3d_eval_alignment.py \
  --model "textmesh-sd" \
  --prompt-file $PROMPT_FILE \
  --source-path "${SOURCE_DIR}/Threestudio-TextMesh/" \
  --out-path "${OUT_DIR}" \
  --skip-existing-renderings \
  --skip-existing-captions \
  --skip-existing-scores

echo ">"
echo "> [alignment] Threestudio-HiFA"
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