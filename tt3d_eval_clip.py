### pylint: disable=missing-function-docstring,missing-class-docstring,missing-module-docstring,wrong-import-order,wrong-import-position,import-error
from typing import Any, Tuple, Dict, List
from dotenv import load_dotenv

load_dotenv()  # take environment variables from .env.

from pathlib import Path
from copy import deepcopy

import argparse
import json
import torch
import traceback
import numpy as np

# from tqdm import tqdm
from PIL import Image
# from transformers import CLIPProcessor
from transformers import CLIPModel
from transformers import AutoProcessor
from transformers import AutoTokenizer

from utils import Utils

###

device = Utils.Cuda.init()

###


def _load_models() -> Tuple[Any, Any, Any]:
    tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
    # processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    return tokenizer, processor, model


def _load_rendering_images_paths(
    model: str,
    prompt: str,
    rootpath: Path,
) -> List[Path]:
    prompt_images_rootpath = Utils.Storage.build_renderings_path(
        model=model,
        prompt=prompt,
        out_rootpath=rootpath,
        eval_type='alignment',
        assert_exists=True,
    )
    prompt_images_paths = list(prompt_images_rootpath.glob("*.png"))
    prompt_images_paths = sorted(prompt_images_paths)
    prompt_images_paths = list(prompt_images_paths)

    ### INFO: T3Bench alignment rendering script outputs 12 images per prompt.
    assert len(prompt_images_paths) == 12

    return prompt_images_paths


_clip_cosine_similarity_fn = torch.nn.CosineSimilarity(dim=1, eps=1e-8)


def _compute_clip_similarity(
    prompt: str,
    images_paths: List[Path],
    clip_tokenizer: Any,
    clip_processor: Any,
    clip_model: Any,
) -> Tuple[float, np.ndarray, int]:
    assert isinstance(images_paths, list)
    assert len(images_paths) > 0
    assert all([isinstance(img_path, Path) for img_path in images_paths])
    assert all([img_path.exists() for img_path in images_paths])

    # images_raw = [Image.open(img_path).convert("RGB") for img_path in images_paths]
    # images_t = clip_processor(images=images_raw, return_tensors='pt')
    # prompt_t = clip_processor(text=prompt, padding=True, return_tensors='pt')  ### text ids
    # image_features = clip_model.get_image_features(images_t)
    # text_features = clip_model.get_text_features(prompt_t)
    # clip_similarities_t = _clip_cosine_similarity_fn(image_features, text_features)
    # assert isinstance(clip_similarities_t, torch.Tensor)
    # clip_similarities = clip_similarities_t.cpu().detach().numpy().tolist()
    # assert len(clip_similarities) == len(images_paths)
    # max_clip_similarity = max(clip_similarities)
    # assert 0.0 <= max_clip_similarity <= 1.0
    # return max_clip_similarity, clip_similarities

    prompt_t = clip_tokenizer(text=prompt, padding=True, return_tensors='pt')  ### text ids
    text_features = clip_model.get_text_features(**prompt_t)

    images_raw = [Image.open(img_path).convert("RGB") for img_path in images_paths]
    images_t = clip_processor(images=images_raw, return_tensors='pt')
    images_features = clip_model.get_image_features(**images_t)

    similarities_t = _clip_cosine_similarity_fn(images_features, text_features)
    assert isinstance(similarities_t, torch.Tensor)

    similarities_np = similarities_t.cpu().detach().numpy()
    assert similarities_np.shape[0] == len(images_paths)

    max_similarity_idx = similarities_np.argmax()
    max_similarity = similarities_np.max()
    assert 0.0 <= max_similarity <= 1.0

    return max_similarity, similarities_np, max_similarity_idx


def _evaluate_clip_similarity(
    model: str,
    prompt: str,
    source_rootpath: Path,
    out_rootpath: Path,
    clip_tokenizer: Any,
    clip_processor: Any,
    clip_model: Any,
    skip_existing: bool,
) -> None:
    assert clip_tokenizer is not None
    assert clip_processor is not None
    assert clip_model is not None

    out_scores_filepath = Utils.Storage.build_clip_similarity_scores_filepath(
        model=model,
        rootpath=out_rootpath,
        assert_exists=False,
    )
    out_scores_filepath.parent.mkdir(parents=True, exist_ok=True)

    scores_map: Dict[str, int] = None
    if not out_scores_filepath.exists():
        scores_map = {}
        out_scores_filepath.write_text("{}", encoding="utf-8")
    else:
        scores_map = json.loads(out_scores_filepath.read_text(encoding="UTF-8"))
        if skip_existing and prompt in scores_map:
            _score = scores_map[prompt]
            assert isinstance(_score, float)
            print("  > SIMILARITY (already exists) = ", _score)
            return _score

    assert isinstance(scores_map, dict)

    #

    # prompt_images_rootpath = Utils.Storage.build_renderings_path(
    #     model=model,
    #     prompt=prompt,
    #     out_rootpath=source_rootpath,
    #     eval_type='alignment',
    #     assert_exists=True,
    # )
    # prompt_images_paths = list(prompt_images_rootpath.glob("*.png"))
    # prompt_images_paths = sorted(prompt_images_paths)
    # prompt_images_paths = list(prompt_images_paths)
    # ### INFO: T3Bench alignment rendering script outputs 12 images per prompt.
    # assert len(prompt_images_paths) == 12
    prompt_images_paths = _load_rendering_images_paths(
        model=model,
        prompt=prompt,
        rootpath=source_rootpath,
    )

    #

    max_score, _, _ = _compute_clip_similarity(
        prompt=prompt,
        images_paths=prompt_images_paths,
        clip_tokenizer=clip_tokenizer,
        clip_processor=clip_processor,
        clip_model=clip_model,
    )

    print("  > SIMILARITY = ", max_score)

    scores_map[prompt] = round(float(max_score), 3)
    with open(out_scores_filepath, 'w', encoding="utf-8") as f:
        json.dump(scores_map, f, indent=4, ensure_ascii=False)


def _evaluate_clip_rprecision(
    model: str,
    positive_prompt: str,
    negative_prompts: List[str],
    source_rootpath: Path,
    out_rootpath: Path,
    clip_tokenizer: Any,
    clip_processor: Any,
    clip_model: Any,
    skip_existing: bool,
) -> None:
    assert isinstance(negative_prompts, list)
    assert len(negative_prompts) > 0
    assert all([isinstance(p, str) for p in negative_prompts])
    assert positive_prompt not in negative_prompts
    assert clip_tokenizer is not None
    assert clip_processor is not None
    assert clip_model is not None

    out_scores_filepath = Utils.Storage.build_clip_rprecision_scores_filepath(
        model=model,
        rootpath=out_rootpath,
        assert_exists=False,
    )
    out_scores_filepath.parent.mkdir(parents=True, exist_ok=True)

    scores_map: Dict[str, int] = None
    if not out_scores_filepath.exists():
        scores_map = {}
        out_scores_filepath.write_text("{}", encoding="utf-8")
    else:
        scores_map = json.loads(out_scores_filepath.read_text(encoding="UTF-8"))
        if skip_existing and positive_prompt in scores_map:
            _score = scores_map[positive_prompt]
            assert isinstance(_score, int)
            print("  > R-PRECISION (already exists) = ", _score)
            return _score

    assert isinstance(scores_map, dict)

    #

    positive_prompt_imgs_paths = _load_rendering_images_paths(
        model=model,
        prompt=positive_prompt,
        rootpath=source_rootpath,
    )

    positive_prompt_similarity, _, _ = _compute_clip_similarity(
        prompt=positive_prompt,
        images_paths=positive_prompt_imgs_paths,
        clip_tokenizer=clip_tokenizer,
        clip_processor=clip_processor,
        clip_model=clip_model,
    )

    #

    rprecision_score: int = 1

    ### INFO: alignment images taken from bottom view to remove.
    ###       (these images may lead to confusion while matching negative prompts)
    IMGS_NAMES_TO_EXCLUDE: List[str] = ["006", "007"]
    positive_prompt_imgs_paths = filter(
        lambda p: p.stem not in IMGS_NAMES_TO_EXCLUDE,
        positive_prompt_imgs_paths,
    )
    positive_prompt_imgs_paths = list(positive_prompt_imgs_paths)
    assert len(positive_prompt_imgs_paths) == 10

    for negative_prompt in negative_prompts:
        negative_prompt_similarity, _, _ = _compute_clip_similarity(
            prompt=negative_prompt,
            images_paths=positive_prompt_imgs_paths,  ### INFO: NOTICE THIS!
            clip_tokenizer=clip_tokenizer,
            clip_processor=clip_processor,
            clip_model=clip_model,
        )
        if negative_prompt_similarity > positive_prompt_similarity:
            print(f"  > R-PRECISION = negative prompt '{negative_prompt}' has greater score.")
            rprecision_score = 0
            break

    print("  > R-PRECISION = ", rprecision_score)

    scores_map[positive_prompt] = rprecision_score
    with open(out_scores_filepath, 'w', encoding="utf-8") as f:
        json.dump(scores_map, f, indent=4, ensure_ascii=False)


###


def main(
    model: str,
    prompt_filepath: Path,
    source_rootpath: Path,
    out_rootpath: Path,
    skip_existing: bool,
) -> None:
    assert isinstance(model, str)
    assert len(model) > 0
    assert model in Utils.Configs.MODELS_SUPPORTED
    assert isinstance(source_rootpath, Path)
    assert isinstance(out_rootpath, Path)
    assert isinstance(skip_existing, bool)

    if out_rootpath.exists():
        assert out_rootpath.is_dir()
    else:
        out_rootpath.mkdir(parents=True, exist_ok=True)

    #

    clip_tokenizer, clip_processor, clip_model = _load_models()

    prompts = Utils.Prompt.extract_from_file(filepath=prompt_filepath)

    print("")
    for prompt in prompts:
        if not isinstance(prompt, str) or len(prompt) < 2:
            continue

        print("")
        print(prompt)

        try:
            _evaluate_clip_similarity(
                model=model,
                prompt=prompt,
                source_rootpath=source_rootpath,
                out_rootpath=out_rootpath,
                clip_tokenizer=clip_tokenizer,
                clip_processor=clip_processor,
                clip_model=clip_model,
                skip_existing=skip_existing,
            )
        except Exception as e:  # pylint: disable=broad-except
            print("")
            print("")
            print("========================================")
            print("CLIP SIMILARITY")
            print("Error while running model -> ", model)
            print("Error while running prompt -> ", prompt)
            print(''.join(traceback.format_exception(type(e), e, e.__traceback__)))
            print("========================================")
            print("")
            print("")
            continue

        try:
            negative_prompts = deepcopy(prompts)
            assert prompt in negative_prompts
            negative_prompts.remove(prompt)
            assert prompt not in negative_prompts
            _evaluate_clip_rprecision(
                model=model,
                positive_prompt=prompt,
                negative_prompts=negative_prompts,
                source_rootpath=source_rootpath,
                out_rootpath=out_rootpath,
                clip_tokenizer=clip_tokenizer,
                clip_processor=clip_processor,
                clip_model=clip_model,
                skip_existing=skip_existing,
            )
        except Exception as e:  # pylint: disable=broad-except
            print("")
            print("")
            print("========================================")
            print("CLIP PRECISION")
            print("Error while running model -> ", model)
            print("Error while running prompt -> ", prompt)
            print(''.join(traceback.format_exception(type(e), e, e.__traceback__)))
            print("========================================")
            print("")
            print("")
            continue

        print("")
    print("")


###

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model',
        type=str,
        choices=Utils.Configs.MODELS_SUPPORTED,
        required=True,
    )
    parser.add_argument('--prompt-file', type=Path, required=True)
    parser.add_argument('--source-path', type=Path, required=True)
    parser.add_argument('--out-path', type=Path, required=True)
    parser.add_argument("--skip-existing", action="store_true", default=False)

    args = parser.parse_args()

    #

    main(
        model=args.model,
        prompt_filepath=args.prompt_file,
        source_rootpath=args.source_path,
        out_rootpath=args.out_path,
        skip_existing=args.skip_existing,
    )
