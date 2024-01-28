### pylint: disable=missing-function-docstring,missing-class-docstring,missing-module-docstring,wrong-import-order
from dotenv import load_dotenv

load_dotenv()  # take environment variables from .env.

import argparse
import os
import shutil
import openai
import backoff
import trimesh

from pathlib import Path
from tqdm import tqdm
from PIL import Image
from lavis.models import load_model_and_preprocess

from utils import Utils

###

device = Utils.Cuda.init()

OPENAI_API_TYPE = os.environ.get("OPENAI_API_TYPE")
OPENAI_KEY = os.environ.get("OPENAI_KEY")
OPENAI_ENDPOINT = os.environ.get("OPENAI_ENDPOINT")
OPENAI_DEPLOYMENT = os.environ.get("OPENAI_DEPLOYMENT")
OPENAI_MODEL = os.environ.get("OPENAI_MODEL")

assert isinstance(OPENAI_API_TYPE, str)
assert OPENAI_API_TYPE in ["openai", "azure"]
assert isinstance(OPENAI_KEY, str)
assert len(OPENAI_KEY) > 0
assert isinstance(OPENAI_ENDPOINT, str)
assert len(OPENAI_ENDPOINT) > 0
assert isinstance(OPENAI_MODEL, str)
assert OPENAI_MODEL in ["gpt-35-turbo", "gpt-4"]

openai.api_type = OPENAI_API_TYPE
openai.api_base = OPENAI_ENDPOINT
openai.api_key = OPENAI_KEY
openai.api_version = "2023-05-15"

###


def _run_mesh_rendering_script(
    model: str,
    prompt: str,
    source_rootpath: Path,
    out_rootpath: Path,
    skip_existing: bool,
) -> None:
    model_dirname = Utils.Storage.get_model_final_dirname_from_id(model)
    source_result_objmodel_path = Utils.Storage.build_result_final_export_obj_path(
        result_path=Utils.Storage.build_result_path_by_prompt(
            model_dirname=model_dirname,
            prompt=prompt,
            out_rootpath=source_rootpath,
            assert_exists=True,
        ),
        assert_exists=True,
    )

    out_prompt_renderings_path = Utils.Storage.build_renderings_path_by_prompt(
        prompt=prompt,
        out_rootpath=out_rootpath,
        eval_type="alignment",
        assert_exists=False,
    )

    if skip_existing and out_prompt_renderings_path.exists():
        print("")
        print("Renderings already exists --> ", out_prompt_renderings_path)
        print("")
        return

    if out_prompt_renderings_path.exists():
        shutil.rmtree(out_prompt_renderings_path)
    out_prompt_renderings_path.mkdir(parents=True, exist_ok=True)

    print("")
    print("")
    print(prompt)
    # print(source_result_objmodel_path)
    # print(out_prompt_renderings_path)
    ### TODO: improve this logic ...
    os.system(
        f'python render/meshrender_cap.py --path {str(source_result_objmodel_path)} --name {str(out_prompt_renderings_path)}'
    )
    print("")
    print("")


@backoff.on_exception(backoff.expo, (openai.error.RateLimitError, openai.error.Timeout, openai.error.APIError))
def predict(prompt, temperature):
    response: dict = None

    # messages = [{"role": "system", "content": prompt}]
    messages = [{"role": "user", "content": prompt}]

    if OPENAI_API_TYPE == "openai":
        response = openai.ChatCompletion.create(model=OPENAI_MODEL, temperature=temperature, messages=messages)
    elif OPENAI_API_TYPE == "azure":
        # engine = "deployment_name".
        response = openai.ChatCompletion.create(engine=OPENAI_DEPLOYMENT, temperature=temperature, messages=[messages])

    assert response is not None

    return response["choices"][0]["message"]["content"]


def _caption_renderings(model: str, prompt: str, out_rootpath: Path) -> None:
    out_prompt_renderings_path = Utils.Storage.build_renderings_path_by_prompt(
        prompt=prompt,
        out_rootpath=out_rootpath,
        eval_type="alignment",
        assert_exists=True,
    )
    out_prompt_renderings_uri = str(out_prompt_renderings_path)

    #

    model, vis_processors, _ = load_model_and_preprocess(
        name='blip2_t5',
        model_type='pretrain_flant5xl',
        is_eval=True,
        device=device,
    )

    radius = 2.2
    icosphere = trimesh.creation.icosphere(subdivisions=0)
    icosphere.vertices *= radius

    texts = []
    for idx, img_path in enumerate(os.listdir(out_prompt_renderings_path)):
        color = Image.open(os.path.join(out_prompt_renderings_uri, img_path)).convert("RGB")
        image = vis_processors["eval"](color).unsqueeze(0).to(device)
        x = model.generate({"image": image}, use_nucleus_sampling=True, num_captions=1)
        texts += x

    prompt_input = 'Given a set of descriptions about the same 3D object, distill these descriptions into one concise caption. The descriptions are as follows:\n\n'
    for idx, txt in enumerate(texts):
        prompt_input += f'view{idx+1}: '
        prompt_input += txt
        prompt_input += '\n'
    prompt_input += '\nAvoid describing background, surface, and posture. The caption should be:'
    res = predict(prompt_input, 0)
    print(res)

    # with open(f'outputs_caption/{args.method}_{args.group}.txt', 'a+') as f:
    #     f.write(prompt + ':' + res + '\n')


def _evaluate_alignment(model: str, prompt: str, out_rootpath: Path) -> None:
    pass


###


def main(
    model: str,
    prompt_filepath: Path,
    source_rootpath: Path,
    out_rootpath: Path,
    skip_existing_renderings: bool,
) -> None:
    assert isinstance(model, str)
    assert len(model) > 0
    assert model in Utils.Configs.MODELS_SUPPORTED
    assert isinstance(source_rootpath, Path)
    assert isinstance(out_rootpath, Path)
    # assert out_rootpath.exists()
    # assert out_rootpath.is_dir()
    assert isinstance(skip_existing_renderings, bool)

    if out_rootpath.exists():
        assert out_rootpath.is_dir()
    else:
        out_rootpath.mkdir(parents=True, exist_ok=True)

    out_alignment_scores_filepath = Utils.Storage.build_prompt_alignment_scores_filepath(
        out_rootpath=out_rootpath,
        assert_exists=False,
    )
    out_alignment_scores_filepath.parent.mkdir(parents=True, exist_ok=True)
    out_alignment_scores_filepath.write_text("", encoding="utf-8")

    #

    prompts = Utils.Prompt.extract_from_file(filepath=prompt_filepath)

    for prompt in prompts:
        if not isinstance(prompt, str) or len(prompt) < 2:
            continue

        _run_mesh_rendering_script(
            model=model,
            prompt=prompt,
            source_rootpath=source_rootpath,
            out_rootpath=out_rootpath,
            skip_existing=skip_existing_renderings,
        )

        _caption_renderings(model=model, prompt=prompt, out_rootpath=out_rootpath)

        # _evaluate_alignment(model=model, prompt=prompt, out_rootpath=out_rootpath)


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
    parser.add_argument("--skip-existing-renderings", action="store_true", default=False)

    args = parser.parse_args()

    #

    main(
        model=args.model,
        prompt_filepath=args.prompt_file,
        source_rootpath=args.source_path,
        out_rootpath=args.out_path,
        skip_existing_renderings=args.skip_existing_renderings,
    )
