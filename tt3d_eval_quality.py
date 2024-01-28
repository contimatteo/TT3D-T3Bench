### pylint: disable=missing-function-docstring,missing-class-docstring,missing-module-docstring,wrong-import-order
import argparse
import os
import shutil
import ImageReward as RM
import trimesh
from PIL import Image

from pathlib import Path

from utils import Utils

###

device = Utils.Cuda.init()

###


def _run_mesh_rendering_script(
    model: str,
    prompt: str,
    source_rootpath: Path,
    out_rootpath: Path,
    skip_existing: bool,
) -> None:
    model_dirname = Utils.Storage.get_model_final_dirname_from_id(model)
    # source_result_path = Utils.Storage.build_result_path_by_prompt(
    #     model_dirname=model_dirname,
    #     prompt=prompt,
    #     out_rootpath=source_rootpath,
    #     assert_exists=True,
    # )
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
        eval_type="quality",
        assert_exists=False,
    )

    if skip_existing and out_prompt_renderings_path.exists():
        return

    if out_prompt_renderings_path.exists():
        shutil.rmtree(out_prompt_renderings_path)
    out_prompt_renderings_path.mkdir(parents=True, exist_ok=True)

    ### TODO: improve this logic ...
    os.system(
        f'python render/meshrender.py --path {str(source_result_objmodel_path)} --name {str(out_prompt_renderings_path)}'
    )


def _evaluate_quality(model: str, prompt: str, out_rootpath: Path) -> None:
    out_prompt_renderings_path = Utils.Storage.build_renderings_path_by_prompt(
        prompt=prompt,
        out_rootpath=out_rootpath,
        eval_type="quality",
        assert_exists=True,
    )
    out_prompt_renderings_uri = str(out_prompt_renderings_path)

    #

    RADIUS = 2.2  ### pylint: disable=invalid-name
    model = RM.load("ImageReward-v1.0")
    icosphere = trimesh.creation.icosphere(subdivisions=2, radius=RADIUS)

    scores = {i: -114514 for i in range(len(icosphere.vertices))}

    #

    for idx in range(len(icosphere.vertices)):
        for j in range(5):
            img_path = f'{idx:03d}_{j}.png'
            # convert color to PIL image
            color = Image.open(os.path.join(out_prompt_renderings_uri, img_path))
            reward = model.score(prompt, color)
            scores[idx] = max(scores[idx], reward)

    # convolute scores on the icosphere for 3 times
    for _ in range(3):
        new_scores = {}
        for idx, v in enumerate(icosphere.vertices):
            new_scores[idx] = scores[idx]
            for n in icosphere.vertex_neighbors[idx]:
                new_scores[idx] += scores[n]
            new_scores[idx] /= (len(icosphere.vertex_neighbors[idx]) + 1)
        scores = new_scores

    #

    out_quality_scores_filepath = Utils.Storage.build_quality_scores_filepath(
        out_rootpath=out_rootpath,
        assert_exists=True,
    )

    for idx in sorted(scores, key=lambda x: scores[x], reverse=True)[:1]:
        _score = scores[idx] * 20 + 50
        # mean_score += _score / len(lines)
        # print(_score)

        ### TODO: improve this logic -> convert the ".txt" to a pandas dataframe.
        with open(str(out_quality_scores_filepath), 'a+', encoding="utf-8") as f:
            f.write(f'{_score:.1f}\t\t{prompt}\n')


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

    out_quality_scores_filepath = Utils.Storage.build_quality_scores_filepath(
        out_rootpath=out_rootpath,
        assert_exists=False,
    )
    out_quality_scores_filepath.parent.mkdir(parents=True, exist_ok=True)
    out_quality_scores_filepath.write_text("", encoding="utf-8")

    #

    prompts = Utils.Prompt.extract_from_file(filepath=prompt_filepath)

    for prompt in prompts:
        if not isinstance(prompt, str) or len(prompt) < 2:
            continue

        print("")
        print("")
        print(prompt)

        _run_mesh_rendering_script(
            model=model,
            prompt=prompt,
            source_rootpath=source_rootpath,
            out_rootpath=out_rootpath,
            skip_existing=skip_existing_renderings,
        )

        _evaluate_quality(model=model, prompt=prompt, out_rootpath=out_rootpath)

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
