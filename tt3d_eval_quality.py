### pylint: disable=missing-function-docstring,missing-class-docstring,missing-module-docstring,wrong-import-order
import argparse
import os
import shutil

from pathlib import Path

from utils import Utils

###

device = Utils.Cuda.init()

###


def _evaluate_quality_of_generated_obj(model: str, prompt: str, source_rootpath: Path, out_rootpath: Path):
    model_dirname = Utils.Storage.get_model_final_dirname_from_id(model)

    source_result_path = Utils.Storage.build_result_path_by_prompt(
        model_dirname=model_dirname,
        prompt=prompt,
        out_rootpath=source_rootpath,
        assert_exists=True,
    )

    source_result_objmodel_path = Utils.Storage.build_result_final_export_obj_path(
        result_path=source_result_path,
        assert_exists=True,
    )

    out_prompt_renderings_path = Utils.Storage.build_renderings_path_by_prompt(
        prompt=prompt,
        out_rootpath=out_rootpath,
        assert_exists=False,
    )

    ### TODO: improve this logic with a {skip_existing} flag ...
    shutil.rmtree(out_prompt_renderings_path)
    out_prompt_renderings_path.mkdir(parents=True, exist_ok=True)

    print("")
    print("")
    print(source_result_objmodel_path)
    print(out_prompt_renderings_path)
    print("")
    print("")

    ### TODO: improve this logic ...
    os.system(
        f'python render/meshrender.py --path {str(source_result_objmodel_path)} --name {str(out_prompt_renderings_path)}'
    )


###


def main(model: str, prompt_filepath: Path, source_rootpath: Path, out_rootpath: Path):
    assert isinstance(model, str)
    assert len(model) > 0
    assert model in Utils.Configs.MODELS_SUPPORTED
    assert isinstance(source_rootpath, Path)
    assert isinstance(out_rootpath, Path)
    assert out_rootpath.exists()
    assert out_rootpath.is_dir()

    out_rootpath.mkdir(parents=True, exist_ok=True)

    prompts = Utils.Prompt.extract_from_file(filepath=prompt_filepath)

    for prompt in prompts:
        if not isinstance(prompt, str) or len(prompt) < 2:
            continue

        _evaluate_quality_of_generated_obj(
            model=model,
            prompt=prompt,
            source_rootpath=source_rootpath,
            out_rootpath=out_rootpath,
        )


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

    args = parser.parse_args()

    #

    main(
        model=args.model,
        prompt_filepath=args.prompt_file,
        source_rootpath=args.source_path,
        out_rootpath=args.out_path,
    )
