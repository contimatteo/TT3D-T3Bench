### pylint: disable=missing-function-docstring,missing-class-docstring,missing-module-docstring,wrong-import-order
from typing import Tuple, List, Callable, Literal

import os
import torch

from pathlib import Path
from datetime import datetime

###


class _Cuda():

    @staticmethod
    def is_available() -> bool:
        _cuda = torch.cuda.is_available()
        _cudnn = torch.backends.cudnn.enabled
        return _cuda and _cudnn

    @classmethod
    def device(cls) -> torch.cuda.device:
        assert cls.is_available()
        return torch.device('cuda')

    @classmethod
    def count_devices(cls) -> int:
        assert cls.is_available()
        return torch.cuda.device_count()

    @classmethod
    def get_current_device_info(cls) -> Tuple[int, str]:
        _idx = torch.cuda.current_device()
        _name = torch.cuda.get_device_name(_idx)
        return _idx, _name

    @staticmethod
    def get_visible_devices_param() -> str:
        return os.environ["CUDA_VISIBLE_DEVICES"]

    @classmethod
    def init(cls) -> torch.cuda.device:
        """
        We run all the experiments on server which have 4 different GPUs.
        Unfortunately, we cannot use all of them at the same time, since many other people are 
        using the server. Therefore, we have to specify which GPU we want to use.
        In particular, we have to use the GPU #1 (Nvidia RTX-3090).
        In order to avoid naive mistakes, we also check that the {CUDA_VISIBLE_DEVICES} environment 
        variable is set.
        """
        assert cls.is_available()
        assert isinstance(cls.get_visible_devices_param(), str)
        # assert cls.get_visible_devices_param() == "1"
        assert cls.count_devices() == 1

        device_idx, _ = cls.get_current_device_info()
        assert device_idx == 0

        return cls.device()


###


class _Prompt():

    ENCODING_CHAR: str = "_"

    @classmethod
    def encode(cls, prompt: str) -> str:
        assert isinstance(prompt, str)
        assert len(prompt) > 0
        prompt = prompt.strip()
        prompt = prompt.replace(" ", cls.ENCODING_CHAR)
        return prompt

    @classmethod
    def decode(cls, prompt: str) -> str:
        assert isinstance(prompt, str)
        assert len(prompt) > 0
        prompt = prompt.strip()
        prompt = prompt.replace(cls.ENCODING_CHAR, " ")
        return prompt

    @staticmethod
    def extract_from_file(filepath: Path) -> List[str]:
        assert isinstance(filepath, Path)
        assert filepath.exists()
        assert filepath.is_file()
        assert filepath.suffix == ".txt"

        with open(filepath, "r", encoding="utf-8") as f:
            prompts = f.readlines()

        prompts = map(lambda p: p.strip(), prompts)
        prompts = filter(lambda p: len(p) > 1, prompts)
        ### TODO: filter out prompts with special chars ...
        prompts = list(prompts)

        return prompts


###


class _Configs():
    MODELS_SUPPORTED: List[str] = [
        "point-e",
        "shap-e",
        "cap3d-point-e",
        "cap3d-shap-e",
        "dreamfusion-sd",
        "dreamfusion-if",
        "fantasia3d",
        "prolificdreamer",
        "magic3d-sd",
        "magic3d-if",
        "textmesh-sd",
        "textmesh-if",
        "hifa",
        "luciddreamer",
    ]


###


class _Storage():

    @staticmethod
    def build_model_path(
        model: str,
        out_rootpath: Path,
        assert_exists: bool,
    ) -> Path:
        assert model in _Configs.MODELS_SUPPORTED

        model_dirname = _Storage.get_model_final_dirname_from_id(model=model)
        out_path = out_rootpath.joinpath(model_dirname)

        if assert_exists:
            assert out_path.exists()
            assert out_path.is_dir()

        return out_path

    @classmethod
    def build_result_path(
        cls,
        # model_dirname: str,
        model: str,
        prompt: str,
        out_rootpath: Path,
        assert_exists: bool,
    ) -> Path:
        assert "_" not in prompt

        out_model_path = cls.build_model_path(
            model=model,
            out_rootpath=out_rootpath,
            assert_exists=False,
        )
        prompt_enc = Utils.Prompt.encode(prompt=prompt)
        out_path = out_model_path.joinpath(prompt_enc)

        if assert_exists:
            assert out_path.exists()
            assert out_path.is_dir()

        return out_path

    @classmethod
    def build_result_export_path(
        cls,
        model: str,
        prompt: str,
        out_rootpath: Path,
        assert_exists: bool,
    ) -> Path:
        result_path = cls.build_result_path(
            model=model,
            prompt=prompt,
            out_rootpath=out_rootpath,
            assert_exists=False,
        )

        #

        result_save_path = result_path.joinpath("save")

        export_candidate_paths: List[Path] = []
        # for export_candidate_path in result_save_path.glob("it*-export"):
        for export_candidate_path in result_save_path.glob("*export"):
            if not export_candidate_path.is_dir():
                continue
            export_candidate_paths.append(export_candidate_path)

        ### INFO: currently we expect to have only one export path.
        ### Originally, threestudio supports multiple exports, but we do not handle them.
        ### At the end, we want to have just one export path, but we need to search for it since
        ### the name od the directory depends on the number of iterations performed during training.
        assert len(export_candidate_paths) == 1

        #

        out_path = export_candidate_paths[0]

        if assert_exists:
            assert out_path.exists()
            assert out_path.is_dir()

        return out_path

    @classmethod
    def build_result_export_obj_path(
        cls,
        model: str,
        prompt: str,
        out_rootpath: Path,
        assert_exists: bool,
    ) -> Path:
        result_export_path = cls.build_result_export_path(
            model=model,
            prompt=prompt,
            out_rootpath=out_rootpath,
            assert_exists=False,
        )

        result_export_obj_path = result_export_path.joinpath("model.obj")

        if assert_exists:
            assert result_export_obj_path.exists()
            assert result_export_obj_path.is_file()

        return result_export_obj_path

    #

    @staticmethod
    def build_renderings_path(
        model: str,
        prompt: str,
        out_rootpath: Path,
        eval_type: Literal["quality", "alignment"],
        assert_exists: bool,
    ) -> Path:
        assert model in _Configs.MODELS_SUPPORTED
        assert "_" not in prompt
        assert isinstance(eval_type, str)
        assert eval_type in ["quality", "alignment"]

        out_renderings_path = out_rootpath.joinpath(model, "renderings", eval_type)
        prompt_enc = Utils.Prompt.encode(prompt=prompt)
        out_path = out_renderings_path.joinpath(prompt_enc)

        if assert_exists:
            assert out_path.exists()
            assert out_path.is_dir()

        return out_path

    @classmethod
    def build_prompt_alignment_caption_filepath(
        cls,
        model: str,
        prompt: str,
        out_rootpath: Path,
        assert_exists: bool,
    ) -> Path:
        assert model in _Configs.MODELS_SUPPORTED
        assert "_" not in prompt

        out_captions_path = out_rootpath.joinpath(model, "captions", "alignment")
        prompt_enc = Utils.Prompt.encode(prompt=prompt)
        out_filepath = out_captions_path.joinpath(f"{prompt_enc}.txt")

        if assert_exists:
            assert out_filepath.exists()
            assert out_filepath.is_file()

        return out_filepath

    @classmethod
    def build_quality_scores_filepath(
        cls,
        model: str,
        out_rootpath: Path,
        assert_exists: bool,
    ) -> Path:
        assert model in _Configs.MODELS_SUPPORTED

        out_scores_path = out_rootpath.joinpath(model, "scores")
        out_filepath = out_scores_path.joinpath("quality.json")

        if assert_exists:
            assert out_filepath.exists()
            assert out_filepath.is_file()

        return out_filepath

    @classmethod
    def build_alignment_scores_filepath(
        cls,
        model: str,
        out_rootpath: Path,
        assert_exists: bool,
    ) -> Path:
        assert model in _Configs.MODELS_SUPPORTED

        out_scores_path = out_rootpath.joinpath(model, "scores")
        out_alignment_scores_filepath = out_scores_path.joinpath("alignment.json")

        if assert_exists:
            assert out_alignment_scores_filepath.exists()
            assert out_alignment_scores_filepath.is_file()

        return out_alignment_scores_filepath

    #

    @classmethod
    def build_clip_scores_path(
        cls,
        model: str,
        rootpath: Path,
        assert_exists: bool,
    ) -> Path:
        assert model in _Configs.MODELS_SUPPORTED
        out_path = rootpath.joinpath(model, "scores", "clip")
        if assert_exists:
            assert out_path.exists()
            assert out_path.is_file()
        return out_path

    @classmethod
    def build_clip_similarity_scores_filepath(
        cls,
        model: str,
        rootpath: Path,
        assert_exists: bool,
    ) -> Path:
        assert model in _Configs.MODELS_SUPPORTED
        out_clip_scores_path = cls.build_clip_scores_path(
            model=model,
            rootpath=rootpath,
            assert_exists=False,
        )
        out_filepath = out_clip_scores_path.joinpath("similarity.json")
        if assert_exists:
            assert out_filepath.exists()
            assert out_filepath.is_file()
        return out_filepath

    @classmethod
    def build_clip_rprecision_scores_filepath(
        cls,
        model: str,
        rootpath: Path,
        assert_exists: bool,
    ) -> Path:
        assert model in _Configs.MODELS_SUPPORTED
        out_clip_scores_path = cls.build_clip_scores_path(
            model=model,
            rootpath=rootpath,
            assert_exists=False,
        )
        out_filepath = out_clip_scores_path.joinpath("rprecision.json")
        if assert_exists:
            assert out_filepath.exists()
            assert out_filepath.is_file()
        return out_filepath

    #

    @staticmethod
    def get_model_final_dirname_from_id(model: str) -> str:
        assert isinstance(model, str)
        assert len(model) > 0
        assert model in Utils.Configs.MODELS_SUPPORTED

        if model == "shap-e":
            return "shap-e"
        if model == "point-e":
            return "point-e"

        if model == "cap3d-shap-e":
            return "cap3d-shap-e"
        if model == "cap3d-point-e":
            return "cap3d-point-e"

        if model == "dreamfusion-sd":
            return "dreamfusion-sd"
        if model == "dreamfusion-if":
            return "dreamfusion-if"

        if model == "fantasia3d":
            return "fantasia3d-texture"

        if model == "prolificdreamer":
            return "prolificdreamer-texture"

        if model == "magic3d-sd":
            return "magic3d-refine-sd"
        if model == "magic3d-if":
            return "magic3d-refine-if"

        if model == "textmesh-sd":
            return "textmesh-sd"
        if model == "textmesh-if":
            return "textmesh-if"

        if model == "hifa":
            return "hifa"

        if model == "luciddreamer":
            return "luciddreamer"

        raise Exception("Model output final dirname not configured.")


###


class Utils():

    Configs = _Configs
    Cuda = _Cuda
    Prompt = _Prompt
    Storage = _Storage
