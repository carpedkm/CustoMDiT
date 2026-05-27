import os

from .utils import get_prompt_from_filename, init_submodules, save_json, load_json
import importlib
from itertools import chain
from pathlib import Path

from .distributed import get_rank, print0
import json

class Evaluator(object):
    def __init__(self, device, output_path):
        self.device = device                        # cuda or cpu
        self.output_path = output_path              # output directory to save VBench results
        os.makedirs(self.output_path, exist_ok=True)

    def build_full_dimension_list(self, ):
        return ["motion_smoothness", "dynamic_degree","clip_text", "clip_image", "regional_clip_image", "dino_image", "regional_dino_image", "regional_clip_text"]

    def check_dimension_requires_extra_info(self, dimension_list):
        dim_custom_not_supported = set(dimension_list) & set([
            'object_class', 'multiple_objects', 'scene', 'appearance_style', 'color', 'spatial_relationship'
        ])

        assert len(dim_custom_not_supported) == 0, f"dimensions : {dim_custom_not_supported} not supported for custom input"

    def build_custom_image_dict(self, directory):
        image_dict = {}

        for filename in os.listdir(directory):
            file_path = os.path.join(directory, filename)

            if os.path.isfile(file_path):
                image_name, extension = os.path.splitext(filename)
                extension = extension.lower()

                if extension in ['.jpg', '.jpeg', '.png']:
                    image_dict[image_name] = file_path

        return image_dict

    def evaluate(self, json_path, name, dimension_list=None, local=False, read_frame=False, **kwargs):
        results_dict = {}
        if dimension_list is None:
            dimension_list = self.build_full_dimension_list()
        submodules_dict = init_submodules(dimension_list, local=local, read_frame=read_frame)

        with open(json_path, 'r') as f:
            data_json = json.load(f)
        video_list = data_json
        for dimension in dimension_list:
            try:
                dimension_module = importlib.import_module(f'evaluator.{dimension}')
                evaluate_func = getattr(dimension_module, f'compute_{dimension}')
            except Exception as e:
                raise NotImplementedError(f'UnImplemented dimension {dimension}!, {e}')
            submodules_list = submodules_dict[dimension] if dimension not in ["clip_text", "clip_image", "regional_clip_image", "dino_image", "regional_dino_image", "regional_clip_text"] else []
            results = evaluate_func(video_list, self.device, submodules_list, **kwargs)
            results_dict[dimension] = results
        output_name = os.path.join(self.output_path, name+'_eval_results.json')
        if get_rank() == 0:
            save_json(results_dict, output_name)
            print0(f'Evaluation results saved to {output_name}')
