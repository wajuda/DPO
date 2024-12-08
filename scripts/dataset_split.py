# Copyright 2024 PKU-Alignment Team. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Trainer for reward model training."""


import argparse
import os
import sys
from typing import Any
import json

def format_preference_sample(raw_sample: dict[str, Any]) -> dict[str, Any]:
        metrics = raw_sample['better_response_id']
        better_response = raw_sample[f'response_{int(metrics)}']
        worse_response = raw_sample[f'response_{1-int(metrics)}']
        prompt = raw_sample['prompt']
        #assert metrics in [0,1] and better_response and worse_response and prompt, f'invalid text: {raw_sample} prompt :{prompt},metrics:{metrics}, better_response:{better_response},worse_response:{worse_response}'
        if metrics in [0,1] and better_response and worse_response and prompt:
            return {
            'prompt': prompt,
            'answer': better_response
            }, {'prompt': prompt,
            'answer': worse_response}
        else:
            return 0

def main():
    path = "../../datasets/PKU-SafeRLHF-single-dimension/data"
    better_dir = "../generate_scripts/test/prefer_better1.json"
    worse_dir = "../generate_scripts/test/prefer_worse1.json"
    better_json = []
    worse_json = []
    dirs = os.listdir(path)
    train_num =0
    test_num=0
    for item in dirs:
        data_path = os.path.join(path , item)
        train_data = os.path.join(data_path, 'train.jsonl')
        test_data = os.path.join(data_path, 'test.jsonl')
        with open(train_data,'r') as f:
            for line in f:
                ans = format_preference_sample(json.loads(line))
                if ans !=0:
                    better_data, worse_data = format_preference_sample(json.loads(line))
                    better_json.append(better_data)
                    worse_json.append(worse_data)
                    train_num +=1

        with open(test_data,'r') as f:
            for line in f:
                ans = format_preference_sample(json.loads(line))
                if ans !=0:
                    better_data, worse_data = format_preference_sample(json.loads(line))
                    better_json.append(better_data)
                    worse_json.append(worse_data)
                    test_num +=1
        print(train_num,test_num)
    print(len(better_json))
    print(len(worse_json))
    print(better_json[0])
    print(worse_json[0])
    with open(better_dir, 'w') as outfile:
        json.dump(better_json, outfile, indent=4)
    with open(worse_dir, 'w') as outfile:
        json.dump(worse_json, outfile, indent=4)


'''import deepspeed
import torch
import torch.distributed as dist
import torch.nn.functional as F
from tqdm import tqdm
from transformers.integrations.deepspeed import HfDeepSpeedConfig

from align_anything.datasets.text_to_text.preference import PreferenceBatch, PreferenceDataset
from align_anything.models.pretrained_model_with_value import load_pretrained_model_with_value_head
from align_anything.trainers.base import SupervisedTrainerBase
from align_anything.utils.multi_process import (
    get_all_reduce_mean,
    get_current_device,
    is_main_process,
)
from align_anything.utils.tools import (
    custom_cfgs_to_dict,
    dict_to_namedtuple,
    prepare_ds_train_cfgs,
    read_cfgs,
    seed_everything,
    split_prompt_response,
    update_dict,
)


class Split(SupervisedTrainerBase):

    def __init__(self, cfgs, ds_cfgs) -> None:
        """Initialize the reward model trainer."""
        self.cfgs = cfgs
        self.ds_train_cfgs = prepare_ds_train_cfgs(custom_cfgs=cfgs.train_cfgs, raw_ds_cfgs=ds_cfgs)
        self.global_step = 0
        self.init_models()
        self.init_datasets()
        

    def init_models(self) -> None:
        """Initialize model and tokenizer."""
        if self.ds_train_cfgs is not None and self.ds_train_cfgs['zero_optimization']['stage'] == 3:
            self.dstchf = HfDeepSpeedConfig(self.ds_train_cfgs)
        self.model, self.tokenizer, self.processor = load_pretrained_model_with_value_head(
            self.cfgs.model_cfgs.model_name_or_path,
            model_max_length=self.cfgs.model_cfgs.model_max_length,
            padding_side='right',
            trust_remote_code=self.cfgs.train_cfgs.trust_remote_code,
            modality='text',
        )

    def init_datasets(self) -> None:
        """Initialize training and evaluation datasets."""
        self.train_dataloader, self.eval_dataloader = self.get_dataloaders(
            PreferenceDataset, PreferenceDataset
        )


    def split(self) -> None:
        eval_dataloader = tqdm(self.eval_dataloader)
        for batch in eval_dataloader:
            print(batch)
            break

    def save(
        self,
        model: deepspeed.DeepSpeedEngine | None = None,
        tag: int | None = None,
    ) -> None:
        """Save model and tokenizer in Hugging Face format."""
        self.save_transformers(model=model, tag=tag)


def main():
    # setup distribution training
    deepspeed.init_distributed()
    current_device = get_current_device()
    torch.cuda.set_device(current_device)

    # read default configs from the yaml file
    task = os.path.join('text_to_text', 'rm')
    dict_cfgs, ds_cfgs = read_cfgs(mode='train', task=task)

    # get custom configs from command line
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    _, unparsed_args = parser.parse_known_args()
    keys = [k[2:] for k in unparsed_args[1::2]]
    values = list(unparsed_args[2::2])
    unparsed_args = dict(zip(keys, values))
    for k, v in unparsed_args.items():
        dict_cfgs = update_dict(dict_cfgs, custom_cfgs_to_dict(k, v))

    # setup training
    cfgs = dict_to_namedtuple(dict_cfgs)
    seed_everything(cfgs.train_cfgs.seed)

    # finetune the model
    trainer = Split(cfgs=cfgs, ds_cfgs=ds_cfgs)
    trainer.split()'''

if __name__ == '__main__':
    sys.exit(main())
