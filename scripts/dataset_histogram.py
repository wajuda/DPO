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
import numpy as np
import matplotlib.pyplot as plt

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

def get_histogram(scores, bins_num, min, max):
    bin_edges = np.linspace(min, max, bins_num+1)
    bin_index = np.digitize(scores, bin_edges, right = False)-1
    bin_index[bin_index == bins_num] = bins_num-1
    bin_index[bin_index == -1] =0
    histogram = np.bincount(bin_index, minlength = bins_num)
    return histogram / len(scores), np.mean(scores), np.var(scores)

def plot_histogram(better_score, worse_score, bins_num=200, min_val=None, max_val=None):
    # Get min and max values from the scores if not provided
    if min_val is None:
        min_val = min(min(better_score), min(worse_score))
    if max_val is None:
        max_val = max(max(better_score), max(worse_score))

    # Calculate histograms, means, and variances
    better_hist, better_mean, better_var = get_histogram(better_score, bins_num, min_val, max_val)
    worse_hist, worse_mean, worse_var = get_histogram(worse_score, bins_num, min_val, max_val)
    
    # Plot histograms
    plt.figure(figsize=(10, 6))
    plt.hist(better_score, bins=np.linspace(min_val, max_val, bins_num + 1), alpha=0.5, label=f'Better Score\nMean: {better_mean:.2f}, Var: {better_var:.2f}', edgecolor='blue', color='blue')
    plt.hist(worse_score, bins=np.linspace(min_val, max_val, bins_num + 1), alpha=0.5, label=f'Worse Score\nMean: {worse_mean:.2f}, Var: {worse_var:.2f}', edgecolor='red', color='red')
    
    # Labels and title
    plt.xlabel('Scores')
    plt.ylabel('Frequency')
    plt.title('Histogram of Better vs Worse Scores of Original')
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.savefig("/home/pku0072/align-anything/output/better_worse_original.png")

def plot_diff(better_score, worse_score, bins_num=200, min_val=None, max_val=None):
    # Get min and max values from the scores if not provided
    diff = better_score - worse_score
    if min_val is None:
        min_val = min(diff)
    if max_val is None:
        max_val = max(diff)

    
    # Plot histograms
    plt.figure(figsize=(10, 6))
    plt.hist(diff, bins=np.linspace(min_val, max_val, bins_num + 1), alpha=0.5, label=f'Score (Better - Worse)\nMean: {np.mean(diff):.2f}, Var: {np.var(diff):.2f}', edgecolor='blue', color='blue')
    
    
    # Labels and title
    plt.xlabel('Scores (Better - Worse)')
    plt.ylabel('Frequency')
    plt.title('Histogram of (Better - Worse) Scores of Original')
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.savefig("/home/pku0072/align-anything/output/better_worse_diff_original.png")

def main():
    
    #better_dir = "/home/pku0072/align-anything/output/rm_score_dpo/eval_data_with_score.json"
    #worse_dir = "/home/pku0072/align-anything/output/rm_score/eval_data_with_score.json"
    better_dir = "/home/pku0072/align-anything/output/Qwen_score_prefer_better/eval_data_with_score.json"
    worse_dir = "/home/pku0072/align-anything/output/Qwen_score_prefer_worse/eval_data_with_score.json"
    better_score = []
    worse_score = []
    with open(better_dir,'r') as f:
        text = json.load(f)
        for item in text:
            better_score.append(item['score'])
    with open(worse_dir,'r') as f:
        text = json.load(f)
        for item in text:
            worse_score.append(item['score'])
               
    print(len(better_score))
    print(len(worse_score))
    print(better_score[0])
    print(np.mean(better_score))
    print(worse_score[0])
    print(np.mean(worse_score))
    plot_histogram(better_score, worse_score)
    plot_diff(np.array(better_score),np.array(worse_score))



if __name__ == '__main__':
    sys.exit(main())
