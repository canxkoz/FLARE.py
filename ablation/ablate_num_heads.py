#
import os
import time
import shutil
import subprocess
import json, yaml
from tqdm import tqdm

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.colors import LogNorm

import numpy as np
import pandas as pd
import argparse
import seaborn as sns

# local
import utils

#======================================================================#
PROJDIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
CASEDIR = os.path.join(PROJDIR, 'out', 'pdebench')
os.makedirs(CASEDIR, exist_ok=True)

#======================================================================#
def collect_data(dataset: str):
    data_dir = os.path.join(CASEDIR, f'abl_num_heads_{dataset}')

    # Initialize empty dataframe
    df = pd.DataFrame()

    # Check if case directory exists
    if os.path.exists(data_dir):
        # Get all subdirectories (each represents a case)
        cases = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]

        for case in cases:
            case_path = os.path.join(data_dir, case)
            
            if not os.path.exists(os.path.join(case_path, 'config.yaml')):
                continue
            if not os.path.exists(os.path.join(case_path, 'num_params.txt')):
                continue

            # Initialize case data dictionary
            case_data = {}
            
            # Check for and load relative error data
            rel_error_path = os.path.join(case_path, 'ckpt10', 'rel_error.json')
            if os.path.exists(rel_error_path):
                with open(rel_error_path, 'r') as f:
                    rel_error = json.load(f)
                case_data.update({
                    'train_rel_error': rel_error.get('train_rel_error'),
                    'test_rel_error': rel_error.get('test_rel_error')
                })
            
            # Load config data
            config_path = os.path.join(case_path, 'config.yaml')
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)
                case_data.update({
                    'channel_dim': config.get('channel_dim'),
                    'num_latents': config.get('num_latents'),
                    'num_blocks': config.get('num_blocks'),
                    'num_heads': config.get('num_heads'),
                    'num_layers_kv_proj': config.get('num_layers_kv_proj'),
                    'num_layers_mlp': config.get('num_layers_mlp'),
                    'num_layers_in_out_proj': config.get('num_layers_in_out_proj'),
                    'seed': config.get('seed'),
                })

            # Load num_params
            num_params_path = os.path.join(case_path, 'num_params.txt')
            if os.path.exists(num_params_path):
                with open(num_params_path, 'r') as f:
                    num_params = int(f.read().strip())
                case_data.update({'num_params': num_params})
            
            # Add case data to dataframe
            df = pd.concat([df, pd.DataFrame([case_data])], ignore_index=True)

            df['head_dim'] = df['channel_dim'] // df['num_heads']

        print(f"Collected {len(df)} cases for {dataset} dataset.")

    return df

def plot_results(dataset: str, df: pd.DataFrame):

    #---------------------------------------------------------#
    df = df.groupby(['num_latents', 'num_heads']).mean().reset_index()

    configs = df[['num_latents',]].drop_duplicates()
    print(f"Found {len(configs)} unique configurations for num_heads lineplot.")
    
    num_latents_list = configs['num_latents'].unique().tolist()

    #---------------------------------------------------------#
    # LINEPLOT of test error vs num_heads
    #---------------------------------------------------------

    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
        "text.latex.preamble": r"\usepackage{amsmath}"
    })

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    fontsize = 28

    ax.set_ylabel(r'Test relative error', fontsize=fontsize)

    ax.set_xscale('log')
    ax.set_yscale('log', base=10)
    ax.grid(True, which="both", ls="-", alpha=0.5)
    ax.set_ylim(3e-3, 7e-3)

    ax.set_yticks([3e-3, 4e-3, 5e-3, 6e-3, 7e-3])
    ax.set_yticklabels(['3e-3', '4e-3', '5e-3', '6e-3', '7e-3'])

    #--------------#
    # ax.set_xlabel(r'\# heads', fontsize=fontsize)
    # heads = [1, 2, 4, 8, 16, 32, 64]
    # ax.set_xticks(heads)
    # ax.set_xticklabels(heads)
    #--------------#
    ax.set_xlabel(r'Head dimension ($D$)', fontsize=fontsize)
    head_dims = [1, 2, 4, 8, 16, 32, 64]
    ax.set_xticks(head_dims)
    ax.set_xticklabels(head_dims)
    #--------------#

    ax.tick_params(axis='both', which='major', labelsize=fontsize)

    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    markers = ['o', 's', 'D', 'v', '^', 'P', 'X', 'd', 'H', 'p']
    linestyles = ['-', '--', '-.', ':']

    linewidth = 2.5
    markersize = 10

    for i, num_latents in enumerate(num_latents_list):
        color = colors[i]
        label = r'M=%s' % num_latents

        df_ = df[df['num_latents'] == num_latents]
        
        kwargs = {
            'marker': markers[i], 'linestyle': linestyles[i],
            'color': color, 'linewidth': linewidth, 'markersize': markersize
        }

        # ax.plot(df_['num_heads'], df_['test_rel_error'], label=label, **kwargs)
        ax.plot(df_['head_dim'], df_['test_rel_error'], label=label, **kwargs)

    #---------------------------------------------------------#
    ax.legend(loc='upper left', ncol=2, fontsize=fontsize, handlelength=2.0)

    plt.tight_layout()

    # Save the figure with both plots
    out_path = os.path.join(PROJDIR, 'figs', f'abl_num_heads_{dataset}.pdf')
    fig.savefig(out_path)
    plt.close()

    #---------------------------------------------------------#
    return

#======================================================================#
def do_training(dataset: str, gpu_count: int = None, max_jobs_per_gpu: int = 2, reverse_queue: bool = False):

    if gpu_count is None:
        import torch
        gpu_count = torch.cuda.device_count()

    if dataset == 'elasticity': # E=500, BS=2, WD=1e-5
        epochs = 500
        batch_size = 2
        weight_decay = 1e-5
    # elif dataset == 'shapenet_car':
    #     epochs = 250
    #     batch_size = 1
    #     weight_decay = 5e-2
    else:
        raise ValueError(f"Dataset {dataset} not supported")
    
    print(f"Using {gpu_count} GPUs to run ablation study on {dataset} dataset.")

    # Create a queue of all jobs
    job_queue = []
    for seed in range(1):
        for num_latents in [16, 32, 64, 128]:
            for num_heads in [1, 2, 4, 8, 16, 32]:
                add_job_to_queue(job_queue, dataset, num_latents=num_latents, num_heads=num_heads, seed=seed, epochs=epochs, batch_size=batch_size, weight_decay=weight_decay)

    utils.run_jobs(job_queue, gpu_count, max_jobs_per_gpu, reverse_queue,
                   dataset=dataset, epochs=epochs, batch_size=batch_size, weight_decay=weight_decay)

    return

#======================================================================#
def add_job_to_queue(
    job_queue: list, dataset: str, num_latents: int, num_heads: int, seed: int,
    epochs: int = 500, batch_size: int = 2, weight_decay: float = 1e-5):

    exp_name = f'abl_num_heads_{dataset}_M_{str(num_latents)}_H_{str(num_heads)}_seed_{str(seed)}'
    exp_name = os.path.join(f'abl_num_heads_{dataset}', exp_name)

    case_dir = os.path.join(CASEDIR, exp_name)
    if os.path.exists(case_dir):
        if os.path.exists(os.path.join(case_dir, 'ckpt10', 'rel_error.json')):
            print(f"Experiment {exp_name} exists. Skipping.")
            return
        else:
            print(f"Experiment {exp_name} exists but ckpt10/rel_error.json does not exist. Removing and re-running.")
            shutil.rmtree(case_dir)

    job_queue.append({
        #
        'exp_name': exp_name,
        'dataset': dataset,
        'seed': seed,
        #
        'epochs': epochs,
        'batch_size': batch_size,
        'weight_decay': weight_decay,
        'mixed_precision': False,
        #
        'model_type': 2,
        #
        'num_blocks': 8,
        'channel_dim': 64,
        'num_heads': num_heads,
        'num_latents': num_latents,
        'num_layers_kv_proj': 3,
        'num_layers_mlp': 3,
        'num_layers_in_out_proj': 2,
    })

    return

#======================================================================#
def clean_results(dataset: str):
    output_dir = os.path.join(CASEDIR, f'abl_num_heads_{dataset}')
    for case_name in [d for d in os.listdir(output_dir) if os.path.isdir(os.path.join(output_dir, d))]:
        case_dir = os.path.join(output_dir, case_name)
        if os.path.exists(os.path.join(case_dir, 'ckpt10', 'rel_error.json')):
            for ckpt in [f'ckpt{i:02d}' for i in range(10)]:
                if os.path.exists(os.path.join(case_dir, ckpt)):
                    shutil.rmtree(os.path.join(case_dir, ckpt))
            if os.path.exists(os.path.join(case_dir, 'ckpt10', 'model.pt')):
                os.remove(os.path.join(case_dir, 'ckpt10', 'model.pt'))
        else:
            shutil.rmtree(case_dir)
    return

#======================================================================#
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Latent Cross Attention model ablation study')

    parser.add_argument('--eval', type=bool, default=False, help='Evaluate ablation study results')
    parser.add_argument('--train', type=bool, default=False, help='Train ablation study')
    parser.add_argument('--clean', type=bool, default=False, help='Clean ablation study results')

    parser.add_argument('--dataset', type=str, default='elasticity', help='Dataset to use')
    parser.add_argument('--gpu-count', type=int, default=None, help='Number of GPUs to use')
    parser.add_argument('--max-jobs-per-gpu', type=int, default=2, help='Maximum number of jobs per GPU')
    parser.add_argument('--reverse-queue', type=bool, default=False, help='Reverse queue')

    args = parser.parse_args()

    if args.clean:
        clean_results(args.dataset)
    if args.train:
        do_training(args.dataset, args.gpu_count, args.max_jobs_per_gpu, args.reverse_queue)
    if args.eval:
        df = collect_data(args.dataset)
        plot_results(args.dataset, df)

    if not args.train and not args.eval and not args.clean:
        print("No action specified. Please specify either --train or --eval or --clean.")

    exit()

#======================================================================#
#