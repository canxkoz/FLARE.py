"""
Forward + Backward Pass Timing Analysis

This script measures the time and memory usage for both forward and backward passes
of different attention models. It's designed for training scenarios where gradients
are computed and propagated.

Key features:
- Times complete forward + backward pass (including loss computation)
- Uses realistic MSE loss for gradient computation
- Models are in training mode with gradients enabled
- Multiple runs with statistical analysis (median ± std)
- Proper GPU warmup and memory management
- Handles OOM errors gracefully
"""

import os
import sys
import argparse

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#======================================================================#
PROJDIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
OUT_CSV = os.path.join(PROJDIR, 'out', 'pdebench', 'time_memory_bwd_fp16_fp32_comp.csv')
OUT_PDF = os.path.join(PROJDIR, 'figs', 'time_memory_bwd_fp16_fp32_comp.pdf')

NUM_LATENTS = [64, 128, 256]
SEQ_LENGTHS = [int(i) for i in np.linspace(1e3, 1e5, 21)]

#======================================================================#
from time_memory_bwd import (
    MultiHeadedSelfAttention,
    PhysicsAttention,
    FLARE,
    run_loop
)

#======================================================================#
def run_analysis():

    #---------------------------------------#
    device = torch.device(0)

    # Check for thermal throttling
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(device)}")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"Initial GPU memory: {torch.cuda.get_device_properties(device).total_memory / (1024**3):.1f} GB")
        
        # Suggest memory optimization
        print("Tip: Set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to reduce fragmentation")

        # Wait for GPU to cool down if it was recently used
        import time
        print("Waiting for GPU to stabilize...")
        time.sleep(5)
        
        # Check initial memory usage
        initial_memory = torch.cuda.memory_allocated(device) / (1024**3)
        if initial_memory > 10:  # More than 10GB already used
            print(f"Warning: GPU already using {initial_memory:.1f}GB - consider restarting process")
            print("Clearing cache and continuing...")
            torch.cuda.empty_cache()
    #---------------------------------------#

    MSHA = MultiHeadedSelfAttention(channel_dim=96, num_heads=6)
    FLAREs = [FLARE(channel_dim=64, num_heads=8, num_latents=NUM_LATENTS[i]) for i in range(len(NUM_LATENTS))]
    PhAs = [PhysicsAttention(dim=128, heads=8, dim_head=8, slice_num=NUM_LATENTS[i]) for i in range(len(NUM_LATENTS))]

    FLARE_names = [f'FLARE Layer (M={NUM_LATENTS[i]})' for i in range(len(NUM_LATENTS))]
    PhA_names = [f'Phys Attn (M={NUM_LATENTS[i]})' for i in range(len(NUM_LATENTS))]
    MSHA_name = 'Vanilla Attn'

    models = [MSHA, *FLAREs, *PhAs]
    model_names  = [MSHA_name, *FLARE_names, *PhA_names]

    for model in models:
        model.to(device)
        model.train()  # Set to training mode for backward pass
        
        # ENABLE gradients for all parameters (needed for backward pass)
        for param in model.parameters():
            param.requires_grad_(True)

    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    data_fp16 = run_loop(models, model_names, device, SEQ_LENGTHS, autocast_enabled=True)
    data_fp32 = run_loop(models, model_names, device, SEQ_LENGTHS, autocast_enabled=False)
    
    for data in data_fp16:
        data['dtype'] = 'fp16'
    for data in data_fp32:
        data['dtype'] = 'fp32'

    df = pd.DataFrame(data_fp16 + data_fp32)
    df.to_csv(OUT_CSV, index=False)

    return df

#======================================================================#
def plot_analysis():
    df = pd.read_csv(OUT_CSV)
    
    # Set matplotlib to use LaTeX fonts
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
        "text.latex.preamble": r"\usepackage{amsmath}"
    })

    fontsize = 26

    # fig, ax1 = plt.subplots(1, 1, figsize=(16, 8))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    for ax in [ax1, ax2]:
        ax.set_xscale('linear')
        ax.grid(True, which="both", ls="-", alpha=0.5)
        ax.set_xlabel(r'Sequence Length', fontsize=fontsize)
        ax.set_yscale('log', base=10)
        ax.set_ylim(8e-4, 2.0)

    ax1.set_ylabel(r'Time (s)', fontsize=fontsize)
    
    # Increase tick label size
    for ax in [ax1, ax2]:
        ax.tick_params(axis='both', which='major', labelsize=fontsize)

    # Define markers for different num_latents
    latent_map = {
        f"{NUM_LATENTS[0]}": ['s', '-' ],
        f"{NUM_LATENTS[1]}": ['v', '--'],
        f"{NUM_LATENTS[2]}": ['D', ':' ]
    }

    # # Set custom x-ticks for sequence lengths
    # x_ticks = [1e3, 1e4, 2e4, 3e4, 4e4, 5e4, 6e4, 7e4, 8e4, 9e4, 1e5]
    # x_tick_labels = ['1k', '10k', '20k', '30k', '40k', '50k', '60k', '70k', '80k', '90k', '100k']
    
    x_ticks = [1e3, 1e4, 2e4, 3e4, 4e4, 5e4, 6e4, 7e4, 8e4, 9e4, 1e5]
    x_tick_labels = ['1k', '', '20k', '', '40k', '', '60k', '', '80k', '', '100k']

    for ax in [ax1, ax2]:
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_tick_labels)

    for model_name in df['model_name'].unique():
        model_data = df[df['model_name'] == model_name]
        model_data = model_data.sort_values(by='N')

        if 'Vanilla' in model_name:
            marker = 'o'
            color = 'black'
            linestyle = '-'
            label = r'Vanilla Attention'
        elif 'Phys' in model_name:
            latent_size = model_name.split('=')[1].strip(')')
            marker, linestyle = latent_map[latent_size]
            color = 'blue'
            label = r'Physics Attention ($%s$ slices)' % latent_size
        elif 'FLARE' in model_name:
            latent_size = model_name.split('=')[1].strip(')')
            marker, linestyle = latent_map[latent_size]
            color = 'red'
            label = r'FLARE ($%s$ latents) (ours)' % latent_size

        marker_size = 7  # Increased marker size

        fp16_data = model_data[model_data['dtype'] == 'fp16']
        fp32_data = model_data[model_data['dtype'] == 'fp32']

        ax1.plot(fp16_data['N'], fp16_data['time'], label=label, marker=marker, 
            linestyle=linestyle, linewidth=2.5, color=color, markersize=marker_size)
        ax2.plot(fp32_data['N'], fp32_data['time'], label=label, marker=marker, 
            linestyle=linestyle, linewidth=2.5, color=color, markersize=marker_size)
    # Add legend to bottom of the figure with 3 columns
    handles, labels = ax1.get_legend_handles_labels()
    
    # Organize legend into 3 columns: Vanilla, Phys Attn, FLARE
    vanilla_items = [(h, l) for h, l in zip(handles, labels) if 'Vanilla' in l]
    phys_items = [(h, l) for h, l in zip(handles, labels) if 'Phys' in l]
    flare_items = [(h, l) for h, l in zip(handles, labels) if 'FLARE' in l]

    # Reorder for 3-column layout: pad shorter columns with empty entries
    max_len = max(len(vanilla_items), len(phys_items), len(flare_items))

    # Create ordered lists for 3-column layout
    ordered_handles = []
    ordered_labels = []

    for i in range(max_len):
        # Row 1: FLARE
        if i < len(flare_items):
            ordered_handles.append(flare_items[i][0])
            ordered_labels.append(flare_items[i][1])
        else:
            ordered_handles.append(plt.Line2D([0], [0], color='white', alpha=0))
            ordered_labels.append('')

        # Row 2: Physics Attention  
        if i < len(phys_items):
            ordered_handles.append(phys_items[i][0])
            ordered_labels.append(phys_items[i][1])
        else:
            ordered_handles.append(plt.Line2D([0], [0], color='white', alpha=0))
            ordered_labels.append('')

        # Row 3: Vanilla Attention
        if i < len(vanilla_items):
            ordered_handles.append(vanilla_items[i][0])
            ordered_labels.append(vanilla_items[i][1])
        else:
            ordered_handles.append(plt.Line2D([0], [0], color='white', alpha=0))
            ordered_labels.append('')

    # Place legend below the subplots with wider line representations
    legend = fig.legend(ordered_handles, ordered_labels, loc='lower center', ncol=3, 
              frameon=True, fancybox=False, shadow=False, fontsize=fontsize, 
              bbox_to_anchor=(0.5, 0.00), columnspacing=0.5, handletextpad=0.2,
              bbox_transform=fig.transFigure, handlelength=1.5, markerscale=1.5)

    # Add title with larger font
    ax1.set_title(r'Execution Time - Mixed Precision', fontsize=fontsize)
    ax2.set_title(r'Execution Time - Full Precision', fontsize=fontsize)

    # Adjust layout with extra space at bottom for legend
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.34)  # Increased bottom margin for legend

    # Save the figure with both plots
    fig.savefig(OUT_PDF, dpi=300, bbox_inches='tight')  # Higher DPI for better quality
    plt.close()

    return
#======================================================================#
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Forward + Backward Pass Timing Analysis for Attention Models')

    parser.add_argument('--run', type=bool, default=False, help='Run forward+backward timing analysis')
    parser.add_argument('--plot', type=bool, default=False, help='Plot forward+backward timing results')
    parser.add_argument('--clean', type=bool, default=False, help='Clean forward+backward timing results')

    args = parser.parse_args()

    # Set memory optimization environment variable
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

    if args.run:
        run_analysis()
    if args.plot:
        plot_analysis()
    if args.clean:
        if os.path.exists(OUT_CSV):
            os.remove(OUT_CSV)
        if os.path.exists(OUT_PDF):
            os.remove(OUT_PDF)

    if not args.run and not args.plot and not args.clean:
        print("No action specified. Please specify either --run or --plot or --clean.")
#======================================================================#
#