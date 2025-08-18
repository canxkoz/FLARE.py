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
OUT_CSV = os.path.join(PROJDIR, 'out', 'pdebench', 'time_memory_flash.csv')
OUT_PDF = os.path.join(PROJDIR, 'figs', 'time_memory_flash.pdf')

SEQ_LENGTHS = [int(i) for i in np.linspace(1e3, 1e6, 11)]

#======================================================================#
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import flash_attn

#======================================================================#
class TorchMHA(nn.Module):
    def __init__(self, channel_dim: int, num_heads: int = 8):
        super().__init__()

        assert channel_dim % num_heads == 0, f"channel_dim must be divisible by num_heads. Got {channel_dim} and {num_heads}."

        self.channel_dim = channel_dim
        self.num_heads = num_heads
        self.head_dim = channel_dim // num_heads 
        self.scale = self.head_dim ** -0.5

        self.qkv_proj = nn.Linear(channel_dim, 3 * channel_dim, bias=False)
        self.out_proj = nn.Linear(channel_dim, channel_dim)

    def forward(self, x):

        q, k, v = self.qkv_proj(x).chunk(3, dim=-1)
        q, k, v = [rearrange(z, 'b n (h d) -> b h n d', h=self.num_heads) for z in [q, k, v]]

        y = F.scaled_dot_product_attention(q, k, v, scale=self.scale)

        y = rearrange(y, 'b h n d -> b n (h d)')
        y = self.out_proj(y)

        return y

#======================================================================#
class FlashMHA(nn.Module):
    def __init__(self, channel_dim: int, num_heads: int = 8):
        super().__init__()

        assert channel_dim % num_heads == 0, f"channel_dim must be divisible by num_heads. Got {channel_dim} and {num_heads}."

        self.channel_dim = channel_dim
        self.num_heads = num_heads
        self.head_dim = channel_dim // num_heads 
        self.scale = self.head_dim ** -0.5

        self.qkv_proj = nn.Linear(channel_dim, 3 * channel_dim, bias=False)
        self.out_proj = nn.Linear(channel_dim, channel_dim)

    def forward(self, x):

        q, k, v = self.qkv_proj(x).chunk(3, dim=-1)
        q, k, v = [rearrange(z, 'b n (h d) -> b n h d', h=self.num_heads) for z in [q, k, v]]

        y = flash_attn.flash_attn_func(q, k, v, softmax_scale=self.scale)
        
        y = rearrange(y, 'b n h d -> b n (h d)')
        y = self.out_proj(y)

        return y

#======================================================================#
from contextlib import contextmanager
@contextmanager
def cuda_memory_manager(model):
    """Context manager to ensure proper GPU memory cleanup"""
    try:
        model.zero_grad()
        torch.cuda.empty_cache()
        yield
    finally:
        model.zero_grad()
        torch.cuda.empty_cache()

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

    num_heads = 8
    channel_dim = 128

    assert channel_dim % num_heads == 0, f"channel_dim must be divisible by num_heads. Got {channel_dim} and {num_heads}."

    # Only use TorchMHA and FlashMHA
    torch_mha = TorchMHA(channel_dim=channel_dim, num_heads=num_heads)
    flash_mha = FlashMHA(channel_dim=channel_dim, num_heads=num_heads)

    models = [torch_mha, flash_mha]
    model_names = ['Torch MHA', 'Flash MHA']

    for model in models:
        model.to(device)
        model.train()  # Set to training mode for backward pass
        
        # ENABLE gradients for all parameters (needed for backward pass)
        for param in model.parameters():
            param.requires_grad_(True)

    # Use specific attention backend for consistency
    backend = torch.nn.attention.SDPBackend.FLASH_ATTENTION

    with torch.nn.attention.sdpa_kernel(backend):
        # Set deterministic behavior for reproducibility
        torch.manual_seed(42)
        torch.cuda.manual_seed(42)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        data = run_loop(models, model_names, device, channel_dim)

    df = pd.DataFrame(data)
    df.to_csv(OUT_CSV, index=False)

    return df

def run_loop(models, model_names, device, channel_dim):
    for model in models:
        model.train()
    
    data = []
    
    # Warmup run to initialize CUDA kernels
    print("Performing warmup runs...")
    warmup_x = torch.randn(1, 1024, channel_dim, device=device, requires_grad=True)
    warmup_target = torch.randn(1, 1024, channel_dim, device=device)
    
    for model in models:
        try:
            for _ in range(3):  # Multiple warmup runs
                model.zero_grad()
                # Forward pass with autocast for efficiency
                with torch.autocast(device_type=device.type):
                    output = model(warmup_x)
                    loss = F.mse_loss(output, warmup_target)
                
                # Backward pass
                loss.backward()
                torch.cuda.synchronize()
                
                # Clean up after each warmup run
                model.zero_grad()
                del output, loss
                torch.cuda.empty_cache()
        except Exception as e:
            print(f"Warmup failed for model: {e}")
            # Clean up on failure
            model.zero_grad()
            torch.cuda.empty_cache()

    # Clear warmup memory
    del warmup_x, warmup_target
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    print("Starting timing measurements...")
    
    for N in SEQ_LENGTHS:
        print(f"Testing sequence length N={N}")

        for model, model_name in zip(models, model_names):
            print(f"  Testing {model_name}")

            times = []
            memories = []
            num_runs = 5  # Multiple runs for statistical stability
            
            for run in range(num_runs + 2):  # Extra runs to discard first two

                # Create fresh input and target for each run
                torch.cuda.empty_cache()
                x = torch.randn(1, N, channel_dim, device=device, requires_grad=True)
                target = torch.randn(1, N, channel_dim, device=device)
                output = None
                loss = None

                try:
                    # CRITICAL: Clear all gradients and cache before each run
                    model.zero_grad()
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats(device)
                    torch.cuda.synchronize()
                    
                    # Create timing events
                    start_event = torch.cuda.Event(enable_timing=True)
                    end_event = torch.cuda.Event(enable_timing=True)

                    # Time forward + backward pass
                    start_event.record()
                    
                    # Forward pass with autocast for efficiency
                    with torch.autocast(device_type=device.type):
                        output = model(x)
                        loss = F.mse_loss(output, target)
                    
                    # Backward pass
                    loss.backward()
                    
                    end_event.record()
                    torch.cuda.synchronize()

                    # Get measurements
                    elapsed_time = start_event.elapsed_time(end_event) / 1000.0  # Convert to seconds
                    peak_memory = torch.cuda.max_memory_allocated(device) / (1024 ** 3)  # Convert to GB

                    # Only keep measurements after warmup runs
                    if run >= 2:
                        times.append(elapsed_time)
                        memories.append(peak_memory)

                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        print(f"    OOM error: {e}")
                        times = [np.nan] * num_runs
                        memories = [np.nan] * num_runs
                        break
                    else:
                        print(f"    Runtime error: {e}")
                        if run >= 2:
                            times.append(np.nan)
                            memories.append(np.nan)
                except Exception as e:
                    print(f"    Unexpected error: {e}")
                    if run >= 2:
                        times.append(np.nan)
                        memories.append(np.nan)
                
                finally:
                    # CRITICAL: Always clean up, even if exception occurred
                    model.zero_grad()
                    if x is not None and x.grad is not None:
                        x.grad.zero_()
                    
                    # Clean up variables
                    del x, target, output, loss
                    
                    # Force immediate cleanup
                    torch.cuda.empty_cache()

            # Compute statistics (median is more robust than mean for timing)
            if len(times) > 0 and not all(np.isnan(times)):
                time_median = np.nanmedian(times)
                time_std = np.nanstd(times)
                memory_median = np.nanmedian(memories)
                memory_std = np.nanstd(memories)

                print(f"    Time: {time_median:.4f}±{time_std:.4f}s, Memory: {memory_median:.3f}±{memory_std:.3f}GB")
            else:
                time_median = time_std = memory_median = memory_std = np.nan
                print(f"    Failed - insufficient valid measurements")

            data.append({
                'model_name': model_name,
                'N': N,
                'time': time_median,
                'time_std': time_std,
                'memory': memory_median,
                'memory_std': memory_std,
                'num_valid_runs': len([t for t in times if not np.isnan(t)])
            })
            
            # Force memory cleanup between models
            torch.cuda.empty_cache()

    return data

#======================================================================#
def plot_analysis():
    df = pd.read_csv(OUT_CSV)
    
    # # Set matplotlib to use LaTeX fonts
    # plt.rcParams.update({
    #     "text.usetex": True,
    #     "font.family": "serif",
    #     "font.serif": ["Computer Modern Roman"],
    #     "text.latex.preamble": r"\usepackage{amsmath}"
    # })

    fontsize = 20
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    for ax in [ax1, ax2]:
        ax.set_xscale('linear')
        ax.grid(True, which="both", ls="-", alpha=0.5)
        ax.set_xlabel(f'Sequence Length', fontsize=fontsize)
        
    ax1.set_yscale('log', base=10)
    ax2.set_yscale('linear')
    ax2.set_ylim(0, 85)

    ax1.set_ylabel(f'Time (s)', fontsize=fontsize)
    ax2.set_ylabel(f'Memory (GB)', fontsize=fontsize)
    
    # Increase tick label size
    for ax in [ax1, ax2]:
        ax.tick_params(axis='both', which='major', labelsize=fontsize)

    # Set custom x-ticks for sequence lengths
    x_ticks = [1000, 100000, 200000, 300000, 400000, 500000, 600000, 700000, 800000, 900000, 1000000]
    x_tick_labels = ['1k', '', '200k', '', '400k', '', '600k', '', '800k', '', '1m']
    
    for ax in [ax1, ax2]:
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_tick_labels)

    # Add horizontal dashed line at Memory = 80 GB
    ax2.axhline(y=80, color='black', linestyle='--', linewidth=3.0)

    for model_name in df['model_name'].unique():
        model_data = df[df['model_name'] == model_name]
        model_data = model_data.sort_values(by='N')

        if 'Torch' in model_name:
            marker = 'o'
            color = 'black'
            linestyle = '-'
            label = f'Torch MHA'
        elif 'Flash' in model_name:
            marker = 's'
            color = 'blue'
            linestyle = '--'
            label = f'Flash MHA'

        marker_size = 7  # Increased marker size

        ax1.plot(model_data['N'], model_data['time'], label=label, marker=marker, 
            linestyle=linestyle, linewidth=2.5, color=color, markersize=marker_size)
        ax2.plot(model_data['N'], model_data['memory'], label=label, marker=marker, 
            linestyle=linestyle, linewidth=2.5, color=color, markersize=marker_size)
    
    # Simple legend since we only have 2 models
    handles, labels = ax1.get_legend_handles_labels()
    legend = fig.legend(handles, labels, loc='lower center', ncol=2, 
              frameon=True, fancybox=False, shadow=False, fontsize=fontsize, 
              bbox_to_anchor=(0.5, 0.00), columnspacing=2.0, handletextpad=0.8,
              bbox_transform=fig.transFigure, handlelength=2.0, markerscale=1.5)

    # Add title with larger font
    ax1.set_title(f'Execution Time - Forward + Backward', fontsize=fontsize)
    ax2.set_title(f'Memory Usage - Forward + Backward', fontsize=fontsize)

    # Adjust layout with extra space at bottom for legend
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)  # Reduced bottom margin since we only have 2 models

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