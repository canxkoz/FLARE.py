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
OUT_CSV = os.path.join(PROJDIR, 'out', 'pdebench', 'time_memory_bwd.csv')
OUT_PNG = os.path.join(PROJDIR, 'figs', 'time_memory_bwd.png')
OUT_PDF = os.path.join(PROJDIR, 'figs', 'time_memory_bwd.pdf')

NUM_LATENTS = [128, 512, 2048]
SEQ_LENGTHS = [int(i) for i in np.linspace(1e3, 1e6, 11)]

#======================================================================#
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

#======================================================================#
class MultiHeadedSelfAttention(nn.Module):
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
        
        # score = (q @ k.transpose(-1, -2)) * self.scale
        # attn = F.softmax(score, dim=-1)
        # y = attn @ v

        y = rearrange(y, 'b h n d -> b n (h d)')
        y = self.out_proj(y)

        return y

#======================================================================#
class PhysicsAttention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0., slice_num=64):
        super().__init__()
        inner_dim = dim_head * heads
        self.dim_head = dim_head
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.temperature = nn.Parameter(torch.ones([1, heads, 1, 1]) * 0.5)

        self.dropout_p = dropout

        self.in_project_x = nn.Linear(dim, inner_dim)
        self.in_project_fx = nn.Linear(dim, inner_dim)
        self.in_project_slice = nn.Linear(dim_head, slice_num)
        for l in [self.in_project_slice]:
            torch.nn.init.orthogonal_(l.weight)  # use a principled initialization
        self.to_q = nn.Linear(dim_head, dim_head, bias=False)
        self.to_k = nn.Linear(dim_head, dim_head, bias=False)
        self.to_v = nn.Linear(dim_head, dim_head, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

        self.channel_dim = dim

    def forward(self, x):
        B, N, C = x.shape

        ### (1) Sliceing (value, key) [B H N C]
        fx_mid = self.in_project_fx(x).reshape(B, N, self.heads, self.dim_head).permute(0, 2, 1, 3).contiguous()
        x_mid = self.in_project_x(x).reshape(B, N, self.heads, self.dim_head).permute(0, 2, 1, 3).contiguous()

        slice_weights = self.softmax(self.in_project_slice(x_mid) / self.temperature)  # B H N G
        slice_norm = slice_weights.sum(2)  # B H G
        slice_token = torch.einsum("bhnc,bhng->bhgc", fx_mid, slice_weights)
        slice_token = slice_token / ((slice_norm + 1e-5)[:, :, :, None].repeat(1, 1, 1, self.dim_head))

        ### (2) Attention among slice tokens
        q_slice_token = self.to_q(slice_token)
        k_slice_token = self.to_k(slice_token)
        v_slice_token = self.to_v(slice_token)
        # dots = torch.matmul(q_slice_token, k_slice_token.transpose(-1, -2)) * self.scale
        # attn = self.softmax(dots)
        # attn = self.dropout(attn)
        out_slice_token = F.scaled_dot_product_attention(q_slice_token, k_slice_token, v_slice_token, scale=self.scale, dropout_p=self.dropout_p)
        # out_slice_token = torch.matmul(attn, v_slice_token)  # B H G D

        ### (3) Deslice
        out_x = torch.einsum("bhgc,bhng->bhnc", out_slice_token, slice_weights)
        out_x = rearrange(out_x, 'b h n d -> b n (h d)')

        return self.to_out(out_x)

#======================================================================#
ACTIVATIONS = {
    'gelu': nn.GELU(approximate='tanh'),
    'silu': nn.SiLU(),
}

class ResidualMLP(nn.Module):
    def __init__(
            self, in_dim: int, hidden_dim: int, out_dim: int, num_layers: int = 2,
            act: str = None, input_residual: bool = False, output_residual: bool = False,
        ):
        super().__init__()
        self.act = ACTIVATIONS[act] if act else ACTIVATIONS['gelu']
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fcs = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)])
        self.fc2 = nn.Linear(hidden_dim, out_dim)

        self.input_residual  = input_residual  and (in_dim  == hidden_dim)
        self.output_residual = output_residual and (hidden_dim == out_dim)

    def forward(self, x):

        x = x + self.act(self.fc1(x)) if self.input_residual else self.act(self.fc1(x))
        for fc in self.fcs:
            x = x + self.act(fc(x))
        x = x + self.fc2(x) if self.output_residual else self.fc2(x)

        return x

class FLARE(nn.Module):
    def __init__(
        self,
        channel_dim: int,
        num_heads: int = 8,
        num_latents: int = 32,
        act: str = None,
        num_layers_kv_proj: int = 3,
    ):
        super().__init__()

        self.channel_dim = channel_dim
        self.num_latents = num_latents
        self.num_heads = num_heads
        self.head_dim = self.channel_dim // self.num_heads

        assert self.channel_dim % self.num_heads == 0, f"channel_dim must be divisible by num_heads. Got {self.channel_dim} and {self.num_heads}."

        self.latent_q = nn.Parameter(torch.empty(self.channel_dim, self.num_latents))
        nn.init.normal_(self.latent_q, mean=0.0, std=0.1)

        self.k_proj, self.v_proj = [ResidualMLP(
            in_dim=self.channel_dim, hidden_dim=self.channel_dim, out_dim=self.channel_dim,
            num_layers=num_layers_kv_proj, act=act, input_residual=True, output_residual=True,
        ) for _ in range(2)]

        self.out_proj = nn.Linear(self.channel_dim, self.channel_dim)

    def forward(self, x):

        q = self.latent_q.view(self.num_heads, self.num_latents, self.head_dim) # [H M D]
        k = rearrange(self.k_proj(x), 'b n (h d) -> b h n d', h=self.num_heads) # [B H N D]
        v = rearrange(self.v_proj(x), 'b n (h d) -> b h n d', h=self.num_heads)

        #--------------------------------------------#
        q = q.unsqueeze(0).expand(x.size(0), -1, -1, -1) # required for fused attention
        z = F.scaled_dot_product_attention(q, k, v, scale=1.0)
        y = F.scaled_dot_product_attention(k, q, z, scale=1.0)
        #--------------------------------------------#

        y = rearrange(y, 'b h n d -> b n (h d)')
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

    MSHA = MultiHeadedSelfAttention(channel_dim=channel_dim, num_heads=num_heads)
    FLAREs = [FLARE(channel_dim=channel_dim, num_heads=num_heads, num_latents=NUM_LATENTS[i]) for i in range(len(NUM_LATENTS))]
    PhAs = [PhysicsAttention(dim=channel_dim, heads=num_heads, dim_head=channel_dim // num_heads, slice_num=NUM_LATENTS[i]) for i in range(len(NUM_LATENTS))]

    FLARE_names = [f'FLARE (M={NUM_LATENTS[i]})' for i in range(len(NUM_LATENTS))]
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

    # Use specific attention backend for consistency
    backend = torch.nn.attention.SDPBackend.FLASH_ATTENTION

    with torch.nn.attention.sdpa_kernel(backend):
        # Set deterministic behavior for reproducibility
        torch.manual_seed(42)
        torch.cuda.manual_seed(42)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        data = run_loop(models, model_names, device, SEQ_LENGTHS, autocast_enabled=True)

    df = pd.DataFrame(data)
    df.to_csv(OUT_CSV, index=False)

    return df

def run_loop(models, model_names, device, seq_lengths, autocast_enabled=True):
    for model in models:
        model.train()
    
    data = []
    
    # Warmup run to initialize CUDA kernels
    print("Performing warmup runs...")
    
    for model in models:
        channel_dim = model.channel_dim

        warmup_x = torch.randn(1, 1024, channel_dim, device=device, requires_grad=True)
        warmup_target = torch.randn(1, 1024, channel_dim, device=device)

        try:
            for _ in range(3):  # Multiple warmup runs
                model.zero_grad()
                # Forward pass with autocast for efficiency
                with torch.autocast(device_type=device.type, enabled=autocast_enabled):
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
    
    for N in seq_lengths:
        print(f"Testing sequence length N={N}")

        for model, model_name in zip(models, model_names):
            print(f"  Testing {model_name}")

            times = []
            memories = []
            num_runs = 5  # Multiple runs for statistical stability

            channel_dim = model.channel_dim
            
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
                    with torch.autocast(device_type=device.type, enabled=autocast_enabled):
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
    
    # Set matplotlib to use LaTeX fonts
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
        "text.latex.preamble": r"\usepackage{amsmath}"
    })

    fontsize = 26
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    for ax in [ax1, ax2]:
        ax.set_xscale('linear')
        ax.grid(True, which="both", ls="-", alpha=0.5)
        ax.set_xlabel(r'Sequence Length', fontsize=fontsize)
        
    ax1.set_yscale('log', base=10)
    ax2.set_yscale('linear')
    ax2.set_ylim(0, 85)

    ax1.set_ylabel(r'Time (s)', fontsize=fontsize)
    ax2.set_ylabel(r'Memory (GB)', fontsize=fontsize)
    
    # Increase tick label size
    for ax in [ax1, ax2]:
        ax.tick_params(axis='both', which='major', labelsize=fontsize)

    # Define markers for different num_latents
    latent_map = {
        f"{NUM_LATENTS[0]}": ['s', '-' ],
        f"{NUM_LATENTS[1]}": ['v', '--'],
        f"{NUM_LATENTS[2]}": ['D', ':' ]
    }

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

        ax1.plot(model_data['N'], model_data['time'], label=label, marker=marker, 
            linestyle=linestyle, linewidth=2.5, color=color, markersize=marker_size)
        ax2.plot(model_data['N'], model_data['memory'], label=label, marker=marker, 
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
    ax1.set_title(r'Execution Time (Forward + Backward)', fontsize=fontsize)
    ax2.set_title(r'Peak Memory Usage (Forward + Backward)', fontsize=fontsize)

    # Adjust layout with extra space at bottom for legend
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.34)  # Increased bottom margin for legend

    # Save the figure with both plots
    fig.savefig(OUT_PDF, dpi=300, bbox_inches='tight')
    fig.savefig(OUT_PNG, dpi=300, bbox_inches='tight')
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
        if os.path.exists(OUT_PNG):
            os.remove(OUT_PNG)

    if not args.run and not args.plot and not args.clean:
        print("No action specified. Please specify either --run or --plot or --clean.")
#======================================================================#
#