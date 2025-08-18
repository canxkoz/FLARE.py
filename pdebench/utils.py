#
import torch
import einops
from torch import nn
import torch.nn.functional as F

__all__ = [
    'make_optimizer_adamw',
    'make_optimizer_lion',
    #
    'darcy_deriv_loss',
    #
    'RelL1Loss',
    'RelL2Loss',
    #
    'IdentityNormalizer',
    'UnitCubeNormalizer',
    'UnitGaussianNormalizer',
]

#======================================================================#
def split_params(model):
    decay = []
    no_decay = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # skip frozen weights
        if name.endswith(".bias") or "LayerNorm" in name or "layernorm" in name or "embedding" in name.lower():
            no_decay.append(param)
        elif 'latent' in name:
            no_decay.append(param)
        elif 'cls_token' in name:
            no_decay.append(param)
        elif 'pos_embed' in name:
            no_decay.append(param)
        else:
            decay.append(param)

    return decay, no_decay

#======================================================================#
def make_optimizer_adamw(model, lr, weight_decay=0.0, betas=None, eps=None, **kwargs):
    betas = betas if betas is not None else (0.9, 0.999)
    eps = eps if eps is not None else 1e-8

    decay, no_decay = split_params(model)

    optimizer = torch.optim.AdamW([
        {'params': decay, 'weight_decay': weight_decay},
        {'params': no_decay, 'weight_decay': 0.0}
    ], lr=lr, betas=betas, eps=eps)

    return optimizer

def make_optimizer_lion(model, lr, weight_decay=0.0, betas=None, eps=None, **kwargs):
    betas = betas if betas is not None else (0.9, 0.999)
    eps = eps if eps is not None else 1e-8

    decay, no_decay = split_params(model)
    
    optimizer = Lion([
        {'params': decay, 'weight_decay': weight_decay},
        {'params': no_decay, 'weight_decay': 0.0}
    ], lr=lr, betas=betas, eps=eps)

    return optimizer

#======================================================================#
from torch.optim.optimizer import Optimizer

class Lion(Optimizer):
    r"""
    Lion Optimizer (Chen et al., 2023):
    https://arxiv.org/abs/2302.06675

    Update rule:
        m_t = beta1 * m_{t-1} + (1 - beta1) * grad
        w_{t+1} = w_t - lr * sign(m_t)

    Args:
        params (iterable): iterable of parameters to optimize
        lr (float): learning rate
        betas (Tuple[float, float]): momentum coefficients (beta1, beta2). Note that beta2 is not used.
        weight_decay (float): optional weight decay (L2 penalty)
        eps (float): optional epsilon. Note that eps is not used.
    """

    def __init__(self, params, lr=1e-4, betas=(0.9, 0.99), weight_decay=0.0, eps=1e-8):
        if lr <= 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta1: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta2: {betas[1]}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")
        if eps <= 0.0:
            raise ValueError(f"Invalid eps: {eps}")

        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay, eps=eps)
        super(Lion, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            wd = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad

                # Apply weight decay directly to weights
                if wd != 0:
                    grad = grad.add(p, alpha=wd)

                # State (momentum) initialization
                state = self.state[p]
                if "exp_avg" not in state:
                    state["exp_avg"] = torch.zeros_like(p)

                exp_avg = state["exp_avg"]

                # Momentum update
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)

                # Parameter update (sign of momentum)
                p.add_(exp_avg.sign(), alpha=-lr)

        return loss

#======================================================================#
def central_diff(x: torch.Tensor, h: float, resolution: int):
    # assuming PBC
    # x: (batch, n, feats), h is the step size, assuming n = h*w
    x = einops.rearrange(x, 'b (h w) c -> b h w c', h=resolution, w=resolution)
    x = F.pad(x, (0, 0, 1, 1, 1, 1), mode='constant', value=0.)  # [b c t h+2 w+2]
    grad_x = (x[:, 1:-1, 2:, :] - x[:, 1:-1, :-2, :]) / (2 * h)  # f(x+h) - f(x-h) / 2h
    grad_y = (x[:, 2:, 1:-1, :] - x[:, :-2, 1:-1, :]) / (2 * h)  # f(x+h) - f(x-h) / 2h

    return grad_x, grad_y

def darcy_deriv_loss(yh, y, s, dx):
    yh = einops.rearrange(yh, 'b (h w) c -> b c h w', h=s)
    yh = yh[..., 1:-1, 1:-1].contiguous()
    yh = F.pad(yh, (1, 1, 1, 1), "constant", 0)
    yh = einops.rearrange(yh, 'b c h w -> b (h w) c')

    gt_grad_x, gt_grad_y = central_diff(y, dx, s)
    pred_grad_x, pred_grad_y = central_diff(yh, dx, s)

    return (gt_grad_x, gt_grad_y), (pred_grad_x, pred_grad_y)

#======================================================================#
class IdentityNormalizer():
    def __init__(self):
        pass
    
    def to(self, device):
        return self

    def encode(self, x):
        return x

    def decode(self, x):
        return x

#======================================================================#
class UnitCubeNormalizer():
    def __init__(self, X):
        xmin = X[:,:,0].min().item()
        ymin = X[:,:,1].min().item()

        xmax = X[:,:,0].max().item()
        ymax = X[:,:,1].max().item()

        self.min = torch.tensor([xmin, ymin])
        self.max = torch.tensor([xmax, ymax])

    def to(self, device):
        self.min = self.min.to(device)
        self.max = self.max.to(device)

        return self

    def encode(self, x):
        x = (x - self.min) / (self.max - self.min)
        return x

    def decode(self, x):
        return x * (self.max - self.min) + self.min

#======================================================================#
class UnitGaussianNormalizer():
    def __init__(self, X):
        self.mean = X.mean(dim=(0, 1), keepdim=True)
        self.std = X.std(dim=(0, 1), keepdim=True) + 1e-8

    def to(self, device):
        self.mean = self.mean.to(device)
        self.std = self.std.to(device)
        return self

    def encode(self, x):
        x = (x - self.mean) / (self.std)
        return x

    def decode(self, x):
        return x * self.std + self.mean

#======================================================================#
class RelL2Loss(nn.Module):
    def forward(self, pred, target):
        assert pred.shape == target.shape
        dim = tuple(range(1, pred.ndim))

        error = torch.sum((pred - target) ** 2, dim=dim).sqrt()
        target = torch.sum(target ** 2, dim=dim).sqrt()

        loss = torch.mean(error / target)
        return loss

class RelL1Loss(nn.Module):
    def forward(self, pred, target):
        assert pred.shape == target.shape
        dim = tuple(range(1, pred.ndim))

        error = torch.sum(torch.abs(pred - target), dim=dim)
        target = torch.sum(torch.abs(target), dim=dim)

        loss = torch.mean(error / target)
        return loss

#======================================================================#
#