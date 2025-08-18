#
import torch
from torch import nn

import mlutils

__all__ = [
    'march_case',
]

#======================================================================#
@torch.no_grad()
def march_case(
    model, case_data, transform,
    autoreg=True, K=1, verbose=False, device=None, tol=1e-4,
):

    if device is None:
        device = mlutils.select_device(device)
    model.to(device)

    if not autoreg:
        K = 1

    nf = transform.nfields

    eval_data = []
    l2s = []
    r2s = []

    for (istep, data) in enumerate(case_data):
        data = data.clone()
        data.y = transform.makefields(data, istep, scale=True)
        data.e = torch.zeros_like(data.y)
        eval_data.append(data)
        l2s.append(0.)
        r2s.append(1.)

    for k in range(K, len(eval_data)):
        _data = eval_data[k-1].to(device) # given (k-1)-th step
        data  = eval_data[k  ].to(device) # predict k-th step
        
        if autoreg:
            _data = _data.clone()
            _data.x[:, -nf:] = _data.y[:, -nf:]

        target = transform.makefields(data, k, scale=True)

        if transform.interpolate:
            _data.x[:, -nf:] = transform.interpolate_layer(
                _data.x[:, -nf:], _data, k-1, tol=tol
            )

        data.y = model(_data) + _data.x[:, -nf:]
        data.e = data.y - target

        l2s[k] = nn.MSELoss()(data.e, 0 * data.e).item()
        r2s[k] = mlutils.r2(data.y, target)

        if verbose:
            print(f'Step {k}: {l2s[k], r2s[k]}')

    return eval_data, l2s, r2s

#======================================================================#
