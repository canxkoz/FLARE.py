#
import torch
from torch import nn
import torch.nn.functional as F
import torch_geometric as pyg
from mlutils.utils import check_package_version_lteq

__all__ = [
    'MaskedLoss',
    'MaskedModel',
]

#======================================================================#
class MaskedLoss(torch.nn.Module):
    """
    Loss function that:
    1.  masks out predictions made at the final step because
        next step prediction doesn't make any sense there.
    2.  Computes batch loss over the active elements of the graph,
        i.e., regions where batch.mask == True.
    """
    def __init__(self, mask: bool, tol: float = 1e-5):
        super().__init__()

        self.tol = tol
        self.mask = mask

    def forward(self, trainer, model, batch):
        losses = []
        for graph in batch.to_data_list():
            losses.append(self.forward_single(trainer, model, graph))
            
        return sum(losses) / len(losses)

    def forward_single(self, trainer, model, batch):
        y  = batch.y
        yh = model(batch)

        # (1) last step mask
        last_step_mask = (batch.t <= batch.T - self.tol).view(-1, 1)
        yh = yh * last_step_mask
        y  = y  * last_step_mask
        
        # (2) remove inactive elements
        if self.mask:
            mask = batch.mask.view(-1,1)
            yh = yh * mask
            y  = y  * mask

        loss = F.mse_loss(yh, y)

        # if loss.item() > 1e1:
        #     print(f'loss: {loss} t: {batch.t[0].item()} T: {batch.T[0].item()} y: {y.norm().item()} yh: {yh.norm().item()}')

        return loss

#======================================================================#
class MaskedModel(nn.Module):
    """
    Reduces the graph to a subgraph of active elements.
    """
    def __init__(self, model, mask=True, reduce_graph=True):
        super().__init__()
        self.mask = mask
        self.reduce_graph = reduce_graph
        self.model = model

    @torch.no_grad()
    def _reduce_graph(self, graph):
        
        if hasattr(graph, 'edge_index'):
            if graph.edge_index is None:
                return graph
        else:
            return graph

        edge_index, edge_attr = pyg.utils.subgraph(
            graph.mask.view(-1), graph.edge_index, graph.edge_attr
        )

        return pyg.data.Data(
            **{
                k: graph[k]
                for k in pyg_get_data_keys(graph)
                if k != 'edge_index' and k != 'edge_attr'
            },
            edge_index=edge_index,
            edge_attr=edge_attr,
        )

    def forward(self, graph):
        if self.mask:
            mask = graph.mask.view(-1, 1)
            graph = self._reduce_graph(graph) if self.reduce_graph else graph
            x = self.model(graph)
            x = x * mask
        else:
            x = self.model(graph)

        return x

#======================================================================#
def pyg_get_data_keys(data: pyg.data.Data):
    if check_package_version_lteq('torch_geometric', '2.4'):
        k = data.keys
    else:
        k = data.keys()

    return k

#======================================================================#
#
