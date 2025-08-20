import torch
import torch_geometric as pyg
import torch.multiprocessing as mp

import os
import numpy as np

from .utils import makegraph

__all__ = [
    'DatasetTransform',
]

#======================================================================#
# TRANSFORM
#======================================================================#
class DatasetTransform:
    def __init__(
        self,
        disp=True, vmstr=False, temp=False,
        sdf=False, mesh=True, elems=False, orig=False, metadata=False,
    ):
        # fields
        self.disp  = disp
        self.vmstr = vmstr
        self.temp  = temp
        self.sdf   = sdf

        self.mesh  = mesh
        self.elems = elems
        self.orig  = orig

        self.metadata = metadata

        # Normalization
        # pos  : x, y [-30, 30] mm, z [-25, 60] mm ([-25, 0] build plate, [0, 60] part)
        # disp : x [-0.5, 0.5] mm, y [-0.05, 0.05] mm, z [-0.1, -1] mm
        # vmstr: [0, ~5e3] Pascal (?)
        # temp : Celcius [25, ~300]
        #
        # layer thickness is 2.5 mm. Total number of layer increments is 60 / 2.5 = 24
        # layer 0 is [0, 2.5] mm
        # layer 1 is [2.5, 5] mm
        # layer 24 is [57.5, 60] mm

        self.pos_scale = torch.tensor([30., 30., 60.])
        self.disp_scale  = 1.
        self.vmstr_scale = 1000.
        self.temp_scale  = 500.

        # makefields
        self.nfields = disp + vmstr + temp 

        scale = []
        scale = [*scale, self.disp_scale ] if self.disp  else scale
        scale = [*scale, self.vmstr_scale] if self.vmstr else scale
        scale = [*scale, self.temp_scale ] if self.temp  else scale
        self.scale = torch.tensor(scale)

        assert self.nfields == len(scale)

        return

    def normalize_fields(self, graph):
        pos   = graph.pos   / self.pos_scale
        disp  = graph.disp  / self.disp_scale
        vmstr = graph.vmstr / self.vmstr_scale
        temp  = graph.temp  / self.temp_scale
        if hasattr(graph, 'edge_dxyz'):
            edge_dxyz = graph.edge_dxyz / self.pos_scale
        else:
            edge_dxyz = None
        return pos, disp, vmstr, temp, edge_dxyz

    def normalize_sdf_x(self, sdf_x):
        # input: surface_mask (1), sdf_direction (3), sdf_magnitude (1), distances (6)
        sdf_x = sdf_x[:, 1:].clone()

        sdf_x[:, :3] /= self.pos_scale
        sdf_x[:, 3] /= self.pos_scale.norm()
        sdf_x[:, 4:7] /= self.pos_scale
        sdf_x[:, 7:10] /= self.pos_scale

        return sdf_x

    def make_pyg_data(self, graph, edge_attr, **kw):
        data = pyg.data.Data(**kw)

        if self.mesh:
            data.edge_attr  = edge_attr
            data.edge_index = graph.edge_index
        if self.elems:
            data.elems = graph.elems
        if self.orig:
            data.pos   = graph.pos
            data.disp  = graph.disp
            data.vmstr = graph.vmstr
            data.temp  = graph.temp
        if self.metadata:
            data.metadata = graph.metadata

        return data

    def __call__(self, graph):
        raise NotImplementedError()

#======================================================================#
#