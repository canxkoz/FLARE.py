import torch
import torch_geometric as pyg
import torch.multiprocessing as mp

import numpy as np
from tqdm import tqdm

import os

from mlutils.utils import (to_numpy, check_package_version_lteq)
from .utils import makegraph
from .transform import DatasetTransform

__all__ = [
    'FinaltimeDatasetTransform',
    'FinaltimeDataset',
]

#======================================================================#
# TRANSFORM
#======================================================================#
class FinaltimeDatasetTransform(DatasetTransform):
    def __call__(self, graph):

        pos, disp, vmstr, temp, edge_dxyz = self.normalize_fields(graph)

        # only consider z disp
        disp = disp[:, 2:]

        # features / labels
        xs = [pos,]
        ys = []

        if self.sdf:
            sdf_x = self.normalize_sdf_x(graph.sdf_x)
            xs.append(sdf_x)
        if self.disp:
            ys.append(disp)
        if self.vmstr:
            ys.append(vmstr)
        if self.temp:
            ys.append(temp)

        assert len(ys) == self.nfields, f"At least one of disp, vmstr, temp must be True. Got {self.disp}, {self.vmstr}, {self.temp}."

        x = torch.cat(xs, dim=-1)
        y = torch.cat(ys, dim=-1)

        edge_attr = edge_dxyz
        data = self.make_pyg_data(graph, edge_attr, x=x, y=y)

        return data

#======================================================================#
# FINALTIME DATASET
#======================================================================#
class FinaltimeDataset(pyg.data.Dataset):
    def __init__(
        self, root, subdirs=None, transform=None, force_reload=False,
        num_workers=None, include_list=None,
    ):
        if num_workers is None:
            self.num_workers = mp.cpu_count() // 2
        else:
            self.num_workers = num_workers

        if subdirs is not None:
            self.subdirs = subdirs
        else:
            self.subdirs = [d for d in sorted(os.listdir(root)) if d.startswith('data_')]

        # full path to subdirs and case files
        self.subdirs = [os.path.join(root, d) for d in self.subdirs]
        self.case_files = [os.path.join(d, f) for d in self.subdirs for f in sorted(os.listdir(d)) if f.endswith('.npz')]

        if include_list is not None:
            include_list = [e + '.npz' for e in include_list]
            self.case_files = [c for c in self.case_files if os.path.basename(c) in include_list]

        if check_package_version_lteq('torch', '2.4.0'):
            assert not force_reload, "force_reload is not supported for torch < 2.4.0. You gotta nuke the .pt files yourself."
            super().__init__(root, transform=transform)
        else:
            super().__init__(root, transform=transform, force_reload=force_reload)

    @property
    def raw_paths(self):
        return self.case_files

    @property
    def processed_paths(self):
        processed_dir = os.path.join(self.root, "processed")
        case_files = [f"case{str(i).zfill(5)}_{os.path.basename(c)[:-4]}.pt" for i, c in enumerate(self.case_files)]
        return [os.path.join(processed_dir, case) for case in case_files]

    #-------------------#
    # OLD PYG VERSION
    #-------------------#
    @property
    def processed_dir(self):
        return os.path.join(self.root, 'processed')

    @property
    def processed_file_names(self):
        return self.processed_paths()

    @property
    def raw_file_names(self):
        return self.raw_paths()
    #-------------------#

    def process(self):
        num_cases = len(self.case_files)
        icases = range(num_cases)

        # for icase in tqdm(range(num_cases)):
        #     self.process_single(icase)

        mp.set_start_method('spawn', force=True)
        with mp.Pool(self.num_workers) as pool:
            list(tqdm(
                pool.imap_unordered(self.process_single, icases), total=num_cases,
                desc=f'FinaltimeDataset',
                ncols=80,
            ))

        return

    def process_single(self, icase):
        data = np.load(self.raw_paths[icase], mmap_mode='r')
        case_name = os.path.basename(self.case_files[icase])[:-4]
        graph = makegraph(data, case_name, 1)
        torch.save(graph, self.processed_paths[icase])
        del data, graph
        return

    def len(self):
        return len(self.case_files)

    def get(self, idx):
        path = self.processed_paths[idx]
        if check_package_version_lteq('torch', '2.4'):
            graph = torch.load(path)
        else:
            graph = torch.load(path, weights_only=False)
        return graph

#======================================================================#
#