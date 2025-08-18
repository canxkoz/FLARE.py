#
import torch
import torch_geometric as pyg
import torch.multiprocessing as mp

import scipy
from tqdm import tqdm

import os
import copy
import json
from typing import Union

from mlutils.utils import (to_numpy, check_package_version_lteq)
from .utils import (timeseries_dataset, merge_timeseries)
from .transform import DatasetTransform

__all__ = [
    'TimeseriesDatasetTransform',
    'TimeseriesDataset',
    'split_timeseries_dataset',
]

#======================================================================#
# TRANSFORM
#======================================================================#
class TimeseriesDatasetTransform(DatasetTransform):
    def __init__(
        self,
        disp=True, vmstr=True, temp=True,
        sdf=False, mesh=True, elems=False, orig=False, metadata=False,
        merge=True, interpolate=False,
    ):

        super().__init__(
            disp=disp, vmstr=vmstr, temp=temp,
            sdf=sdf, mesh=mesh, elems=elems, orig=orig, metadata=metadata,
        )

        self.merge = merge
        self.interpolate = interpolate

        return

    def makefields(self, data, istep, scale=False):
        '''
        used in am.time_march
        '''

        xs = []
        if self.merge:
            xs = [*xs, data.disp[istep, :, 2].reshape(-1,1)] if self.disp  else xs
            xs = [*xs, data.vmstr[istep, :].reshape(-1,1)  ] if self.vmstr else xs
            xs = [*xs, data.temp[istep, :].reshape(-1,1)   ] if self.temp  else xs
        else:
            xs = [*xs, data.disp[:,2].reshape(-1,1)] if self.disp  else xs
            xs = [*xs, data.vmstr.reshape(-1,1)    ] if self.vmstr else xs
            xs = [*xs, data.temp.reshape(-1,1)     ] if self.temp  else xs

        out = torch.cat(xs, dim=-1)
        
        if scale:
            out = out / self.scale.to(xs[0].device).view(-1, 1)

        return out

    @torch.no_grad()
    def interpolate_layer(self, u: torch.tensor, graph, istep: int, tol=1e-4):
        '''
        fill `u` in between `istep` and `istep+1`
        '''

        if not self.merge:
            return u

        # ensure not final step
        assert istep + 1 != graph.metadata['time_steps']

        z = graph.pos[:,2].reshape(-1)
        zmax = graph.metadata['zmax']
        z0 = zmax[istep-1] if istep != 0 else 0
        z1 = zmax[istep]
        z2 = zmax[istep+1]

        idx0 = (z0 - tol <= z) * (z <= z1 + tol) # [istep-1, istep]
        idx1 = (z1 + tol <  z) * (z <= z2 + tol) # (istep, istep+1]

        out = scipy.interpolate.griddata(
            graph.pos[idx0].numpy(force=True),
            u[idx0].numpy(force=True),
            graph.pos[idx1].numpy(force=True),
            method='nearest',
        )

        out = scipy.interpolate.griddata(
            to_numpy(graph.pos[idx0]),
            to_numpy(u[idx0]),
            to_numpy(graph.pos[idx1]),
            method='nearest',
        )

        u.clone()
        u[idx1, :] = torch.tensor(out, dtype=torch.float, device=u.device)

        return u

    def __call__(self, graph, tol=1e-4):

        N  = graph.pos.size(0)
        md = graph.metadata
        istep  = md['time_step'] # zero indexed
        nsteps = md['time_steps']

        last_step  = (istep + 1) == nsteps
        
        # interface mask (hide inactive layers)
        if not last_step:
            zm = md['zmax'][istep+1]
            mask = graph.pos[:,2] <= (zm + tol)
        else:
            mask = torch.full((N,), True)

        # normalize fields
        pos, disp, vmstr, temp, edge_dxyz = self.normalize_fields(graph)

        # time
        if nsteps == 1:
            t, dt = 1., 0.
        else:
            T_val  = md['zmax'][-1] / self.pos_scale[2]
            dt_val = T_val / (nsteps - 1)
            t_val  = istep * dt_val
        
        t  = torch.full((N, 1), t_val)
        T  = torch.full((N, 1), T_val)
        dt = torch.full((N, 1), dt_val)

        # fields (works for merged=True)
        if self.merge:
            if not last_step:

                disp0  = disp[ istep, :, 2].reshape(-1,1)
                vmstr0 = vmstr[istep, :, 0].reshape(-1,1)
                temp0  = temp[ istep, :, 0].reshape(-1,1)

                disp1  = disp[ istep+1, :, 2].reshape(-1,1)
                vmstr1 = vmstr[istep+1, :, 0].reshape(-1,1)
                temp1  = temp[ istep+1, :, 0].reshape(-1,1)

                if self.interpolate:
                    disp0  = self.interpolate_layer(disp0,  graph, istep, tol=tol)
                    vmstr0 = self.interpolate_layer(vmstr0, graph, istep, tol=tol)
                    temp0  = self.interpolate_layer(temp0,  graph, istep, tol=tol)

                disp_in  = disp0
                vmstr_in = vmstr0
                temp_in  = temp0

                disp_out  = disp1  - disp0
                vmstr_out = vmstr1 - vmstr0
                temp_out  = temp1  - temp0

            else:
                disp_in  = torch.zeros((N, 1))
                disp_out = torch.zeros((N, 1))

                vmstr_in  = torch.zeros((N, 1))
                vmstr_out = torch.zeros((N, 1))

                temp_in  = torch.zeros((N, 1))
                temp_out = torch.zeros((N, 1))
        else:
            # look back
            raise NotImplementedError("TimeseriesTransform not implemented for merge=False")

        # features / labels
        xs = [pos,]
        ys = []

        if self.sdf:
            sdf_x = self.normalize_sdf_x(graph.sdf_x)
            xs.append(sdf_x)
        if self.disp:
            xs.append(disp_in)
            ys.append(disp_out)
        if self.vmstr:
            xs.append(vmstr_in)
            ys.append(vmstr_out)
        if self.temp:
            xs.append(temp_in)
            ys.append(temp_out)

        assert len(ys) == self.nfields, f"At least one of disp, vmstr, temp must be True. Got {self.disp}, {self.vmstr}, {self.temp}."

        x = torch.cat(xs, dim=-1)
        y = torch.cat(ys, dim=-1)

        edge_attr = edge_dxyz
        data = self.make_pyg_data(
            graph,
            edge_attr,
            x=x, y=y,
            t=t, T=T, dt=dt,
            t_val=t_val, T_val=T_val, dt_val=dt_val,
            mask=mask,
        )
        
        del graph

        return data

#======================================================================#
# TIMESERIES DATASET
#======================================================================#
class TimeseriesDataset(pyg.data.Dataset):
    def __init__(
        self, roots, transform=None, force_reload=False,
        merge=None, num_workers=None, exclude_list=None,
        verbose=True,
    ):
        """
        Create dataset of time-series

        Arguments:
        - `roots`: list of root directories containing case files
        - `merge`: return fields on graph made by merging all the timeseries
        meshes.
        """
        self.num_workers = mp.cpu_count() // 2 if num_workers is None else num_workers

        self.merge = merge
        self.roots = [roots] if isinstance(roots, str) else roots
        
        assert isinstance(self.roots, list)

        # Collect case files from all roots
        self.case_files = []
        for root in self.roots:
            cases = [os.path.join(root, c) for c in sorted(os.listdir(root)) if c.endswith('.pt')]
            self.case_files.extend(cases)
            
        n0 = len(self.case_files)

        if exclude_list is not None:
            exclude_list = [e + '.pt' for e in exclude_list]
            self.case_files = [c for c in self.case_files if os.path.basename(c) not in exclude_list]

        n1 = len(self.case_files)
        
        if verbose:
            print(f"Excluded {n0 - n1} / {n0} cases based on exclude_list.")

        # Load time steps from all series.json files
        time_step_dict = {}
        for root in self.roots:
            with open(os.path.join(root, 'series.json')) as file:
                time_step_dict.update(json.load(file))
                
        self.time_steps = torch.tensor(
            [time_step_dict[os.path.basename(case_file)[:-3]] for case_file in self.case_files])
        self.time_steps_cum = self.time_steps.cumsum(0)

        if check_package_version_lteq('torch', '2.4.0'):
            super().__init__(transform=transform)
        else:
            super().__init__(transform=transform, force_reload=force_reload)

    @property
    def raw_paths(self):
        return self.case_files

    @property
    def processed_paths(self):
        processed_dirname = 'processed_merged' if self.merge else 'processed'
        for root in self.roots:
            os.makedirs(os.path.join(root, processed_dirname), exist_ok=True)
        return [os.path.join(
            os.path.dirname(case_file), processed_dirname, os.path.basename(case_file)
        ) for case_file in self.case_files]

    #-------------------#
    # OLD PYG VERSION
    #-------------------#
    @property
    def processed_dir(self):
        return os.path.join(self.roots[0], 'processed')

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
                desc=f'Processing TimeseriesDataset',
                ncols=80,
            ))

        return

    def process_single(self, icase):
        case_file = self.case_files[icase]
        dataset = timeseries_dataset(case_file)
        if self.merge:
            graph = merge_timeseries(dataset, os.path.basename(case_file)[:-3])
            torch.save(graph, self.processed_paths[icase])
            del graph
        else:
            torch.save(dataset, self.processed_paths[icase])

        del dataset
        return

    def len(self):
        return sum(self.time_steps)

    def case_range(self, case: Union[int, str]):
        if isinstance(case, int):
            i0 = 0 if case == 0 else self.time_steps_cum[case-1]
            i1 = self.time_steps_cum[case]
            assert i1-i0 == self.time_steps[case]
            return range(i0, i1)
        else: # str
            for icase, case_file in enumerate(self.case_files):
                if case in case_file:
                    break
            else:
                raise ValueError(f"Case '{case}' not found in case_files.")
            return self.case_range(icase)

    def get(self, idx):
        # get case and time step
        # icase = torch.argwhere(idx < self.time_steps_cum)[0].item() # not 1.10 compatible
        icase = torch.nonzero(idx < self.time_steps_cum)[0].item()
        nprev = 0 if icase == 0 else self.time_steps_cum[icase-1].item()
        time_step  = idx - nprev
        time_steps = self.time_steps[icase]

        # get graph
        path = self.processed_paths[icase]

        if check_package_version_lteq('torch', '2.4'):
            graph = torch.load(path)
        else:
            graph = torch.load(path, weights_only=False, mmap=True)

        if not self.merge:
            graph = graph[time_step]

        assert time_steps == graph.metadata['time_steps'] == len(graph.metadata['zmax']) > 0

        # write time step to graph (zero indexed)
        graph.metadata['time_step'] = time_step

        return graph

#======================================================================#
# SPLIT TIMESERIES DATASET
#======================================================================#
def split_timeseries_dataset(dataset, split=None, indices=None):
    if split is None and indices is None:
        raise ValueError('split_timeseries_dataset: pass in either indices or split')

    num_cases = len(dataset.case_files)

    if indices is None:
        indices = [int(s * num_cases) for s in split]
        indices[-1] += num_cases - sum(indices)

    indices = torch.utils.data.random_split(range(num_cases), indices)

    num_split = len(indices)

    # deepcopy
    subsets = [copy.deepcopy(dataset) for _ in range(num_split)]

    for s in range(num_split):
        idxs   = list(indices[s])
        subset = subsets[s]

        subset.case_files = [subset.case_files[idx] for idx in idxs]
        subset.time_steps = torch.tensor([subset.time_steps[idx].item() for idx in idxs])
        subset.time_steps_cum = subset.time_steps.cumsum(0)

    return subsets

#======================================================================#
#
