#
import torch
import numpy as np
from tqdm import tqdm

import torch.multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, as_completed

import os
import json
import struct
import zipfile
import shutil
import collections
import io

__all__ = [
    'extract_zips',
    'extract_from_zip',
]

#=================================#
# Based on Andrew Porco's
# https://github.com/andrewporco/netfabb//netfabb-tools/extract-stress-strain.py
# which in turn is based on Kevin Ferguson's and Sean Chen's code
# https://github.com/kevinferg/netfabb-tools/extract-netfabb-data-binary.py
#=================================#

#=================================#
# read files
#=================================#

def get_basefile(full_path):
    return os.path.normpath(os.path.basename(full_path))

def extract_frames_from_case(casefile): # mechanical.case
    with open(casefile,'r') as f:
        lines = f.readlines()

    nframes = []
    for line in lines:
        if "number of steps:" in line:
            fr = int(line[17:])
            nframes.append(fr)
            
    return nframes

def read80(f):
    return f.read(80).decode('utf-8').strip()

def read_floats(f,N):
    arr = np.array(struct.unpack(f"<{N}f", f.read(4*N)))
    return arr[0] if N == 1 else arr

def read_ints(f,N):
    arr = np.array(struct.unpack(f"<{N}i", f.read(4*N)))
    return arr[0] if N == 1 else arr

def read_geo_binary(file_or_path):
    # Handle both file paths and file-like objects
    if isinstance(file_or_path, (str, bytes, os.PathLike)):
        with open(file_or_path, 'rb') as f:
            return _read_geo_binary(f)
    else:
        # Assume it's a file-like object
        return _read_geo_binary(file_or_path)

def _read_geo_binary(f):
    ############ Info
    assert read80(f) == 'Fortran Binary'
    description1 = read80(f)
    description2 = read80(f)
    assert read80(f) == 'node id off'
    assert read80(f) == 'element id off'
    # extents_str = read80(f)       # No extents
    # extents = read_floats(f,6)    # No extents

    ############ Begin part 1
    assert read80(f) == 'part'
    assert read_ints(f,1) == 1 # Should be 1 part only
    description3 = read80(f)

    ############ Coordinates
    assert read80(f) == 'coordinates'
    nn = read_ints(f,1)
    # node_ids = read_ints(f,nn) # node id is off
    x = read_floats(f,nn)
    y = read_floats(f,nn)
    z = read_floats(f,nn)
    nodes = np.vstack([x,y,z]).T

    ############ Elements
    element_type = read80(f)
    assert(element_type == 'hexa8')
    # element_ids = read_ints(f,nn) # element id is off
    ne = read_ints(f, 1)
    elems = read_ints(f, 8*ne).reshape(ne,8)

    data = dict(description1=description1, description2=description2, description3=description3, 
                nn=nn, nodes=nodes, element_type=element_type, ne=ne, elems=elems)
    return data


def read_ens_binary(file_or_path, num_nodes, dim):
    # Handle both file paths and file-like objects
    if isinstance(file_or_path, (str, bytes, os.PathLike)):
        with open(file_or_path, 'rb') as f:
            return _read_ens_binary(f, num_nodes, dim)
    else:
        # Assume it's a file-like object
        return _read_ens_binary(file_or_path, num_nodes, dim)

def _read_ens_binary(f, num_nodes, dim):
    description = read80(f)
    assert(read80(f) == 'part')
    assert(read_ints(f, 1) == 1)
    assert(read80(f) == 'coordinates')
    arr = read_floats(f, num_nodes * dim)
    data = arr.reshape(dim, num_nodes).T
    return dict(description=description, data=data)

def get_vertices_from_geo(zip_path, internal_path, return_elements=False):
    internal_path = internal_path.lstrip('/')  # Normalize path
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        with zip_ref.open(internal_path) as f:
            data = read_geo_binary(f)

    if not return_elements:
        return data["nodes"]
    else:
        return data["nodes"], data["elems"]

def get_values_from_ens(zip_path, internal_path, num_nodes, dim):
    internal_path = internal_path.lstrip('/')  # Normalize path
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        with zip_ref.open(internal_path) as f:
            data = read_ens_binary(f, num_nodes, dim)
    return data["data"]

#=================================#
# grab results
#=================================#

def get_stl_mesh(zip_path, stl_path):
    import trimesh

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        with zip_ref.open(stl_path) as f:
            # Read the file content into a BytesIO object
            file_content = io.BytesIO(f.read())
            # Load the mesh from the BytesIO object
            mesh = trimesh.load(file_content, file_type='stl')

    return mesh.vertices, mesh.faces

def get_case_info(casedir):
    # Split the path into zip file and internal path
    if '.zip/' in casedir:
        zip_path, internal_path = casedir.split('.zip/', 1)
        zip_path += '.zip'  # Add back the .zip extension
        internal_path = internal_path.rstrip('/')  # Remove trailing slash if present
        
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            basename = os.path.basename(internal_path) + "_"
            
            # Check for mechanical.case file
            case_file_path = f"{internal_path}/results/{basename}mechanical.case"
            if case_file_path not in zip_ref.namelist():
                basename = ""
                case_file_path = f"{internal_path}/results/mechanical.case"
                if case_file_path not in zip_ref.namelist():
                    return dict(error="Error preparing simulation.")

            # Check mechanical.out for termination
            mech_out_path = f"{internal_path}/{basename}mechanical.out"
            if mech_out_path in zip_ref.namelist():
                with zip_ref.open(mech_out_path) as f:
                    if 'Analysis terminated' in f.read().decode('utf-8'):
                        return dict(error="Error running simulation.")
            
            # Get frame count from case file
            with zip_ref.open(case_file_path) as f:
                lines = f.read().decode('utf-8').splitlines()
                nframes = []
                for line in lines:
                    if "number of steps:" in line:
                        fr = int(line[17:])
                        nframes.append(fr)

            if nframes == []:
                return dict(error=f"Case file missing frame count. {casedir}")
            if len(nframes) != 2:
                return dict(error=f"Case file with incorrect frame count. {casedir}")

            frame_count, semi_frame_count = nframes
            if (frame_count % 2) != 0:
                return dict(error=f"Frame count must be even. Got {frame_count} in {casedir}.")
            if (semi_frame_count - 2) != (frame_count - 2) // 2:
                return dict(error=f"Got incompatible frame counts, {frame_count}, {semi_frame_count} in {casedir}.")
            if frame_count > 100:
                return dict(error=f"Frame count is too large in {casedir}. Got {frame_count} > 100.")

            basefilename = f"{internal_path}/results/{basename}mechanical00_{frame_count}."
            geo_file = f"{internal_path}/results/{basename}mechanical_{semi_frame_count}.geo"
            if geo_file not in zip_ref.namelist():
                return dict(error="Missing Geo file.")

            geo_files = []
            base_names = []

            for i in range(1, semi_frame_count - 1):
                geo_path = f"{internal_path}/results/{basename}mechanical_{i}.geo"
                if geo_path not in zip_ref.namelist():
                    return dict(error=f"Geo file {geo_path} not found.")
                geo_files.append(geo_path)

                ii = 2 * i
                base_name = f"{internal_path}/results/{basename}mechanical00_{ii}"
                dis_path = base_name + '.dis.ens'
                if dis_path not in zip_ref.namelist():
                    return dict(error=f"Base file {dis_path} not found.")
                base_names.append(base_name)

            assert len(geo_files) == len(base_names)

            # Grab files corresponding to all time-steps
            # NetFabb write state twice for every layer. This is why there are ~twice
            # as many `base_names` as `geo_files`. We want to grab the later file.
            # 
            # For both geo_files and base_names, the last two files correspond to 
            # "cooling" and "substrate removal". We will skip those files for now.
            #
            # Based on mechanical.out, thermal.out

            # for i in range(1, semi_frame_count + 1):
            #     geo_path  = os.path.join(casedir, 'results', f'{basename}mechanical_{i}.geo')
            #     assert os.path.exists(geo_path ), f"Geo file {geo_path} not found."
            #     geo_files.append(geo_path)
            #
            # for i in range(frame_count):
            #     ii = i+1
            #     base_name = os.path.join(casedir, 'results', f'{basename}mechanical00_{ii}')
            #     dis_path = base_name + '.dis.ens'
            #     assert os.path.exists(dis_path), f"Base file {dis_path} not found."
            #     base_names.append(base_name)
            
            # grab part stl file
            stl_path = f"{internal_path}/test.stl"

            return dict(
                zip_path=zip_path,
                internal_path=internal_path,
                stl_path=stl_path,
                geo=geo_file, 
                basefilename=basefilename,
                geo_files=geo_files, 
                base_names=base_names,
            )
    else:
        return dict(error=f"Case directory {casedir} is not a zip file.")

def get_finaltime_results(casedir):
    case_info = get_case_info(casedir)
    if len(case_info) == 1:
        return [case_info["error"],] # Case failed
    
    verts, elems = get_vertices_from_geo(case_info["zip_path"], case_info["geo"], return_elements=True)
    N_verts = verts.shape[0]

    result_types = dict(
        dis="disp", ept="strain", rcd="recoater_status", rct="recoater_clearance_percent",
        sd1="max_dir", sd2="mid_dir", sd3="min_dir", sig="cauchy_stress", sp1="max_stress",
        sp2="mid_stress", sp3="min_stress", svm="von_mises_stress", tmp="temp",
    )
    result_nums = dict(dis=3, ept=6, rcd=1, rct=1, sd1=3, sd2=3, sd3=3, sig=6, sp1=1, sp2=1, sp3=1, svm=1, tmp=1)

    results = dict(verts=verts, elems=elems)
    for key in result_types:
        result = get_values_from_ens(
            case_info["zip_path"],
            case_info["basefilename"] + key + ".ens",
            N_verts,
            result_nums[key],
        )
        results[result_types[key]] = result.astype(np.float32)
        
    stl_verts, stl_faces = get_stl_mesh(case_info["zip_path"], case_info["stl_path"])
    results["stl_verts"] = stl_verts.astype(np.float32)
    results["stl_faces"] = stl_faces.astype(np.int32)

    return results

def get_timeseries_results(casedir):
    case_info = get_case_info(casedir)
    if len(case_info) == 1:
        return [case_info["error"],] # Case failed

    results = collections.defaultdict(list) # initializes every item to []
    fields  = dict(dis=("disp", 3), svm=("von_mises_stress", 1), tmp=("temp", 1))

    for geo_file in case_info['geo_files']:
        v, e = get_vertices_from_geo(case_info["zip_path"], case_info["geo"], return_elements=True)
        v = v.astype(np.float32)
        e = e.astype(np.int32)
        results['verts'].append(v) # [Nv, 3]
        results['elems'].append(e) # [Ne, 8]

    for (i, base_name) in enumerate(case_info['base_names']):
        nv = results['verts'][i].shape[0]
        for key in fields:
            path = base_name + f'.{key}.ens'
            field, dim = fields[key]
            val = get_values_from_ens(
                case_info["zip_path"],
                case_info["basefilename"] + key + ".ens",
                nv,
                dim
            )
            val = val.astype(np.float32)
            results[field].append(val)

    stl_verts, stl_faces = get_stl_mesh(case_info["zip_path"], case_info["stl_path"])
    results["stl_verts"] = stl_verts.astype(np.float32)
    results["stl_faces"] = stl_faces.astype(np.int32)

    return results

#=================================#
# Process results and put them in the right spot
#=================================#

class Processor:
    def __init__(self, data_dir, out_dir, timeseries):
        self.data_dir = data_dir
        self.out_dir = out_dir
        self.timeseries = timeseries
        if self.timeseries:
            self.result_func = get_timeseries_results
        else:
            self.result_func = get_finaltime_results
        return

    def __call__(self, case):
        casedir = os.path.join(self.data_dir, case)
        results = self.result_func(casedir)
        case = os.path.basename(case.rstrip('/'))
        if len(results) == 1:
            return False, [case], dict() # succ/fail, fail-case, series_dict
        else:
            if self.timeseries:
                out_path = os.path.join(self.out_dir, case + '.pt')
                torch.save(results, out_path)
                N = len(results['verts'])
                del results
                return True, [], {case : N}
            else:
                out_path = os.path.join(self.out_dir, case + '.npz')
                np.savez(out_path, **results)
                del results
                return True, [], dict()

def extract_from_dir(data_dir, out_dir, timeseries=None, num_workers=None):
    os.makedirs(out_dir, exist_ok=True)

    if num_workers is None:
        num_workers = mp.cpu_count() // 2

    assert data_dir.endswith('.zip')

    with zipfile.ZipFile(data_dir, 'r') as zip_ref:
        cases = [f for f in zip_ref.namelist() 
                 if f.startswith('SandBox/') and f.endswith('/') and f.count('/') == 2]
    
    print(f"Loading {len(cases)} cases from: {data_dir} into {out_dir}")

    processor = Processor(data_dir, out_dir, timeseries)

    mp.set_start_method('spawn', force=True)
    with mp.Pool(num_workers) as pool:
        outlist = list(tqdm(pool.imap_unordered(processor, cases), total=len(cases), ncols=80))

    num_success = sum([out[0] for out in outlist])
    error_list = [o for out in outlist for o in out[1]]
    series_dict = {k:v for out in outlist for (k, v) in out[2].items()}

    num_failure = len(cases) - num_success

    error_file = os.path.join(out_dir, 'error.txt')
    with open(error_file, 'w') as file:
        for case in error_list:
            file.write(f'{case}\n')

    print(f"Successfully saved {num_success} / {num_success + num_failure} cases to NPZ/PT format.")
    if timeseries:
        series_file = os.path.join(out_dir, 'series.json')
        with open(series_file, 'w') as file:
            json.dump(series_dict, file)
        print(f"Saved number of frames per case to {series_file}")
    print(f"Failed simulation cases are logged to {error_file}")

    return

#=================================#
# assemble data
#=================================#
def extract_from_zip(source_zip, target_dir, timeseries=None, num_workers=None):
    os.makedirs(target_dir, exist_ok=True)
    extract_from_dir(source_zip, target_dir, timeseries=timeseries, num_workers=num_workers)
    return

def extract_zips(source_dir, target_dir, timeseries=None, num_workers=None):
    os.makedirs(target_dir, exist_ok=True)
    zip_names = [f for f in os.listdir(source_dir) if f.endswith('.zip')]

    for zip_name in zip_names:
        zip_file = os.path.join(source_dir, zip_name)
        out_dir  = os.path.join(target_dir, zip_name[:-4])

        if os.path.exists(out_dir):
            shutil.rmtree(out_dir)
        os.makedirs(out_dir)
        extract_from_zip(zip_file, out_dir, timeseries=timeseries,  num_workers=num_workers)

    return
#=================================#
#