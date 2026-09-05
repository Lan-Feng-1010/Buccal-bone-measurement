"""Configuration-relative input paths and explicit NIfTI geometry validation."""
import json
from pathlib import Path
import nibabel as nib
import numpy as np


def read_config(path):
    file=Path(path).expanduser().resolve()
    config=json.loads(file.read_text(encoding='utf-8-sig'))
    if not isinstance(config,dict):raise ValueError('Configuration must be a JSON object.')
    return config,file.parent


def input_path(config,key,base,required=True):
    value=config.get(key)
    if value is None:
        if required:raise ValueError(f'Missing input path: {key}')
        return None
    path=Path(value).expanduser()
    if not path.is_absolute():path=base/path
    if not path.is_file():raise FileNotFoundError(f'Input does not exist: {key} ({path})')
    return path.resolve()


def save_json(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')


def load_volume(path):
    image=nib.load(path)
    if len(image.shape)!=3:raise ValueError('A three-dimensional NIfTI is required.')
    if not np.isfinite(image.affine).all():raise ValueError('NIfTI affine must be finite.')
    return image


def matched_grid(first,second):
    return first.shape==second.shape and np.allclose(first.affine,second.affine,atol=1e-6,rtol=0)


def verify_geometry(images,reference=None):
    if not images:raise ValueError('No input images.')
    if any(not matched_grid(images[0],n) for n in images[1:]):
        raise ValueError('Input masks have different shapes or affines; implicit resampling is not allowed.')
    if reference is not None and not matched_grid(images[0],reference):
        raise ValueError('Reference CBCT and masks do not share the same physical grid.')
    units=[n.header.get_xyzt_units()[0] for n in images]
    if any(u not in ('mm','unknown') for u in units):
        raise ValueError('Mask spatial units must be mm, or unknown with a matched mm reference.')
    if reference is not None and reference.header.get_xyzt_units()[0]!='mm':
        raise ValueError('Reference CBCT must explicitly use mm units.')
    if all(u=='mm' for u in units):return 'mask_headers_mm'
    if reference is None:raise ValueError('Unknown mask units require reference_cbct_nifti with the same grid and mm units.')
    return 'inherited_from_matched_raw_CBCT_mm_header'


def binary_array(image):
    array=np.asarray(image.dataobj)
    if not np.isfinite(array).all() or not np.isin(array,[0,1]).all():
        raise ValueError('Measurement inputs must be finite binary masks (0/1). Use prepare for multi-label inputs.')
    return array.astype(bool)


def label_array(image):
    array=np.asarray(image.dataobj)
    if not np.isfinite(array).all() or np.any(array<0) or not np.equal(array,np.floor(array)).all():
        raise ValueError('A segmentation must contain finite, nonnegative integer labels.')
    return array


def save_mask(path,array,affine):
    image=nib.Nifti1Image(np.asarray(array,dtype=np.uint8),affine)
    image.header.set_xyzt_units('mm');nib.save(image,path)
