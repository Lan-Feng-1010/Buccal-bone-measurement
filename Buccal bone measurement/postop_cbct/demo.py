"""Small synthetic NIfTI example: contains no clinical images or identifiers."""
from pathlib import Path
import numpy as np
from .io import save_json,save_mask
from .pipeline import measure_case


def create_demo(output):
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    implant=np.zeros((40,40,50),dtype=np.uint8);implant[18:23,18:23,10:31]=1
    bone=np.zeros_like(implant);bone[24:30,15:26,0:46]=1
    affine=np.diag([.5,.5,.5,1.])
    save_mask(out/'target_implant.nii.gz',implant,affine);save_mask(out/'jaw.nii.gz',bone,affine)
    config={'target_implant_mask_nifti':'target_implant.nii.gz','jaw_mask_nifti':'jaw.nii.gz',
        'jaw':'upper','platform_ras':[10,10,5],'axis_apical_ras':[0,0,1],
        'arch_tangent_ras':[0,1,0],'buccal_hint_ras':[1,0,0],
        'orientation_source':'synthetic_known_geometry_not_clinical_data'}
    save_json(out/'request.json',config)
    save_json(out/'expected.json',{'source':'synthetic_geometry_only','distance_mm':[4.75]*4,'automatic_release':False})
    return out/'request.json'


def run_demo(output,plot=False):
    request=create_demo(output)
    result=measure_case(request,Path(output)/'result',plot)
    np.testing.assert_allclose([r['distance_mm'] for r in result['measurements']],[4.75]*4,atol=1e-8)
    return result
