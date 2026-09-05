"""Portable adapters around the previously tested numerical core."""
from pathlib import Path
import numpy as np
from scipy import ndimage
from .io import *
from .segmentation_measurement_3d import run_case,world,sample_slice

# ToothSeg 32-tooth convention. This is not the ToothFairy3 anatomy map.
TOOTHSEG_TO_FDI={i+1:fdi for i,fdi in enumerate(
    list(range(21,29))+list(range(11,19))+list(range(41,49))+list(range(31,39)))}


def load_multilabel(config_path):
    c,base=read_config(config_path)
    tf=load_volume(input_path(c,'toothfairy3_nifti',base))
    ts_path=input_path(c,'toothseg_nifti',base,False)
    ts=load_volume(ts_path) if ts_path else None
    ref_path=input_path(c,'reference_cbct_nifti',base,False)
    ref=load_volume(ref_path) if ref_path else None
    units=verify_geometry([tf]+([ts] if ts is not None else []),ref)
    data=label_array(tf)
    component_map,n=ndimage.label(data==int(c.get('implant_label',10)))
    components=[]
    for k in range(1,n+1):
        vox=np.argwhere(component_map==k)
        components.append({'component_id':k,'voxel_count':len(vox),'centroid_ras':world(vox,tf.affine).mean(0).tolist()})
    return c,tf,ts,ref,data,component_map,components,units


def inspect_case(config_path,output):
    c,tf,ts,ref,data,parts,components,units=load_multilabel(config_path)
    report={'shape':list(tf.shape),'affine':tf.affine.tolist(),
        'physical_unit_provenance':units,'implant_components':components,
        'selection_note':'Component IDs are computational identifiers, not confirmed FDI numbers.'}
    save_json(output,report);return report


def prepare_case(config_path,output_directory):
    c,tf,ts,ref,data,parts,components,units=load_multilabel(config_path)
    jaw=c.get('jaw')
    if jaw not in ('upper','lower'):raise ValueError('jaw must be upper or lower.')
    selected=c.get('target_component_id')
    if selected is None:
        if len(components)!=1:raise ValueError('Multiple or no implant components: run inspect and supply target_component_id.')
        selected=components[0]['component_id']
    if selected not in [r['component_id'] for r in components]:raise ValueError('target_component_id does not exist.')
    implant=parts==selected
    bone=data==int(c.get('jaw_label',2 if jaw=='upper' else 1))
    if not bone.any():raise ValueError('The selected jaw label is empty.')
    centers=c.get('same_jaw_tooth_centers_ras')
    tooth_rows=[]
    if ts is not None:
        teeth=label_array(ts)
        mapping=c.get('toothseg_label_to_fdi',TOOTHSEG_TO_FDI)
        for label,fdi in mapping.items():
            fdi=int(fdi)
            if fdi//10 not in [1,2,3,4] or fdi%10 not in range(1,9):raise ValueError('Invalid FDI in toothseg_label_to_fdi.')
            if (fdi//10 in (1,2))!=(jaw=='upper'):continue
            vox=np.argwhere(teeth==int(label))
            if len(vox):tooth_rows.append({'fdi':fdi,'center_ras':world(vox,ts.affine).mean(0).tolist()})
        if centers is None:centers=[r['center_ras'] for r in tooth_rows]
    out=Path(output_directory);out.mkdir(parents=True,exist_ok=True)
    save_mask(out/'target_implant.nii.gz',implant,tf.affine)
    save_mask(out/'jaw.nii.gz',bone,tf.affine)
    keys=['case_id','jaw','platform_ras','axis_apical_ras','arch_tangent_ras','buccal_hint_ras','max_search_mm','orientation_source','target_fdi']
    request={k:c[k] for k in keys if k in c}
    request.update(target_implant_mask_nifti='target_implant.nii.gz',jaw_mask_nifti='jaw.nii.gz')
    if centers is not None:request['same_jaw_tooth_centers_ras']=centers
    save_json(out/'request.json',request)
    save_json(out/'preparation.json',{'physical_unit_provenance':units,'target_component_id':selected,
        'target_fdi':c.get('target_fdi'),'target_assignment':'caller_supplied_not_automatically_validated',
        'tooth_centers':tooth_rows,'implant_components':components})
    return out/'request.json'


def measure_case(config_path,output_directory,plot=False):
    c,base=read_config(config_path)
    ni=load_volume(input_path(c,'target_implant_mask_nifti',base))
    nb=load_volume(input_path(c,'jaw_mask_nifti',base))
    refpath=input_path(c,'reference_cbct_nifti',base,False)
    ref=load_volume(refpath) if refpath else None
    units=verify_geometry([ni,nb],ref)
    iv,bv=binary_array(ni),binary_array(nb)
    result,section=run_case(iv,bv,ni.affine,c,c.get('same_jaw_tooth_centers_ras'))
    if c.get('case_id') is not None:result['case_id']=c['case_id']
    result['physical_unit_provenance']=units
    out=Path(output_directory);out.mkdir(parents=True,exist_ok=True)
    if section is not None:
        np.save(out/'segmentation_slice.npy',section)
        if ref is not None:
            g=result['geometry']
            raw,_=sample_slice(np.asarray(ref.dataobj),ref.affine,
                np.asarray(g['platform_ras']),np.asarray(g['axis_apical_ras']),np.asarray(g['buccal_ras']),order=1)
            np.save(out/'cbct_slice.npy',raw)
        if plot:
            from .visualization import plot_measurement
            plot_measurement(result,section,out,raw if ref is not None else None)
    save_json(out/'measurement_3d.json',result)
    return result
