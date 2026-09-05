"""3-D segmentation-start prototype, explicit RAS mm and auditable failure states.

No weights/training required. Segmentation and target/pose semantics are inputs;
automatic pose proposals are not a validated postoperative platform detector.
"""
import json,argparse
from pathlib import Path
import numpy as np
from scipy import ndimage


def unit(v):
    a=np.asarray(v,float)
    if a.shape!=(3,) or not np.isfinite(a).all() or np.linalg.norm(a)<1e-10:raise ValueError('invalid_direction')
    return a/np.linalg.norm(a)


def affine_check(affine):
    a=np.asarray(affine,float)
    if a.shape!=(4,4) or not np.isfinite(a).all() or not np.allclose(a[3],[0,0,0,1]) or abs(np.linalg.det(a[:3,:3]))<1e-12:
        raise ValueError('invalid_affine')
    return a


def world(ijk,a):return np.asarray(ijk)@a[:3,:3].T+a[:3,3]
def voxel(points,a):return (np.asarray(points)-a[:3,3])@np.linalg.inv(a[:3,:3]).T


def pose_proposal(implant,a,jaw):
    """SVD axis + coronal support plane candidate, NOT anatomical confirmation."""
    if jaw not in ('upper','lower'):raise ValueError('jaw_required')
    xyz=world(np.argwhere(implant),a)
    if len(xyz)<3:raise ValueError('implant_mask_too_small')
    center=xyz.mean(axis=0);_,sv,vh=np.linalg.svd(xyz-center,full_matrices=False);u=unit(vh[0])
    apical=np.array([0,0,1 if jaw=='upper' else -1])
    if abs(np.dot(u,apical))<.2:raise ValueError('axis_sign_needs_anatomical_reference')
    if np.dot(u,apical)<0:u=-u
    z=(xyz-center)@u
    half_support=.5*np.sum(np.abs(u@a[:3,:3]))
    p=center+u*(z.min()-half_support)
    return {'platform_candidate_ras':p.tolist(),'axis_candidate_ras':u.tolist(),
        'singular_values':sv.tolist(),'platform_method':'coronal_mask_support_estimate_not_validated',
        'axis_method':'SVD_implant_mask_oriented_by_jaw_RAS_superior',
        'extent_mm':float(z.max()-z.min()+2*half_support)}


def arch_tangent(tooth_centers_ras,target,allow_extrapolation_proposal=False):
    """Adapt fixed-rule PCA-XY polynomial, evaluate at actual implant not neighbor."""
    centers=np.asarray(tooth_centers_ras,float)
    if centers.ndim!=2 or centers.shape[1]!=3 or len(centers)<3 or not np.isfinite(centers).all():
        raise ValueError('at_least_three_same_jaw_teeth_required')
    xy=centers[:,:2];mu=xy.mean(axis=0);_,sv,vh=np.linalg.svd(xy-mu,full_matrices=False)
    uv=(xy-mu)@vh.T
    if np.ptp(uv[:,0])<1e-8:raise ValueError('degenerate_arch_points')
    degree=2 if len(centers)>=4 else 1
    if np.linalg.matrix_rank(np.vander(uv[:,0],degree+1))<degree+1:raise ValueError('arch_fit_rank_deficient')
    coeff=np.polyfit(uv[:,0],uv[:,1],degree)
    at=(np.asarray(target)[:2]-mu)@vh.T
    outside=bool(at[0]<uv[:,0].min() or at[0]>uv[:,0].max())
    if outside and not allow_extrapolation_proposal:raise ValueError('arch_extrapolation_needs_review')
    tangent=vh[0]+np.polyval(np.polyder(coeff),at[0])*vh[1]
    residual=float(np.sqrt(np.mean((np.polyval(coeff,uv[:,0])-uv[:,1])**2)))
    return unit([tangent[0],tangent[1],0]),{'method':'PCA_XY_polynomial_at_implant',
        'degree':degree,'rmse_mm':residual,'n_teeth':len(centers),'coefficients':coeff.tolist(),
        'status':'extrapolation_proposal_needs_review' if outside else 'interpolated_geometry_not_clinical_validation'}


def plane_basis(axis,tangent,buccal_hint):
    u=unit(axis);t=unit(tangent);n=unit(t-np.dot(t,u)*u);b=unit(np.cross(u,n))
    hint=unit(buccal_hint)
    if abs(np.dot(b,hint))<.2:raise ValueError('buccal_sign_not_resolvable')
    if np.dot(b,hint)<0:b=-b
    return u,b,n


def ray_voxel_intervals(shape,a,origin,direction,max_mm=20.):
    """Exact voxel-cell ray traversal: no step-size skipping of thin bone voxels."""
    a=affine_check(a);origin=np.asarray(origin,float);direction=unit(direction)
    if origin.shape!=(3,) or not np.isfinite(origin).all() or not np.isfinite(max_mm) or max_mm<=0:raise ValueError('invalid_ray')
    q=voxel(origin,a);d=np.linalg.solve(a[:3,:3],direction);size=np.array(shape)
    if np.any(q<-.5) or np.any(q>=size-.5):return [],'origin_out_of_fov'
    cuts=[0.,float(max_mm)];exit_t=float('inf')
    for k in range(3):
        if abs(d[k])<1e-12:continue
        boundary=size[k]-.5 if d[k]>0 else -.5
        exit_t=min(exit_t,float((boundary-q[k])/d[k]))
        times=(np.arange(size[k]+1)-.5-q[k])/d[k]
        cuts.extend(times[(times>1e-10)&(times<max_mm-1e-10)].tolist())
    stop=min(float(max_mm),exit_t)
    cuts=np.unique(np.round([c for c in cuts if 0<=c<=stop]+[stop],12))
    rows=[]
    for left,right in zip(cuts[:-1],cuts[1:]):
        if right-left<1e-10:continue
        idx=np.floor(q+(left+right)*.5*d+.5).astype(int)
        if np.all(idx>=0) and np.all(idx<size):rows.append((float(left),float(right),tuple(idx)))
    return rows,'fov_limit' if exit_t<=max_mm else 'search_limit'


def measure_ray(implant,bone,a,origin,direction,max_mm=20.):
    intervals,limit=ray_voxel_intervals(implant.shape,a,origin,direction,max_mm)
    base={'distance_mm':None,'bone_point_ras':None,'bone_intervals_mm':[],
          'method':'voxel_cell_intersections_not_subvoxel_anatomical_truth'}
    if not intervals:return dict(base,status=limit)
    states=[(x,y,bool(implant[i]),bool(bone[i])) for x,y,i in intervals]
    if not states[0][2]:return dict(base,status='axis_origin_not_in_target_implant')
    exited=False;groups=[];active=None
    for left,right,is_implant,is_bone in states:
        if not is_implant:exited=True
        elif exited:return dict(base,status='target_implant_reentry_ambiguous')
        in_bone=exited and is_bone and not is_implant
        if in_bone:
            if active is None:active=[left,right]
            else:active[1]=right
        elif active is not None:groups.append(active);active=None
    if active is not None:groups.append(active)
    base['bone_intervals_mm']=groups
    if not groups:return dict(base,status='no_bone_observed_not_zero')
    if len(groups)>1:return dict(base,status='multiple_bone_intervals_needs_review')
    start,end=groups[0]
    if abs(end-states[-1][1])<1e-8:return dict(base,status='bone_outer_boundary_unobserved_'+limit)
    q=np.asarray(origin,float)+end*unit(direction)
    return dict(base,status='computed_segmentation_geometry_only',distance_mm=end,
                bone_point_ras=q.tolist(),bone_entry_mm=start,
                note='axis_to_outer_edge_distance_no_implant_radius_subtraction')


def sample_slice(volume,a,p,u,b,lateral_mm=(-12,12),axial_mm=(-2,14),spacing=.1,order=0):
    xs=np.arange(lateral_mm[0],lateral_mm[1]+spacing*.1,spacing)
    zs=np.arange(axial_mm[0],axial_mm[1]+spacing*.1,spacing)
    grid=np.asarray(p)[None,None,:]+zs[:,None,None]*u+xs[None,:,None]*b
    ijk=voxel(grid.reshape(-1,3),a)
    data=ndimage.map_coordinates(np.asarray(volume,float),ijk.T,order=order,mode='constant',cval=np.nan).reshape(len(zs),len(xs))
    transform=np.eye(4);transform[:3,0]=b*spacing;transform[:3,1]=u*spacing
    transform[:3,2]=np.cross(b,u);transform[:3,3]=np.asarray(p)+lateral_mm[0]*b+axial_mm[0]*u
    return data,{'pixel_to_ras':transform.tolist(),'spacing_mm':spacing,
                 'axis_origin_in_slice_pixel':[-lateral_mm[0]/spacing,-axial_mm[0]/spacing],
                 'x_mm':xs.tolist(),'y_mm':zs.tolist()}


def run_case(implant,bone,a,config,tooth_centers=None):
    a=affine_check(a)
    for mask in (implant,bone):
        raw=np.asarray(mask)
        if not np.isfinite(raw).all() or not np.isin(raw,[0,1]).all():raise ValueError('finite_binary_masks_required')
    implant=np.asarray(implant,bool);bone=np.asarray(bone,bool)
    if implant.ndim!=3 or bone.shape!=implant.shape:raise ValueError('same_3d_grid_required')
    _,n=ndimage.label(implant)
    if n!=1:raise ValueError('exactly_one_target_implant_component_required')
    try:proposal=pose_proposal(implant,a,config['jaw'])
    except ValueError as e:proposal={'status':'proposal_unavailable','reason':str(e)}
    result={'status':'needs_review','automatic_release':False,'pose_proposal':proposal,
        'source':'3d_existing_segmentation','measurements':[]}
    platform=config.get('platform_ras');axis=config.get('axis_apical_ras')
    if platform is None or axis is None:
        result['missing']=['confirmed_platform_ras_and_apical_axis'];return result,None
    p=np.asarray(platform,float)
    tangent=config.get('arch_tangent_ras')
    if tangent is None:tangent,fit=arch_tangent(tooth_centers,p);result['arch_fit']=fit
    u,b,normal=plane_basis(axis,tangent,config['buccal_hint_ras'])
    result['geometry']={'platform_ras':p.tolist(),'axis_apical_ras':u.tolist(),
        'buccal_ras':b.tolist(),'slice_normal_ras':normal.tolist(),
        'orientation_source':config.get('orientation_source','explicit_input_not_validated')}
    for h in [0.,2.,4.,6.]:
        origin=p+h*u
        row=measure_ray(implant,bone,a,origin,b,config.get('max_search_mm',20.))
        row.update(height_mm=h,axis_point_ras=origin.tolist());result['measurements'].append(row)
    result['status']='computed_geometry_needs_image_and_semantic_QC' if all(r['distance_mm'] is not None for r in result['measurements']) else 'partial_needs_review'
    section,metadata=sample_slice(bone.astype(float)+2*implant.astype(float),a,p,u,b)
    result['slice']=metadata
    return result,section
