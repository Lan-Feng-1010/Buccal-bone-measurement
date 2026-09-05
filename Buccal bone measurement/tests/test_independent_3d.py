"""Independent 3-D segmentation geometry checks; synthetic only."""
import unittest
import numpy as np
from postop_cbct.segmentation_measurement_3d import measure_ray, plane_basis, run_case, sample_slice, world, arch_tangent

def masks():
    implant=np.zeros((40,40,50),bool);implant[18:23,18:23,10:31]=True
    bone=np.zeros_like(implant);bone[24:30,15:26,0:46]=True
    return implant,bone

class Independent3DReview(unittest.TestCase):
    def test_axis_inside_implant_hole_measures_outer_not_inner_boundary(self):
        i,b=masks();r=measure_ray(i,b,np.eye(4),[20,20,20],[1,0,0])
        self.assertEqual(r['distance_mm'],9.5);self.assertEqual(r['bone_entry_mm'],3.5)
    def test_no_bone_is_null_not_zero(self):
        i,b=masks();b[:]=False;r=measure_ray(i,b,np.eye(4),[20,20,20],[1,0,0])
        self.assertIsNone(r['distance_mm']);self.assertEqual(r['status'],'no_bone_observed_not_zero')
    def test_disjoint_bone_is_ambiguous(self):
        i,b=masks();b[26]=False;r=measure_ray(i,b,np.eye(4),[20,20,20],[1,0,0])
        self.assertIsNone(r['distance_mm']);self.assertEqual(r['status'],'multiple_bone_intervals_needs_review')
    def test_search_limit_is_not_outer_boundary(self):
        i,b=masks();r=measure_ray(i,b,np.eye(4),[20,20,20],[1,0,0],8)
        self.assertIsNone(r['distance_mm']);self.assertEqual(r['status'],'bone_outer_boundary_unobserved_search_limit')
    def test_field_limit_is_not_outer_boundary(self):
        i,b=masks();b[24:,:,:]=True;r=measure_ray(i,b,np.eye(4),[20,20,20],[1,0,0],50)
        self.assertIsNone(r['distance_mm']);self.assertEqual(r['status'],'bone_outer_boundary_unobserved_fov_limit')
    def test_one_voxel_thin_bone_is_not_skipped(self):
        i,b=masks();b[25:]=False;r=measure_ray(i,b,np.eye(4),[20,20,20],[1,0,0])
        self.assertEqual(r['distance_mm'],4.5)
    def test_oblique_affine_preserves_physical_distance(self):
        i,b=masks();theta=.63;rot=np.array([[np.cos(theta),0,np.sin(theta)],[0,1,0],[-np.sin(theta),0,np.cos(theta)]])
        a=np.eye(4);a[:3,:3]=rot;a[:3,3]=[12,-30,8]
        r=measure_ray(i,b,a,world([20,20,20],a),rot@[1,0,0])
        self.assertAlmostEqual(r['distance_mm'],9.5)
    def test_anisotropic_affine_uses_mm(self):
        i,b=masks();a=np.diag([2,.7,.4,1]);r=measure_ray(i,b,a,world([20,20,20],a),[1,0,0],30)
        self.assertAlmostEqual(r['distance_mm'],19)
    def test_oblique_ray_exact_voxel_exit(self):
        i,b=masks();v=np.array([1.,.2,.1]);v/=np.linalg.norm(v)
        r=measure_ray(i,b,np.eye(4),[20,20,20],v)
        self.assertAlmostEqual(r['distance_mm'],9.5/v[0])
    def test_implant_reentry_is_ambiguous(self):
        i,b=masks();i[27,20,20]=True;r=measure_ray(i,b,np.eye(4),[20,20,20],[1,0,0])
        self.assertEqual(r['status'],'target_implant_reentry_ambiguous');self.assertIsNone(r['distance_mm'])
    def test_section_basis_contains_axis_and_is_normal_to_projected_arch(self):
        u,b,n=plane_basis([.2,.1,1],[1,.2,.1],[0,1,0]);mat=np.stack([u,b,n])
        np.testing.assert_allclose(mat@mat.T,np.eye(3),atol=1e-12)
        self.assertGreater(np.dot(b,[0,1,0]),0)
    def test_slice_pixel_transform_matches_sampling_coordinates(self):
        i,b=masks();p=np.array([20.,20.,20.]);u=np.array([0.,0.,1.]);bu=np.array([1.,0.,0.])
        section,meta=sample_slice(b,np.eye(4),p,u,bu,lateral_mm=(-2,2),axial_mm=(-2,2),spacing=.5)
        m=np.array(meta['pixel_to_ras']);np.testing.assert_allclose((m@np.array([4,4,0,1]))[:3],p)
    def test_run_case_reports_four_heights_but_no_clinical_release(self):
        i,b=masks();c=dict(jaw='upper',platform_ras=[20,20,10],axis_apical_ras=[0,0,1],arch_tangent_ras=[0,1,0],buccal_hint_ras=[1,0,0])
        r,section=run_case(i,b,np.eye(4),c)
        self.assertEqual([q['distance_mm'] for q in r['measurements']],[9.5]*4)
        self.assertFalse(r['automatic_release']);self.assertEqual([q['axis_point_ras'][2] for q in r['measurements']],[10,12,14,16])

    def test_run_case_rejects_nonfinite_and_nonbinary_masks(self):
        for bad in [float('nan'),2.]:
            i,b=masks();i=i.astype(float);i[0,0,0]=bad
            with self.assertRaisesRegex(ValueError,'finite_binary_masks_required'):run_case(i,b,np.eye(4),dict(jaw='upper'))
    def test_confirmed_pose_not_blocked_by_horizontal_svd_proposal(self):
        i,b=masks();i=np.swapaxes(i,0,2);b=np.swapaxes(b,0,2)
        c=dict(jaw='upper',platform_ras=[10,20,20],axis_apical_ras=[1,0,0],arch_tangent_ras=[0,1,0],buccal_hint_ras=[0,0,1])
        r,_=run_case(i,b,np.eye(4),c)
        self.assertEqual(r['pose_proposal']['status'],'proposal_unavailable')
        self.assertEqual([q['distance_mm'] for q in r['measurements']],[9.5]*4)
    def test_half_open_coronal_boundary_refers_for_review_without_shift(self):
        i,b=masks();c=dict(jaw='lower',platform_ras=[20,20,30.5],axis_apical_ras=[0,0,-1],arch_tangent_ras=[0,1,0],buccal_hint_ras=[1,0,0])
        r,_=run_case(i,b,np.eye(4),c);zero=r['measurements'][0]
        self.assertIsNone(zero['distance_mm']);self.assertEqual(zero['status'],'axis_origin_not_in_target_implant')
        self.assertEqual(zero['axis_point_ras'],[20,20,30.5]);self.assertFalse(r['automatic_release'])

if __name__=='__main__':unittest.main(verbosity=2)
