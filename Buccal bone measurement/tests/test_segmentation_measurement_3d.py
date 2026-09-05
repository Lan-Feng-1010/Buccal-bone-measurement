import unittest
import numpy as np
from postop_cbct.segmentation_measurement_3d import *


def phantom():
    shape=(180,140,160);a=np.diag([.1,.1,.1,1.])
    x,y,z=np.indices(shape)
    implant=((x-90)**2+(y-70)**2<=100)&(z>=40)&(z<=120)
    bone=(x>=110)&(x<=144)&(y>=50)&(y<=90)&(z>=30)&(z<=130)
    config={'jaw':'lower','platform_ras':[9,7,12], 'axis_apical_ras':[0,0,-1],
        'arch_tangent_ras':[0,1,0],'buccal_hint_ras':[1,0,0],
        'orientation_source':'synthetic_phantom_known_geometry'}
    return implant,bone,a,config


class ThreeDTest(unittest.TestCase):
    def test_hole_then_bone_not_zero(self):
        i,b,a,c=phantom();r,_=run_case(i,b,a,c)
        np.testing.assert_allclose([v['distance_mm'] for v in r['measurements']],[5.45]*4,atol=1e-8)
        self.assertFalse(r['automatic_release'])

    def test_no_bone_null(self):
        i,b,a,c=phantom();b[:]=False
        self.assertEqual(measure_ray(i,b,a,c['platform_ras'],[1,0,0])['status'],'no_bone_observed_not_zero')

    def test_two_bone_intervals_null(self):
        i,b,a,c=phantom();b[125:130]=False
        self.assertEqual(measure_ray(i,b,a,c['platform_ras'],[1,0,0])['status'],'multiple_bone_intervals_needs_review')

    def test_fov_truncation_null(self):
        i,b,a,c=phantom();b[110:,:,30:131]=True
        self.assertEqual(measure_ray(i,b,a,c['platform_ras'],[1,0,0])['status'],'bone_outer_boundary_unobserved_fov_limit')

    def test_search_truncation_null(self):
        i,b,a,c=phantom()
        self.assertEqual(measure_ray(i,b,a,c['platform_ras'],[1,0,0],4)['status'],'bone_outer_boundary_unobserved_search_limit')

    def test_thin_single_voxel_bone_not_skipped(self):
        i,b,a,c=phantom();b[:]=False;b[110,50:91,30:131]=True
        self.assertAlmostEqual(measure_ray(i,b,a,c['platform_ras'],[1,0,0])['distance_mm'],2.05)

    def test_buccal_plane_orthogonal(self):
        u,b,n=plane_basis([.2,.3,1],[1,.2,.5],[0,1,0])
        np.testing.assert_allclose([np.dot(u,b),np.dot(u,n),np.dot(b,n)],[0,0,0],atol=1e-9)

    def test_arch_evaluated_at_implant(self):
        c=np.array([[9+.03*y*y,7+y,12] for y in [-6,-3,0,3,6]])
        t,fit=arch_tangent(c,[9,7,12]);self.assertAlmostEqual(abs(t[1]),1.,places=7)

    def test_missing_platform_returns_proposal_not_truth(self):
        i,b,a,c=phantom();c.pop('platform_ras');r,s=run_case(i,b,a,c)
        self.assertEqual(r['status'],'needs_review');self.assertIsNone(s);self.assertFalse(r['measurements'])

    def test_rigid_affine_invariance(self):
        i,b,a,c=phantom();angle=.53
        rot=np.array([[np.cos(angle),-np.sin(angle),0],[np.sin(angle),np.cos(angle),0],[0,0,1]])
        aa=a.copy();aa[:3,:3]=rot@a[:3,:3];aa[:3,3]=[20,-10,5]
        p=rot@np.array(c['platform_ras'])+aa[:3,3]
        r=measure_ray(i,b,aa,p,rot@np.array([1,0,0]))
        self.assertAlmostEqual(r['distance_mm'],5.45)


if __name__=='__main__':unittest.main(verbosity=2)
