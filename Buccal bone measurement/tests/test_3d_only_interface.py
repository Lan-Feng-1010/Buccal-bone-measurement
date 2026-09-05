"""Regression checks for the 3-D-only delivery and portable volume inputs."""
import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import nibabel as nib
import numpy as np
from postop_cbct.cli import main
from postop_cbct.demo import create_demo
from postop_cbct.io import load_volume, verify_geometry, save_json
from postop_cbct.pipeline import measure_case, prepare_case


class ThreeDOnlyInterfaceTest(unittest.TestCase):
    def test_only_volume_workflow_commands(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output), self.assertRaises(SystemExit) as cm:
            main(['--help'])
        self.assertEqual(cm.exception.code, 0)
        self.assertIn('{demo,inspect,prepare,measure}', output.getvalue())
        for command in ('annotate2d', 'measure2d'):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as cm:
                main([command])
            self.assertEqual(cm.exception.code, 2)

    def test_legacy_annotation_modules_absent(self):
        self.assertIsNone(importlib.util.find_spec('postop_cbct.annotation_adapter'))
        self.assertIsNone(importlib.util.find_spec('postop_cbct.measurement_core'))

    def test_moved_case_and_different_working_directory(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root/'original case'
            create_demo(source)
            moved = root/'moved case'
            source.rename(moved)
            other = root/'other directory'
            other.mkdir()
            previous = Path.cwd()
            try:
                os.chdir(other)
                result = measure_case(moved/'request.json', 'result')
            finally:
                os.chdir(previous)
            self.assertEqual([r['distance_mm'] for r in result['measurements']], [4.75]*4)
            self.assertFalse(result['automatic_release'])
            self.assertNotIn(str(root), json.dumps(result))

    def test_non_3d_nifti_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/'flat.nii.gz'
            image = nib.Nifti1Image(np.zeros((10,10), np.uint8), np.eye(4))
            nib.save(image, path)
            with self.assertRaisesRegex(ValueError, 'three-dimensional'):
                load_volume(path)

    def test_slice_display_values_do_not_control_distances(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            request = create_demo(root/'input')
            original = measure_case(request, root/'original')
            # Replace only the display sampler. Real 3-D ray traversal still executes.
            with patch('postop_cbct.segmentation_measurement_3d.sample_slice',
                       return_value=(np.zeros((2,2)), {'display_test': True})):
                changed = measure_case(request, root/'changed')
            self.assertEqual(original['measurements'], changed['measurements'])
            self.assertEqual(np.load(root/'changed/segmentation_slice.npy').shape, (2,2))

    def test_prepare_preserves_anonymous_id_and_target(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            request = create_demo(root)
            c = json.loads(request.read_text(encoding='utf-8'))
            implant = nib.load(root/'target_implant.nii.gz')
            bone = nib.load(root/'jaw.nii.gz')
            labels = np.asarray(implant.dataobj)*10 + np.asarray(bone.dataobj)*2
            image = nib.Nifti1Image(labels.astype(np.uint8), implant.affine)
            image.header.set_xyzt_units('mm')
            nib.save(image, root/'labels.nii.gz')
            c.update(toothfairy3_nifti='labels.nii.gz', case_id='CASE001', target_component_id=1)
            save_json(root/'full.json', c)
            prepared = prepare_case(root/'full.json', root/'prepared')
            result = measure_case(prepared, root/'measured')
            self.assertEqual(result['case_id'], 'CASE001')
            self.assertEqual([r['distance_mm'] for r in result['measurements']], [4.75]*4)

    def test_units_require_matching_mm_reference(self):
        def volume(unit, affine=None):
            image = nib.Nifti1Image(np.zeros((4,4,4), np.uint8), np.eye(4) if affine is None else affine)
            image.header.set_xyzt_units(unit)
            return image
        unknown, mm = volume('unknown'), volume('mm')
        with self.assertRaises(ValueError):
            verify_geometry([unknown])
        self.assertEqual(verify_geometry([unknown], mm), 'inherited_from_matched_raw_CBCT_mm_header')
        with self.assertRaises(ValueError):
            verify_geometry([volume('meter')], mm)
        shifted = np.eye(4)
        shifted[0,3] = 1
        with self.assertRaises(ValueError):
            verify_geometry([unknown], volume('mm', shifted))


if __name__ == '__main__':
    unittest.main()
