from __future__ import annotations

import unittest

import cv2
import numpy as np

from photoperfect_engine import PhotoPerfectEngine


class PhotoPerfectEngineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.engine = PhotoPerfectEngine()

    def test_monochrome_photo_uses_restore_plan(self) -> None:
        gradient = np.tile(np.linspace(45, 205, 720, dtype=np.uint8), (1100, 1))
        image = cv2.cvtColor(gradient, cv2.COLOR_GRAY2BGR)
        inspection = self.engine.inspect(image)
        plan = self.engine.plan(inspection, 'Auto Detect')
        self.assertTrue(inspection.is_monochrome)
        self.assertEqual(plan.name, 'Black & White Restore')
        self.assertIn('restore monochrome contrast', plan.stages)

    def test_good_colour_photo_uses_light_polish(self) -> None:
        image = np.zeros((1200, 1600, 3), dtype=np.uint8)
        image[:] = (110, 145, 175)
        cv2.rectangle(image, (100, 100), (1500, 1100), (60, 100, 150), 8)
        cv2.line(image, (100, 1100), (1500, 100), (240, 240, 240), 5)
        inspection = self.engine.inspect(image)
        plan = self.engine.plan(inspection, 'Auto Detect')
        self.assertFalse(inspection.is_monochrome)
        self.assertIn(plan.name, {'Professional Light Polish', 'Automatic Photo Recovery'})

    def test_quality_validator_rejects_clear_degradation(self) -> None:
        rng = np.random.default_rng(4)
        image = rng.integers(20, 235, size=(640, 960, 3), dtype=np.uint8)
        inspection = self.engine.inspect(image)
        plan = self.engine.plan(inspection, 'Auto Enhance')
        result, validation = self.engine.execute(image, plan)
        self.assertEqual(result.ndim, 3)
        self.assertGreaterEqual(validation.after_score + 1.5, validation.before_score)


if __name__ == '__main__':
    unittest.main()
