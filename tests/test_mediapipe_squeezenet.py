#!/usr/bin/env python3
"""
Unit and Regression Tests for MediaPipe-Compatible SqueezeNet Model & Preprocessing

Validates:
1. Production model (App/final_model_26) remains completely unaltered and loads cleanly.
2. Production inference pipeline behavior is intact.
3. Experimental model (models/mediapipe_squeezenet_v1) adheres to the [1, 224, 224, 3] -> [1, 26] contract.
4. Class mapping consistency between production and experimental models.
5. Probability normalization and softmax output properties.
"""

import os
import sys
import json
import unittest
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

APP_DIR = os.path.join(BASE_DIR, "App")
if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)

import tf_keras
from pipeline import ISLPipeline


class TestMediaPipeSqueezeNetRegression(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.prod_model_path = os.path.join(BASE_DIR, "App", "final_model_26")
        cls.prod_class_map_path = os.path.join(BASE_DIR, "App", "model_class_26.json")
        cls.exp_model_dir = os.path.join(BASE_DIR, "models", "mediapipe_squeezenet_v1")
        cls.exp_saved_model_path = os.path.join(cls.exp_model_dir, "saved_model")
        cls.exp_class_map_path = os.path.join(cls.exp_model_dir, "model_class_26.json")

    def test_production_model_intact(self):
        """Verify production model App/final_model_26 exists and loads with correct 26-class head."""
        self.assertTrue(os.path.exists(self.prod_model_path), "Production model missing!")
        prod_model = tf_keras.models.load_model(self.prod_model_path)
        self.assertEqual(prod_model.input_shape, (None, 224, 224, 3))
        self.assertEqual(prod_model.output_shape, (None, 26))

        # Check dummy inference
        dummy = np.zeros((1, 224, 224, 3), dtype=np.float32)
        preds = prod_model.predict(dummy, verbose=0)
        self.assertEqual(preds.shape, (1, 26))
        self.assertAlmostEqual(float(np.sum(preds)), 1.0, places=4)

    def test_class_mapping_alignment(self):
        """Verify class mappings have identical 26 alphabet labels in exact alphabetical order."""
        with open(self.prod_class_map_path, "r") as f:
            prod_map = json.load(f)

        self.assertEqual(len(prod_map), 26)
        expected_alphabet = [chr(ord('A') + i) for i in range(26)]
        for i, letter in enumerate(expected_alphabet):
            self.assertEqual(prod_map[str(i)], letter)

        if os.path.exists(self.exp_class_map_path):
            with open(self.exp_class_map_path, "r") as f:
                exp_map = json.load(f)
            self.assertEqual(prod_map, exp_map, "Experimental class map deviates from production!")

    def test_experimental_model_contract(self):
        """Verify experimental model (if saved) satisfies [1, 224, 224, 3] contract."""
        if not os.path.exists(self.exp_saved_model_path):
            self.skipTest("Experimental model not yet saved.")

        exp_model = tf_keras.models.load_model(self.exp_saved_model_path)
        self.assertEqual(exp_model.input_shape, (None, 224, 224, 3))
        self.assertEqual(exp_model.output_shape, (None, 26))

        dummy = np.random.uniform(0.0, 1.0, size=(1, 224, 224, 3)).astype(np.float32)
        preds = exp_model.predict(dummy, verbose=0)
        self.assertEqual(preds.shape, (1, 26))
        self.assertAlmostEqual(float(np.sum(preds)), 1.0, places=4)

    def test_pipeline_runtime_defaults(self):
        """Verify ISLPipeline defaults remain preserved and uncompromised."""
        pipeline = ISLPipeline()
        # Lap hand suppression should be enabled by default (Phase 8)
        self.assertTrue(pipeline.suppress_lap_hand)
        # Direct resize and square crop remain disabled by default
        self.assertFalse(pipeline.direct_resize)
        self.assertFalse(pipeline.square_crop)


if __name__ == "__main__":
    unittest.main()
