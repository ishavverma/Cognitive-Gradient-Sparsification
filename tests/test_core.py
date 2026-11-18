import os
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cgs.controller.adaptive import AdaptiveController
from cgs.model.cgs_net import CGSNet
from cgs.nn.module import Module
from cgs.sparse.parameter_block import ModularParameterBlock
from cgs.sparse.router import SparseUpdateRouter
from cgs.sparse.sparsifier import GradientSparsifier
from cgs.tensor.tensor import CGSTensor
from cgs.utils.config import load_config


class SparsifierTests(unittest.TestCase):
    def test_hybrid_requires_both_filters(self):
        sparsifier = GradientSparsifier(mode='hybrid', threshold=0.5)
        scores = {'a': 0.8, 'b': 0.4, 'c': 0.9, 'd': 0.3}
        gradients = {name: np.ones(2) for name in scores}
        sparse, stats = sparsifier.sparsify(scores, gradients)
        kept = sum(np.count_nonzero(value) for value in sparse.values())
        self.assertGreater(stats['sparsity'], 0)
        self.assertEqual(kept, 4)

    def test_elementwise_filter_creates_zero_elements(self):
        sparsifier = GradientSparsifier(mode='topk', threshold=0.3, top_k_ratio=0.25)
        grad = np.array([1.0, 0.1, 0.9, 0.0])
        sparse, stats = sparsifier.sparsify(
            {'weight': 1.0}, {'weight': grad}, elementwise=True
        )
        output = sparse['weight']
        self.assertEqual(np.count_nonzero(output), 1)
        self.assertEqual(stats['total_params'], 4)
        self.assertTrue(stats['elementwise'])


class RouterTests(unittest.TestCase):
    def test_router_enforces_minimum_active_ratio(self):
        param = CGSTensor(np.ones(2), requires_grad=True)
        blocks = [
            ModularParameterBlock('a', [param], ['a']),
            ModularParameterBlock('b', [param], ['b']),
            ModularParameterBlock('c', [param], ['c']),
        ]
        router = SparseUpdateRouter(threshold=0.9, min_active_ratio=0.5)
        decisions = router.route({'a': 0.1, 'b': 0.1, 'c': 0.1}, blocks)
        self.assertGreaterEqual(sum(decisions.values()), 1)


class ControllerTests(unittest.TestCase):
    def test_threshold_and_lr_remain_bounded(self):
        controller = AdaptiveController(
            initial_threshold=0.3, initial_lr=0.001,
            warmup_epochs=0, patience=1,
        )
        for step in range(30):
            result = controller.step(
                loss=1.0, sparsity=0.5, avg_gid=0.2,
                gradient_stats={'avg_magnitude': 0.1, 'avg_novelty': 0.2},
            )
        self.assertGreaterEqual(result['threshold'], 0.05)
        self.assertLessEqual(result['threshold'], 0.8)
        self.assertGreaterEqual(result['lr'], 1e-5)


class ConfigTests(unittest.TestCase):
    def test_nested_yaml_benchmark_is_loaded(self):
        config = load_config('config/default.yaml')
        self.assertIn('benchmark', config)
        self.assertIn('cgs_2pct', config['benchmark'])
        self.assertEqual(config['benchmark']['cgs_2pct']['batch_size'], 32)


class SerializationTests(unittest.TestCase):
    def test_model_save_load_round_trip(self):
        from cgs.export.serializer import save_weights, load_weights

        model = CGSNet(input_dim=20, num_classes=5, variant='S')
        model.eval()
        x = CGSTensor(np.random.randn(2, 20))
        before = model(x).data.copy()

        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, 'model')
            save_weights(model, path)
            restored = CGSNet(input_dim=20, num_classes=5, variant='S')
            restored.eval()
            load_weights(restored, path)
            after = restored(x).data.copy()

        np.testing.assert_allclose(before, after, rtol=1e-7, atol=1e-7)


if __name__ == '__main__':
    unittest.main()
