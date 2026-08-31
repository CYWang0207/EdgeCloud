"""Small deterministic tests for the four submission-critical primitives."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

try:
    import torch
    from torch import nn
except ModuleNotFoundError:  # Allows scheduler/arbitration checks on a minimal Python install.
    torch = None
    nn = None

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [
    str(ROOT / "module_edge_perception"),
    str(ROOT / "common"),
    str(ROOT / "module_scheduling/EdgeCloud_RL"),
    str(ROOT / "module_scheduling/multi_node"),
]

from arbiter import Arbiter, Report, WeightedVoteFusion  # noqa: E402
from network_sim import NetworkSimulator, mb_per_slot_to_mbps, mbps_to_mb_per_slot  # noqa: E402

if torch is not None:
    from adaptformer import AdaptFormerMLPWrapper  # noqa: E402
    from drift_dataset import drift_spec_for_index, structural_drift_score  # noqa: E402


class SubmissionLogicTests(unittest.TestCase):
    @unittest.skipUnless(torch is not None, "requires PyTorch and torchvision; exercised in CI")
    def test_adapter_zero_initialization_preserves_wrapped_mlp(self) -> None:
        torch.manual_seed(42)
        base = nn.Linear(8, 8)
        wrapped = AdaptFormerMLPWrapper(base, dim=8, r=2)
        values = torch.randn(3, 8)
        self.assertTrue(torch.allclose(base(values), wrapped(values)))
        wrapped.enabled = False
        self.assertTrue(torch.allclose(base(values), wrapped(values)))

    @unittest.skipUnless(torch is not None, "requires PyTorch and torchvision; exercised in CI")
    def test_drift_schedule_is_bounded_and_deterministic(self) -> None:
        drift, severity = drift_spec_for_index(7, 20, "mixed")
        self.assertIn(drift, {"normal", "bright", "dark", "blur", "noise", "occlusion"})
        self.assertGreaterEqual(severity, 0.0)
        self.assertLessEqual(structural_drift_score("noise", 2.0), 1.0)

    def test_network_unit_conversion_and_deadline(self) -> None:
        mb = mbps_to_mb_per_slot(80.0, 0.2)
        self.assertAlmostEqual(mb_per_slot_to_mbps(mb, 0.2), 80.0)
        simulator = NetworkSimulator(mode="static", seed=42)
        state = simulator.step()
        self.assertIn("bandwidth_mbps", state)
        self.assertEqual(simulator.compute_e2e(0, 0.0)["e2e_delay_ms"], 80.0)

    def test_weighted_arbitration_resolves_high_conflict(self) -> None:
        arbiter = Arbiter(WeightedVoteFusion())
        arbiter.receive([Report(0, 1, 2, 0.91), Report(1, 1, 3, 0.72), Report(2, 1, 2, 0.88)])
        arbiter.detect_conflicts()
        arbiter.arbitrate()
        arbiter.dispatch_and_rollback()
        stats = arbiter.stats()
        self.assertEqual(stats["num_conflicts"], 1)
        self.assertEqual(stats["resolve_rate"], 1.0)


if __name__ == "__main__":
    unittest.main()
