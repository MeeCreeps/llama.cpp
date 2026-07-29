import unittest

from runtime.plan.summarize_granularity_frontier_trace import (
    bucket_for_budget,
    dominant_mode,
)


class FrontierTraceSummaryTests(unittest.TestCase):
    def test_budget_is_floored_and_clamped_to_available_plan(self):
        available = [3712, 3840, 3968, 4096]
        self.assertEqual(
            bucket_for_budget(4080.0, 128, available), 3968)
        self.assertEqual(
            bucket_for_budget(2000.0, 128, available), 3712)
        self.assertEqual(
            bucket_for_budget(9000.0, 128, available), 4096)

    def test_dominance_uses_weight_coverage_not_decision_count(self):
        self.assertEqual(
            dominant_mode({
                "multi": 60,
                "tensor": 90,
                "cut": 20,
            }),
            "multi",
        )
        self.assertEqual(
            dominant_mode({
                "multi": 20,
                "tensor": 30,
                "cut": 100,
            }),
            "cut",
        )


if __name__ == "__main__":
    unittest.main()
