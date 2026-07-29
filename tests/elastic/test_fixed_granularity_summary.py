import unittest

from runtime.plan.summarize_fixed_granularity_sweep import (
    summarize_granularity_schemes,
    summarize_rows,
)


class FixedGranularitySummaryTests(unittest.TestCase):
    def test_phase_totals_are_normalized_per_forward(self):
        row = {
            "ratio_pct": "30",
            "mode": "cut",
            "fixed_decode_tokens_total": "3",
            "decode_phase_runs": "4",
            "decode_phase_nonresident_units": "16",
            "steady_median_latency_ms": "100",
            "weight_target_mib": "1000",
            "wbm_total_mib": "4436.789",
            "decode_phase_direct_read_mib": "400",
            "decode_phase_direct_read_ms": "40",
            "decode_phase_load_ms": "44",
            "decode_phase_prepare_ms": "80",
            "decode_phase_compute_ms": "120",
            "decode_phase_pipeline_wait_ms": "20",
            "decode_phase_pipeline_residency_ms": "24",
            "unit_count": "10",
            "cut_op_count": "4",
            "peak_unit_mib": "2",
            "physical_cut_tensors": "10",
            "physical_cut_parts": "20",
            "pin_token_embd_inside_budget_mib": "281.8125",
            "pin_output_inside_budget_mib": "410.9765625",
            "pipeline_budget_violations": "0",
            "pipeline_plan_protection_relaxations": "0",
        }
        summary = summarize_rows([row], backend="cpu")
        self.assertEqual(len(summary), 1)
        self.assertEqual(summary[0]["direct_read_mib_per_forward"], 100.0)
        self.assertEqual(summary[0]["load_stage_ms_per_forward"], 11.0)
        self.assertEqual(
            summary[0]["nonresident_units_per_forward"], 4.0)
        self.assertEqual(summary[0]["layout_prepare_ms_per_forward"], 20.0)
        self.assertEqual(summary[0]["compute_ms_per_forward"], 30.0)
        self.assertEqual(summary[0]["pipeline_wait_ms_per_forward"], 5.0)

    def test_gpu_compute_is_not_mislabeled_as_cpu_fallback_time(self):
        row = {
            "ratio_pct": "90",
            "mode": "multi",
            "fixed_decode_tokens_total": "1",
            "steady_median_latency_ms": "10",
            "physical_cut_tensors": "10",
            "physical_cut_parts": "20",
            "pipeline_budget_violations": "0",
            "pipeline_plan_protection_relaxations": "0",
        }
        summary = summarize_rows([row], backend="gpu")
        self.assertEqual(summary[0]["compute_ms_per_forward"], "")
        self.assertEqual(summary[0]["layout_prepare_ms_per_forward"], "")

    def test_multi_implementation_is_selected_once_for_all_budgets(self):
        rows = []
        for ratio, ordinary, fused in (
            (30, 90.0, 100.0),
            (90, 50.0, 40.0),
        ):
            for mode, latency in (
                ("multi", ordinary),
                ("multi_fused", fused),
                ("tensor", 80.0),
                ("cut", 70.0),
            ):
                rows.append({
                    "ratio_pct": str(ratio),
                    "mode": mode,
                    "steady_median_latency_ms": str(latency),
                    "physical_cut_tensors": "10",
                    "physical_cut_parts": "20",
                    "pipeline_budget_violations": "0",
                    "pipeline_plan_protection_relaxations": "0",
                })
        summary, implementation, selection = (
            summarize_granularity_schemes(rows, backend="cpu"))
        # The paired geometric mean is sqrt((100/90)*(40/50)) < 1, so fused
        # is selected globally even though ordinary Multi wins at 30%.
        self.assertEqual(implementation, "multi_fused")
        self.assertEqual(
            selection["selection_scope"],
            "one implementation across all budgets",
        )
        self.assertEqual(len(summary), 6)
        self.assertEqual(
            {
                row["mode"]
                for row in summary
            },
            {"multi", "tensor", "cut"},
        )
        multi_by_ratio = {
            row["ratio_pct"]: row["steady_median_latency_ms"]
            for row in summary
            if row["mode"] == "multi"
        }
        self.assertEqual(multi_by_ratio, {30: 100.0, 90: 40.0})


if __name__ == "__main__":
    unittest.main()
