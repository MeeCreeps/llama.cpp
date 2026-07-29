import json
import tempfile
import unittest
from pathlib import Path

from runtime.plan.select_mixed_boundary_profile import (
    measured_frontier_transition_cost,
    mode_counts,
    require_calibration_lineage,
    speedup_pct,
    token_budget_latency_breakdown,
    valid_device_row,
)
from runtime.plan.super_tensor_planner import GranularityCostModel


class MixedBoundaryProfileSelectionTests(unittest.TestCase):
    def test_cut_tiles_count_as_one_flexible_logical_weight(self):
        model_meta = {
            "weights": [
                {
                    "weight_id": weight_id,
                    "name": f"blk.0.w{weight_id}.weight",
                    "layer": 0,
                    "byte_size": 1024,
                    "shape": [4096, 512],
                    "quant_name": "Q4_0",
                }
                for weight_id in range(3)
            ],
            "ops": [
                {"op_id": weight_id, "weight_id": weight_id}
                for weight_id in range(3)
            ],
        }
        plan = {
            "weights": [
                {
                    "weight_id": weight_id,
                    "name": f"blk.0.w{weight_id}.weight",
                    "byte_size": 1024,
                    "location": "disk",
                }
                for weight_id in range(3)
            ],
            "working_unit": {
                "units": [
                    {
                        "unit_id": 10,
                        "tiles": [{
                            "weight_id": 0,
                            "row_start": 0,
                            "row_count": 256,
                            "byte_size": 512,
                        }],
                    },
                    {
                        "unit_id": 11,
                        "tiles": [{
                            "weight_id": 0,
                            "row_start": 256,
                            "row_count": 256,
                            "byte_size": 512,
                        }],
                    },
                    {
                        "unit_id": 12,
                        "tiles": [
                            {
                                "weight_id": weight_id,
                                "row_start": 0,
                                "row_count": 512,
                                "byte_size": 1024,
                            }
                            for weight_id in (1, 2)
                        ],
                    },
                ],
            },
        }
        with tempfile.TemporaryDirectory() as temporary:
            plan_path = Path(temporary) / "plan.json"
            plan_path.write_text(
                json.dumps(plan), encoding="utf-8")
            counts = mode_counts(
                plan_path,
                model_meta,
                GranularityCostModel({}, "cpu"),
            )
            alternate = json.loads(json.dumps(plan))
            alternate["working_unit"]["units"] = [{
                    "unit_id": 20,
                    "tiles": [
                        {
                            "weight_id": weight_id,
                            "row_start": 0,
                            "row_count": 512,
                            "byte_size": 1024,
                        }
                        for weight_id in (0, 1)
                    ],
                }] + [
                {
                    "unit_id": 21 + part,
                    "tiles": [{
                        "weight_id": 2,
                        "row_start": part * 256,
                        "row_count": 256,
                        "byte_size": 512,
                    }],
                }
                for part in range(2)
            ]
            alternate_path = Path(temporary) / "alternate.json"
            alternate_path.write_text(
                json.dumps(alternate), encoding="utf-8")
            alternate_counts = mode_counts(
                alternate_path,
                model_meta,
                GranularityCostModel({}, "cpu"),
            )
        self.assertEqual(counts["cut"], 1)
        self.assertEqual(counts["multi"], 1)
        self.assertEqual(counts["cut_flexible_weight_coverage"], 1)
        self.assertEqual(counts["multi_flexible_weight_coverage"], 2)
        self.assertEqual(alternate_counts["cut"], counts["cut"])
        self.assertEqual(alternate_counts["multi"], counts["multi"])
        self.assertNotEqual(
            alternate_counts["frontier_sha256"],
            counts["frontier_sha256"],
        )

    def test_node_speedup_uses_measured_tensor_reference(self):
        self.assertAlmostEqual(
            speedup_pct({"mean_ms": 100.0}, {"mean_ms": 80.0}),
            20.0,
        )
        with self.assertRaisesRegex(ValueError, "must be positive"):
            speedup_pct({"mean_ms": 0.0}, {"mean_ms": 1.0})

    def base_row(self):
        return {
            "status": "ok",
            "rc": "0",
            "valid_token_trace": "True",
            "token_trace_timing_contract_valid": "True",
            "decode_wall_contract_valid": "True",
            "decode_phase_contract_valid": "True",
            "decode_phase_counter_scope":
                "decode-only-backend-counter-delta",
            "valid_measurement_isolation": "True",
            "measurement_monitor_samples": "4",
            "measurement_monitor_failures": "0",
            "measurement_llama_pid": "123",
            "monitor_thread_incomplete": "False",
            "pipeline_budget_violations": "0",
            "timed_run_contract_valid": "True",
            "pin_contract_valid": "True",
            "fused_lane_contract_valid": "True",
            "unit_pipeline_stage_authority_contract_valid": "True",
            "backend_compute_contract_valid": "True",
            "frontier_contract_valid": "True",
            "pin_outside_budget_count": "0",
            "pipeline_plan_protection_relaxations": "0",
            "physical_cut_tensors": "7",
            "physical_cut_parts": "14",
            "wbm_total_mib": "4436.789",
            "expected_wbm_total_mib_configured": "4436.7890625",
            "bench_exit_reason": "duration",
            "bench_time_done": "1",
            "frontier_apply_count": "4",
            "frontier_publish_ms_total": "2.0",
            "frontier_publish_ms_max": "0.8",
            "frontier_transition_publish_count": "3",
            "frontier_transition_publish_ms_total": "1.2",
            "frontier_transition_publish_ms_max": "0.5",
            "remote_binary_sha256": "binary",
            "remote_model_sha256": "model",
            "model_meta_sha256": "meta",
            "remote_model_meta_sha256": "meta",
            "cost_dir_sha256": "cost",
            "remote_cost_dir_sha256": "cost",
            "dynamic_run_config_sha256": "config",
            "dynamic_run_core_config_sha256": "core",
            "source_file_sha256": "source",
            "expected_source_trace_sha256": "source",
            "source_window_sha256": "window",
            "runtime_trace_sha256": "runtime",
            "remote_runtime_trace_sha256": "runtime",
            "token_prefix_sha256": "token",
        }

    def test_accepts_single_method_row_without_peer_sequence_flag(self):
        self.assertTrue(valid_device_row(self.base_row()))

    def test_transition_cost_uses_isolated_frontier_publish_max(self):
        calibration = measured_frontier_transition_cost(self.base_row())
        self.assertEqual(
            calibration["selected_transition_fixed_ms"], 0.5)
        self.assertEqual(
            calibration["frontier_publish_ms_mean"], 0.5)
        self.assertAlmostEqual(
            calibration["frontier_transition_publish_ms_mean"], 0.4)
        row = self.base_row()
        row["frontier_transition_publish_ms_max"] = "0"
        with self.assertRaisesRegex(
                ValueError, "no measured frontier transition publication"):
            measured_frontier_transition_cost(row)

    def test_rejects_explicit_cross_method_token_mismatch(self):
        row = self.base_row()
        row["valid_token_sequence"] = "False"
        self.assertFalse(valid_device_row(row))

    def test_rejects_missing_token_trace(self):
        row = self.base_row()
        row["valid_token_trace"] = "False"
        self.assertFalse(valid_device_row(row))

    def test_rejects_plan_protection_relaxation(self):
        row = self.base_row()
        row["pipeline_plan_protection_relaxations"] = "1"
        self.assertFalse(valid_device_row(row))

    def test_rejects_missing_plan_protection_audit(self):
        row = self.base_row()
        del row["pipeline_plan_protection_relaxations"]
        self.assertFalse(valid_device_row(row))

    def test_rejects_invalid_pin_contract(self):
        row = self.base_row()
        row["pin_contract_valid"] = "False"
        self.assertFalse(valid_device_row(row))

    def test_rejects_invalid_unit_pipeline_stage_authority(self):
        row = self.base_row()
        row["unit_pipeline_stage_authority_contract_valid"] = "False"
        self.assertFalse(valid_device_row(row))

    def test_requires_one_boundary_calibration_lineage(self):
        rows = [self.base_row(), self.base_row()]
        audit = require_calibration_lineage(rows)
        self.assertEqual(audit["rows"], 2)
        self.assertEqual(audit["physical_cut_parts"], 14)
        rows[1]["remote_binary_sha256"] = "other"
        with self.assertRaisesRegex(
                ValueError, "one remote_binary_sha256"):
            require_calibration_lineage(rows)

    def test_node_breakdown_skips_prompt_and_groups_decode_latency(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "tokens.csv"
            path.write_text(
                "sequence,budget_mib,latency_ms\n"
                "0,3712,9000\n"
                "1,3712,100\n"
                "2,3712,200\n"
                "3,4224,80\n"
                "4,5081,90\n",
                encoding="utf-8",
            )
            result = token_budget_latency_breakdown(path)
            self.assertEqual(result["3712"]["samples"], 2)
            self.assertEqual(result["3712"]["mean_ms"], 150.0)
            self.assertEqual(result["3712"]["median_ms"], 150.0)
            self.assertEqual(result["3712"]["p95_ms"], 200.0)
            self.assertEqual(result["4224"]["mean_ms"], 80.0)
            filtered = token_budget_latency_breakdown(
                path, {3712, 4224})
            self.assertNotIn("5081", filtered)


if __name__ == "__main__":
    unittest.main()
