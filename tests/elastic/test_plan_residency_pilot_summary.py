#!/usr/bin/env python3

import unittest

from runtime.plan.summarize_plan_residency_pilot import (
    common_candidate_audit,
    row_contract_failures,
)


class PlanResidencyPilotSummaryTests(unittest.TestCase):
    def valid_row(self):
        return {
            "status": "ok",
            "rc": "0",
            "valid_token_trace": "True",
            "token_trace_timing_contract_valid": "True",
            "decode_wall_contract_valid": "True",
            "decode_phase_contract_valid": "True",
            "valid_measurement_isolation": "True",
            "pin_contract_valid": "True",
            "fused_lane_contract_valid": "True",
            "unit_pipeline_stage_authority_contract_valid": "True",
            "frontier_contract_valid": "True",
            "backend_compute_contract_valid": "True",
            "remote_solver_contract_valid": "True",
            "timed_run_contract_valid": "True",
            "measurement_monitor_samples": "10",
            "measurement_monitor_failures": "0",
            "measurement_llama_pid": "123",
            "monitor_thread_incomplete": "False",
            "pipeline_budget_violations": "0",
            "pipeline_plan_protection_relaxations": "0",
            "pin_outside_budget_count": "0",
            "physical_cut_tensors": "7",
            "physical_cut_parts": "14",
            "wbm_total_mib": "4436.789",
            "expected_wbm_total_mib_configured": "4436.7890625",
            "source_start_sec": "0",
            "source_span_sec": "120",
            "replay_span_sec": "120",
            "runtime_trace_sha256": "runtime-trace",
            "remote_runtime_trace_sha256": "runtime-trace",
            "expected_model_sha256": "model",
            "remote_model_sha256": "model",
            "expected_binary_sha256": "binary",
            "remote_binary_sha256": "binary",
            "model_meta_sha256": "meta",
            "remote_model_meta_sha256": "meta",
            "cost_dir_sha256": "cost",
            "remote_cost_dir_sha256": "cost",
            "granularity_profile_sha256": "profile",
            "multi_implementation_configured": "multi_fused",
            "mixed_offline_table_sha256": "mixed-table",
            "source_file_sha256": "source",
            "expected_source_trace_sha256": "source",
            "source_window_sha256": "source",
            "planner_stream_reserve_mib_configured": "16",
            "dynamic_run_config_sha256": "full-config",
            "dynamic_run_core_config_sha256": "core-config",
        }

    def candidate_rows(self):
        rows = []
        for index in range(3):
            row = self.valid_row()
            row["dynamic_run_config_sha256"] = f"full-config-{index}"
            rows.append(row)
        return rows

    def test_valid_row_contract_requires_common_physical_tiling(self):
        row = self.valid_row()
        self.assertEqual(row_contract_failures(row, {}), [])
        row["physical_cut_parts"] = "7"
        self.assertIn(
            "physical-tiling-mismatch",
            row_contract_failures(row, {}),
        )

    def test_valid_row_contract_requires_unit_pipeline_stage_authority(self):
        row = self.valid_row()
        row["unit_pipeline_stage_authority_contract_valid"] = "False"
        self.assertIn(
            "unit-pipeline-stage-authority-contract",
            row_contract_failures(row, {}),
        )

    def test_common_candidate_audit_accepts_policy_only_difference(self):
        audit = common_candidate_audit(self.candidate_rows())
        self.assertEqual(
            audit["dynamic_run_core_config_sha256"], "core-config")
        self.assertEqual(audit["physical_cut_tensors"], 7)
        self.assertEqual(audit["physical_cut_parts"], 14)
        self.assertEqual(audit["planner_stream_reserve_mib"], 16.0)

    def test_common_candidate_audit_rejects_runtime_difference(self):
        rows = self.candidate_rows()
        rows[2]["dynamic_run_core_config_sha256"] = "different-core"
        with self.assertRaisesRegex(ValueError, "core_config"):
            common_candidate_audit(rows)

    def test_common_candidate_audit_rejects_physical_difference(self):
        rows = self.candidate_rows()
        rows[1]["wbm_total_mib"] = "4000"
        with self.assertRaisesRegex(ValueError, "physical tiling"):
            common_candidate_audit(rows)


if __name__ == "__main__":
    unittest.main()
