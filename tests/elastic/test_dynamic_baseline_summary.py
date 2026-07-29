import csv
import tempfile
import unittest
from pathlib import Path

from runtime.plan.summarize_dynamic_baseline_matrix import (
    budget_trace_points,
    enrich_from_raw_log,
    only_result,
    validate,
    validate_cut_compute_lineage,
    validate_profile_lineage,
    validate_runtime_input_lineage,
)


class DynamicBaselineSummaryTests(unittest.TestCase):
    def valid_row(self):
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
            "measurement_monitor_samples": "10",
            "measurement_monitor_failures": "0",
            "measurement_llama_pid": "123",
            "require_device_idle_configured": "True",
            "cooldown_thermal_max_c_configured": "42",
            "cooldown_thermal_status_max_configured": "0",
            "min_battery_level_pct_configured": "20",
            "thermal_status_before": "0",
            "thermal_cpu_max_c_before": "34",
            "thermal_gpu_max_c_before": "35",
            "thermal_skin_max_c_before": "30",
            "battery_level_pct_before": "90",
            "pipeline_budget_violations": "0",
            "expected_model_sha256": "model-hash",
            "remote_model_sha256": "model-hash",
            "expected_binary_sha256": "binary-hash",
            "remote_binary_sha256": "binary-hash",
            "model_meta_sha256": "meta-hash",
            "remote_model_meta_sha256": "meta-hash",
            "cost_dir_sha256": "cost-hash",
            "remote_cost_dir_sha256": "cost-hash",
            "dynamic_run_config_sha256": "config-hash",
            "dynamic_run_core_config_sha256": "core-config-hash",
            "pin_contract_valid": "True",
            "fused_lane_contract_valid": "True",
            "unit_pipeline_stage_authority_contract_valid": "True",
            "plan_stage_defer_apply_count": "4",
            "plan_stage_defer_min": "1",
            "plan_stage_defer_max": "1",
            "plan_anchor_fired": "0",
            "plan_anchor_load_events": "0",
            "plan_anchor_transfer_events": "0",
            "plan_anchor_xform_events": "0",
            "backend_compute_contract_valid": "True",
            "physical_cut_tensors": "7",
            "physical_cut_parts": "14",
            "wbm_total_mib": "4436.789",
            "expected_wbm_total_mib_configured": "4436.7890625",
            "planner_stream_reserve_mib_configured": "16",
            "pin_outside_budget_count": "0",
            "pipeline_plan_protection_relaxations": "0",
            "source_start_sec": "0",
            "source_span_sec": "600",
            "source_file_sha256": "source-hash",
            "expected_source_trace_sha256": "source-hash",
            "normalized_source_input_sha256": "source-hash",
            "expected_normalized_source_input_sha256": "source-hash",
            "original_source_trace": "/tmp/historical-trace.csv",
            "original_source_trace_sha256": "historical-hash",
            "expected_original_source_trace_sha256":
                "historical-hash",
            "original_source_trace_semantically_equal": "True",
            "replay_span_sec": "600",
            "runtime_trace_sha256": "trace-hash",
            "remote_runtime_trace_sha256": "trace-hash",
            "bench_exit_reason": "duration",
            "bench_time_done": "1",
            "budget_decode_reset_count": "1",
            "budget_decode_reset_generated": "0",
            "budget_backend_reset_count": "1",
            "frontier_apply_count": "4",
            "frontier_multi_units_max": "8",
            "frontier_tensor_units_max": "0",
            "frontier_cut_units_max": "12",
            "frontier_mixed_mode_generations": "2",
            "frontier_trace_sha256": "residual-frontier-hash",
            "frontier_transition_trace_sha256":
                "residual-transition-path-hash",
            "frontier_state_changes": "3",
            "frontier_transition_publish_count": "3",
            "frontier_transition_publish_ms_total": "1.5",
            "frontier_transition_publish_ms_max": "0.8",
            "frontier_transition_delta_weights_total": "6",
            "frontier_transition_delta_weights_max": "3",
            "online_calls": "4",
            "remote_solver_fresh_process": "True",
            "remote_solver_contract_valid": "True",
            "granularity_profile": "/tmp/profile.json",
            "granularity_profile_sha256": "residual-hash",
            "granularity_profile_kind": "pipeline-residual",
            "multi_implementation_configured": "multi",
            "cut_compute_grouping_configured": "independent_halves",
            "transition_fixed_ms_configured": "0.8",
            "transition_cost_source_configured":
                "held-in isolated frontier publication",
            "granularity_placement_source": "stateful-cp",
            "granularity_placement_source_configured": "stateful-cp",
            "granularity_policy_configured": "mixed-cost-model",
            "granularity_policy_effective": "diff-tree",
            "mixed_offline_table_sha256": "residual-table-hash",
            "remote_mixed_offline_table_sha256":
                "residual-table-hash",
            "offline_table_sha256": "tensor-table-hash",
            "remote_offline_table_sha256": "tensor-table-hash",
        }

    def test_validate_accepts_full_audited_trace(self):
        validate("Diff-now", self.valid_row())

    def test_validate_rejects_unobserved_measurement_process(self):
        row = self.valid_row()
        row["measurement_monitor_samples"] = "0"
        row["measurement_llama_pid"] = ""
        with self.assertRaisesRegex(
                ValueError, "measurement isolation samples"):
            validate("Diff-now", row)

    def test_validate_rejects_missing_stream_reserve(self):
        row = self.valid_row()
        row["planner_stream_reserve_mib_configured"] = ""
        with self.assertRaisesRegex(
                ValueError, "planner stream reserve configuration"):
            validate("Diff-now", row)

    def test_validate_rejects_missing_cut_compute_grouping(self):
        row = self.valid_row()
        row["cut_compute_grouping_configured"] = ""
        with self.assertRaisesRegex(
                ValueError, "Cut compute grouping contract"):
            validate("Diff-now", row)

    def test_gpu_cut_lineage_rejects_budget_fallback(self):
        rows = {
            "Diff-now": {
                "cut_compute_grouping_configured": "fused_pair",
                "cut_dual_budget_fallbacks": "0",
            },
            "Online": {
                "cut_compute_grouping_configured": "fused_pair",
                "cut_dual_budget_fallbacks": "0",
            },
        }
        validate_cut_compute_lineage("gpu", rows)
        rows["Online"]["cut_dual_budget_fallbacks"] = "1"
        with self.assertRaisesRegex(
                ValueError, "used 1 budget fallback"):
            validate_cut_compute_lineage("gpu", rows)
        rows["Online"]["cut_dual_budget_fallbacks"] = ""
        with self.assertRaisesRegex(
                ValueError, "missing fused Cut"):
            validate_cut_compute_lineage("gpu", rows)

    def test_validate_rejects_remote_mixed_table_mismatch(self):
        row = self.valid_row()
        row["remote_mixed_offline_table_sha256"] = "stale-table"
        with self.assertRaisesRegex(
                ValueError, "mixed offline plan table device lineage"):
            validate("Diff-now", row)

    def test_validate_rejects_remote_runtime_trace_mismatch(self):
        row = self.valid_row()
        row["remote_runtime_trace_sha256"] = "old-trace"
        with self.assertRaisesRegex(
                ValueError, "remote runtime trace lineage"):
            validate("Diff-now", row)

    def test_validate_rejects_historical_trace_mismatch(self):
        row = self.valid_row()
        row["original_source_trace_sha256"] = "other-history"
        with self.assertRaisesRegex(
                ValueError, "historical original trace lineage"):
            validate("Diff-now", row)

    def test_validate_rejects_plan_protection_relaxation(self):
        row = self.valid_row()
        row["pipeline_plan_protection_relaxations"] = "1"
        with self.assertRaisesRegex(
                ValueError, "plan-protection relaxation"):
            validate("Diff-now", row)

    def test_validate_rejects_mixed_whole_weight_stage_authority(self):
        row = self.valid_row()
        row[
            "unit_pipeline_stage_authority_contract_valid"
        ] = "False"
        with self.assertRaisesRegex(
                ValueError, "unit-pipeline stage authority"):
            validate("Diff-now", row)

    def test_validate_rejects_unverified_remote_model(self):
        row = self.valid_row()
        row["remote_model_sha256"] = "stale-model"
        with self.assertRaisesRegex(
                ValueError, "remote model lineage"):
            validate("Diff-now", row)

    def test_validate_rejects_missing_plan_protection_audit(self):
        row = self.valid_row()
        del row["pipeline_plan_protection_relaxations"]
        with self.assertRaisesRegex(
                ValueError, "plan-protection audit missing"):
            validate("Diff-now", row)

    def test_validate_rejects_nonzero_trace_start(self):
        row = self.valid_row()
        row["source_start_sec"] = "130"
        with self.assertRaisesRegex(ValueError, "trace start"):
            validate("Diff-now", row)

    def test_validate_rejects_missing_clean_start_temperature(self):
        row = self.valid_row()
        row["thermal_cpu_max_c_before"] = ""
        row["thermal_gpu_max_c_before"] = ""
        row["thermal_skin_max_c_before"] = ""
        with self.assertRaisesRegex(
                ValueError, "clean-start temperature audit missing"):
            validate("Diff-now", row)

    def test_validate_rejects_unexercised_mixed_frontier(self):
        row = self.valid_row()
        row["frontier_cut_units_max"] = "0"
        with self.assertRaisesRegex(
                ValueError, "mixed-granularity frontier not exercised"):
            validate("Diff-now", row)

    def test_validate_rejects_missing_changed_frontier_timing(self):
        row = self.valid_row()
        row["frontier_transition_publish_count"] = "2"
        with self.assertRaisesRegex(
                ValueError, "changed-frontier publication count mismatch"):
            validate("Diff-now", row)

        row = self.valid_row()
        row["frontier_transition_publish_ms_max"] = "0"
        with self.assertRaisesRegex(
                ValueError, "changed-frontier publication audit missing"):
            validate("Diff-now", row)

        row = self.valid_row()
        row["frontier_transition_delta_weights_total"] = "0"
        with self.assertRaisesRegex(
                ValueError, "changed-frontier publication audit missing"):
            validate("Diff-now", row)

    def test_validate_rejects_missing_granularity_profile_hash(self):
        row = self.valid_row()
        row["granularity_profile_sha256"] = ""
        with self.assertRaisesRegex(
                ValueError, "granularity profile hash"):
            validate("Diff-now", row)

    def test_validate_rejects_wrong_effective_policy(self):
        row = self.valid_row()
        row["granularity_policy_effective"] = "online"
        with self.assertRaisesRegex(
                ValueError, "effective granularity policy"):
            validate("Diff-now", row)

    def test_validate_rejects_offline_placement_for_online_diff(self):
        row = self.valid_row()
        row["granularity_placement_source"] = "offline-table"
        with self.assertRaisesRegex(
                ValueError, "granularity placement source"):
            validate("Diff-now", row)

    def test_validate_rejects_noncommon_physical_tiling(self):
        row = self.valid_row()
        row["physical_cut_parts"] = "7"
        with self.assertRaisesRegex(
                ValueError, "physical tiling audit mismatch"):
            validate("Diff-now", row)

    def test_profile_lineage_accepts_one_residual_and_phase_ablation(self):
        selected = {
            label: self.valid_row()
            for label in (
                "Offline", "Online", "Diff-before", "Diff-now")
        }
        selected["Offline"]["granularity_placement_source"] = (
            "offline-table")
        selected["Diff-before"]["granularity_profile_sha256"] = (
            "phase-hash")
        selected["Diff-before"]["granularity_profile_kind"] = "phase-only"
        selected["Diff-before"]["frontier_trace_sha256"] = (
            "phase-frontier-hash")
        selected["Diff-before"]["frontier_transition_trace_sha256"] = (
            "phase-transition-path-hash")
        validate_profile_lineage(selected)

    def test_profile_lineage_rejects_unchanged_diff_frontier(self):
        selected = {
            label: self.valid_row()
            for label in (
                "Offline", "Online", "Diff-before", "Diff-now")
        }
        selected["Offline"]["granularity_placement_source"] = (
            "offline-table")
        selected["Diff-before"]["granularity_profile_sha256"] = (
            "phase-hash")
        selected["Diff-before"]["granularity_profile_kind"] = "phase-only"
        with self.assertRaisesRegex(
                ValueError, "different ordered split/merge frontier"):
            validate_profile_lineage(selected)

    def test_profile_lineage_rejects_different_online_profile(self):
        selected = {
            label: self.valid_row()
            for label in (
                "Offline", "Online", "Diff-before", "Diff-now")
        }
        selected["Offline"]["granularity_placement_source"] = (
            "offline-table")
        selected["Diff-before"]["granularity_profile_sha256"] = (
            "phase-hash")
        selected["Diff-before"]["granularity_profile_kind"] = "phase-only"
        selected["Diff-before"]["frontier_transition_trace_sha256"] = (
            "phase-transition-path-hash")
        selected["Online"]["granularity_profile_sha256"] = "other-hash"
        with self.assertRaisesRegex(
                ValueError, "must share one granularity profile hash"):
            validate_profile_lineage(selected)

    def test_profile_lineage_rejects_different_residual_table(self):
        selected = {
            label: self.valid_row()
            for label in (
                "Offline", "Online", "Diff-before", "Diff-now")
        }
        selected["Offline"]["granularity_placement_source"] = (
            "offline-table")
        selected["Diff-before"]["granularity_profile_sha256"] = (
            "phase-hash")
        selected["Diff-before"]["granularity_profile_kind"] = "phase-only"
        selected["Diff-before"]["frontier_transition_trace_sha256"] = (
            "phase-transition-path-hash")
        selected["Online"]["mixed_offline_table_sha256"] = (
            "other-table")
        with self.assertRaisesRegex(
                ValueError, "share one mixed offline table hash"):
            validate_profile_lineage(selected)

    def test_profile_lineage_rejects_mixed_coarse_implementation(self):
        selected = {
            label: self.valid_row()
            for label in (
                "Offline", "Online", "Diff-before", "Diff-now")
        }
        selected["Offline"]["granularity_placement_source"] = (
            "offline-table")
        selected["Diff-before"]["granularity_profile_sha256"] = (
            "phase-hash")
        selected["Diff-before"]["granularity_profile_kind"] = "phase-only"
        selected["Diff-before"]["frontier_transition_trace_sha256"] = (
            "phase-transition-path-hash")
        selected["Diff-before"]["multi_implementation_configured"] = (
            "multi_fused")
        with self.assertRaisesRegex(
                ValueError, "one coarse-unit implementation"):
            validate_profile_lineage(selected)

    def test_profile_lineage_rejects_mixed_transition_calibration(self):
        selected = {
            label: self.valid_row()
            for label in (
                "Offline", "Online", "Diff-before", "Diff-now")
        }
        selected["Offline"]["granularity_placement_source"] = (
            "offline-table")
        selected["Diff-before"]["granularity_profile_sha256"] = (
            "phase-hash")
        selected["Diff-before"]["granularity_profile_kind"] = "phase-only"
        selected["Diff-before"]["frontier_trace_sha256"] = (
            "phase-frontier-hash")
        selected["Diff-before"]["frontier_transition_trace_sha256"] = (
            "phase-transition-path-hash")
        selected["Online"]["transition_fixed_ms_configured"] = "0.9"
        with self.assertRaisesRegex(
                ValueError, "one positive measured frontier transition"):
            validate_profile_lineage(selected)

        selected["Online"]["transition_fixed_ms_configured"] = "0.8"
        selected["Online"]["transition_cost_source_configured"] = ""
        with self.assertRaisesRegex(
                ValueError, "one positive measured frontier transition"):
            validate_profile_lineage(selected)

    def test_runtime_input_lineage_rejects_mixed_binaries(self):
        selected = {
            label: self.valid_row()
            for label in (
                "Static-Min", "Static-Max", "MRU", "Offline",
                "Online", "Diff-before", "Diff-now")
        }
        selected["Online"]["remote_binary_sha256"] = "other-binary"
        with self.assertRaisesRegex(
                ValueError, "one verified binary SHA-256"):
            validate_runtime_input_lineage(selected)

    def test_runtime_input_lineage_rejects_mixed_cost_directory(self):
        selected = {
            label: self.valid_row()
            for label in (
                "Static-Min", "Static-Max", "MRU", "Offline",
                "Online", "Diff-before", "Diff-now")
        }
        selected["Online"]["remote_cost_dir_sha256"] = "other-cost"
        with self.assertRaisesRegex(
                ValueError, "one verified cost_dir SHA-256"):
            validate_runtime_input_lineage(selected)

    def test_runtime_input_lineage_rejects_mixed_runtime_config(self):
        selected = {
            label: self.valid_row()
            for label in (
                "Static-Min", "Static-Max", "MRU", "Offline",
                "Online", "Diff-before", "Diff-now")
        }
        selected["Diff-before"]["dynamic_run_config_sha256"] = (
            "other-config")
        with self.assertRaisesRegex(
                ValueError, "one dynamic runtime configuration"):
            validate_runtime_input_lineage(selected)

    def test_runtime_input_lineage_rejects_mixed_core_config(self):
        selected = {
            label: self.valid_row()
            for label in (
                "Static-Min", "Static-Max", "MRU", "Offline",
                "Online", "Diff-before", "Diff-now")
        }
        selected["Diff-before"]["dynamic_run_core_config_sha256"] = (
            "other-core-config")
        with self.assertRaisesRegex(
                ValueError, "one dynamic runtime core configuration"):
            validate_runtime_input_lineage(selected)

    def test_runtime_input_lineage_rejects_mixed_physical_tiling(self):
        selected = {
            label: self.valid_row()
            for label in (
                "Static-Min", "Static-Max", "MRU", "Offline",
                "Online", "Diff-before", "Diff-now")
        }
        selected["Diff-before"]["wbm_total_mib"] = "4000"
        with self.assertRaisesRegex(
                ValueError, "one physical tiling and WBM total"):
            validate_runtime_input_lineage(selected)

    def test_only_result_filters_even_single_wrong_method(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "results.csv"
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=["method"])
                writer.writeheader()
                writer.writerow({"method": "online"})
            with self.assertRaisesRegex(ValueError, "found 0"):
                only_result(path, "diff-tree-mixed")

    def test_budget_trace_points_reads_runtime_schema(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "static.csv"
            path.write_text(
                "t_sec,mem_available_mb\n"
                "0,3712\n"
                "600,3712\n",
                encoding="utf-8",
            )
            self.assertEqual(
                budget_trace_points(path),
                [(0.0, 3712.0), (600.0, 3712.0)])

    def test_raw_log_enrichment_preserves_runner_invalid_status(self):
        with tempfile.TemporaryDirectory() as temporary:
            log = Path(temporary) / "run.log"
            log.write_text(
                "elastic benchmark exit: reason=duration time_done=1\n",
                encoding="utf-8",
            )
            row = {
                "status": "frequency_invalid",
                "method": "static-min",
                "log": str(log),
            }
            self.assertEqual(
                enrich_from_raw_log(row)["status"],
                "frequency_invalid",
            )


if __name__ == "__main__":
    unittest.main()
