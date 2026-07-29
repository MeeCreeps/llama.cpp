import json
import tempfile
import unittest
from pathlib import Path

from runtime.plan.calibrate_granularity_profile import (
    fit_gpu_compute_from_resident_tail,
    fit_phase,
    fastest_mode_by_ratio,
    normalized_curve,
    pipeline_curve,
    resident_ratio,
    require_calibration_matrix,
    require_input_lineage,
    require_physical_tiling_contract,
    select_multi_implementation,
    strip_pipeline_residual_profile,
    valid_row,
    write_cpu_placement_cost_dir,
    write_gpu_placement_cost_dir,
)


class GranularityProfileCalibrationTest(unittest.TestCase):
    def test_diff_before_removes_only_pipeline_residual(self):
        root = {
            "source": "measured",
            "profiles": {
                "gpu": {
                    "cut_compute_grouping": "fused_pair",
                    "multi_implementation": "multi",
                    "transition_fixed_ms": 0.8,
                    "transition_ms_per_mib": 0.0,
                    "transition_cost_source":
                        "held-in isolated frontier publication",
                    "mode_pipeline_efficiency_curve": {
                        "cut": [{"resident_ratio": 0.3, "scale": 0.8}],
                    },
                    "mode_prepare_scale": {"cut": 0.7},
                },
            },
            "calibration": {
                "gpu": {
                    "pipeline_efficiency_residual_curve": {
                        "enabled": True,
                        "points": 7,
                    },
                    "cut_compute_grouping": "fused_pair",
                },
            },
        }
        stripped = strip_pipeline_residual_profile(root, "gpu")
        profile = stripped["profiles"]["gpu"]
        self.assertNotIn("mode_pipeline_efficiency_curve", profile)
        self.assertEqual(profile["cut_compute_grouping"], "fused_pair")
        self.assertEqual(profile["mode_prepare_scale"], {"cut": 0.7})
        self.assertEqual(profile["transition_fixed_ms"], 0.8)
        self.assertEqual(profile["transition_ms_per_mib"], 0.0)
        self.assertEqual(
            profile["transition_cost_source"],
            "held-in isolated frontier publication",
        )
        self.assertEqual(stripped["source"], "measured-phase-only")
        residual = stripped["calibration"]["gpu"][
            "pipeline_efficiency_residual_curve"]
        self.assertFalse(residual["enabled"])
        self.assertEqual(residual["baseline"], "Diff-before")
        # The source profile remains reusable for Diff-now.
        self.assertIn(
            "mode_pipeline_efficiency_curve",
            root["profiles"]["gpu"],
        )

    def test_fixed_winners_come_from_latency_not_expected_labels(self):
        rows = [
            {
                "ratio_pct": "60",
                "mode": "multi",
                "steady_median_latency_ms": "90",
            },
            {
                "ratio_pct": "60",
                "mode": "tensor",
                "steady_median_latency_ms": "80",
            },
            {
                "ratio_pct": "60",
                "mode": "cut",
                "steady_median_latency_ms": "70",
            },
            {
                "ratio_pct": "90",
                "mode": "multi",
                "steady_median_latency_ms": "40",
            },
            {
                "ratio_pct": "90",
                "mode": "tensor",
                "steady_median_latency_ms": "50",
            },
            {
                "ratio_pct": "90",
                "mode": "cut",
                "steady_median_latency_ms": "60",
            },
        ]
        self.assertEqual(
            fastest_mode_by_ratio(rows),
            {"60": "cut", "90": "multi"},
        )

    def test_calibration_rejects_zero_sample_isolation_audit(self):
        row = {
            "backend": "cpu",
            "mode": "tensor",
            "returncode": "0",
            "valid_runtime": "true",
            "decode_wall_contract_valid": "true",
            "decode_phase_contract_valid": "true",
            "decode_phase_counter_scope":
                "decode-only-backend-counter-delta",
            "latency_timing_source": "decode-wall",
            "valid_token_sequence": "true",
            "valid_device_idle": "true",
            "valid_device_awake": "true",
            "steady_median_latency_ms": "10",
            "measurement_monitor_samples": "0",
            "measurement_monitor_failures": "0",
            "measurement_llama_pid": "",
            "budget_trace_sha256": "trace-hash",
            "remote_budget_trace_sha256": "trace-hash",
        }
        self.assertFalse(valid_row(row, "cpu"))
        row.update({
            "measurement_monitor_samples": "2",
            "measurement_llama_pid": "123",
        })
        self.assertTrue(valid_row(row, "cpu"))

    def test_requires_one_fixed_sweep_input_lineage(self):
        rows = [
            {
                "remote_binary_sha256": "binary-a",
                "remote_model_sha256": "model-a",
                "fixed_run_config_sha256": "config-a",
            },
            {
                "remote_binary_sha256": "binary-a",
                "remote_model_sha256": "model-a",
                "fixed_run_config_sha256": "config-a",
            },
        ]
        self.assertEqual(
            require_input_lineage(rows),
            ("binary-a", "model-a"),
        )
        rows[1]["remote_binary_sha256"] = "binary-b"
        with self.assertRaisesRegex(
                ValueError, "mixes or omits runtime input lineage"):
            require_input_lineage(rows)
        rows[1]["remote_binary_sha256"] = "binary-a"
        rows[1]["fixed_run_config_sha256"] = "config-b"
        with self.assertRaisesRegex(
                ValueError, "mixes or omits runtime input lineage"):
            require_input_lineage(rows)

    def test_requires_complete_valid_fixed_budget_matrix(self):
        rows = [
            {"mode": mode, "ratio_pct": str(ratio)}
            for ratio in (30, 90)
            for mode in ("multi", "tensor", "cut")
        ]
        require_calibration_matrix(
            rows,
            ratios=(30, 90),
            modes=("multi", "tensor", "cut"),
        )
        with self.assertRaisesRegex(
                ValueError, "multi_fused@30%.*multi_fused@90%"):
            require_calibration_matrix(
                rows,
                ratios=(30, 90),
                modes=("multi", "multi_fused", "tensor", "cut"),
            )

    def test_requires_one_physical_representation_per_budget(self):
        rows = [
            {
                "mode": mode,
                "ratio_pct": str(ratio),
                "physical_cut_tensors": "7",
                "physical_cut_parts": "14",
                "wbm_total_mib": "4436.789062",
                "weight_target_mib": str(4436.789062 * ratio / 100),
            }
            for ratio in (30, 90)
            for mode in ("multi", "multi_fused", "tensor", "cut")
        ]
        audit = require_physical_tiling_contract(rows)
        self.assertEqual(audit["physical_cut_tensors"], 7)
        self.assertEqual(audit["physical_cut_parts"], 14)
        rows[1]["weight_target_mib"] = "1400"
        with self.assertRaisesRegex(
                ValueError, "different physical targets"):
            require_physical_tiling_contract(rows)
        rows[1]["weight_target_mib"] = rows[0]["weight_target_mib"]
        rows[1]["physical_cut_parts"] = "7"
        with self.assertRaisesRegex(
                ValueError, "invalid common physical representation"):
            require_physical_tiling_contract(rows)

    def test_selects_one_multi_implementation_across_all_budgets(self):
        rows = []
        for ratio, ordinary, fused in (
            (30, 100.0, 95.0),
            (60, 80.0, 76.0),
            (90, 60.0, 57.0),
        ):
            rows.extend([
                {
                    "ratio_pct": str(ratio),
                    "mode": "multi",
                    "steady_median_latency_ms": str(ordinary),
                },
                {
                    "ratio_pct": str(ratio),
                    "mode": "multi_fused",
                    "steady_median_latency_ms": str(fused),
                },
                {
                    "ratio_pct": str(ratio),
                    "mode": "tensor",
                    "steady_median_latency_ms": "90",
                },
                {
                    "ratio_pct": str(ratio),
                    "mode": "cut",
                    "steady_median_latency_ms": "85",
                },
            ])
        selected, implementation, details = select_multi_implementation(rows)
        self.assertEqual(implementation, "multi_fused")
        self.assertEqual(details["common_ratios"], [30, 60, 90])
        self.assertTrue(all(row["mode"] != "multi_fused"
                            for row in selected))
        self.assertEqual(
            sum(row["mode"] == "multi" for row in selected), 3)

    def test_fit_phase_recovers_common_launch_and_mode_rates(self):
        rates = {"multi": 0.2, "tensor": 0.3, "cut": 0.4}
        rows = []
        for mode in ("multi", "tensor", "cut"):
            for calls, work in ((10.0, 100.0), (20.0, 250.0)):
                rows.append({
                    "mode": mode,
                    "elapsed": str(0.5 * calls + rates[mode] * work),
                    "work": str(work),
                    "calls": str(calls),
                })
        launch, fitted, samples = fit_phase(
            rows,
            time_key="elapsed",
            work_key="work",
            calls_key="calls",
        )
        self.assertEqual(samples, 6)
        self.assertAlmostEqual(launch, 0.5, places=8)
        for mode in rates:
            self.assertAlmostEqual(fitted[mode], rates[mode], places=8)

    def test_work_curve_is_normalized_to_tensor_at_each_ratio(self):
        rows = [
            {"ratio_pct": "70", "mode": "multi", "work": "80"},
            {"ratio_pct": "70", "mode": "tensor", "work": "100"},
            {"ratio_pct": "70", "mode": "cut", "work": "60"},
            {"ratio_pct": "90", "mode": "multi", "work": "45"},
            {"ratio_pct": "90", "mode": "tensor", "work": "50"},
            {"ratio_pct": "90", "mode": "cut", "work": "55"},
        ]
        curve = normalized_curve(rows, "work")
        self.assertEqual(
            curve["tensor"],
            [
                {"resident_ratio": 0.7, "scale": 1.0},
                {"resident_ratio": 0.9, "scale": 1.0},
            ],
        )
        self.assertAlmostEqual(curve["multi"][0]["scale"], 0.8)
        self.assertAlmostEqual(curve["cut"][1]["scale"], 1.1)

    def test_pipeline_residual_uses_critical_stage_not_additive_sum(self):
        rows = []
        for mode, observed, io_ms, prepare_ms, compute_ms in (
            ("tensor", 12.0, 80.0, 100.0, 40.0),
            ("multi", 13.0, 90.0, 100.0, 40.0),
            ("cut", 11.0, 70.0, 100.0, 60.0),
        ):
            rows.append({
                "ratio_pct": "50",
                "mode": mode,
                "decode_phase_runs": "10",
                "steady_median_latency_ms": str(observed),
                "decode_phase_load_ms": str(io_ms),
                "decode_phase_prepare_ms": str(prepare_ms),
                "decode_phase_compute_ms": str(compute_ms),
            })
        curve, details = pipeline_curve(
            rows, prepare_time_key="decode_phase_prepare_ms")
        self.assertAlmostEqual(curve["tensor"][0]["scale"], 1.0)
        self.assertAlmostEqual(curve["multi"][0]["scale"], 13.0 / 12.0)
        self.assertAlmostEqual(curve["cut"][0]["scale"], 11.0 / 12.0)
        point = details["0.5"]["cut"]
        self.assertIn("critical_stage_scale", point)
        self.assertNotIn("additive_phase_scale", point)

    def test_gpu_compute_fit_uses_resident_tail_not_cpu_fallback_counter(self):
        rows = []
        for mode, observed, wait in (
            ("multi", 240.0, 300.0),
            ("tensor", 260.0, 300.0),
            ("cut", 280.0, 300.0),
        ):
            rows.extend([
                {
                    "ratio_pct": "80",
                    "weight_target_mib": "800",
                    "mode": mode,
                    "steady_median_latency_ms": str(observed + 100.0),
                    "decode_phase_pipeline_wait_ms": "900",
                    "decode_phase_runs": "10",
                    # This is the CPU fallback counter that must be ignored.
                    "compute_total_ms": "0.001",
                },
                {
                    "ratio_pct": "90",
                    "weight_target_mib": "900",
                    "mode": mode,
                    "steady_median_latency_ms": str(observed),
                    "decode_phase_pipeline_wait_ms": str(wait),
                    "decode_phase_runs": "10",
                    "compute_total_ms": "0.001",
                },
            ])
        launch, rates, samples, details = (
            fit_gpu_compute_from_resident_tail(
                rows, planned_weight_mib=1000.0)
        )
        self.assertEqual(launch, 0.0)
        self.assertEqual(samples, 3)
        self.assertAlmostEqual(rates["multi"], 0.210)
        self.assertAlmostEqual(rates["tensor"], 0.230)
        self.assertAlmostEqual(rates["cut"], 0.250)
        self.assertEqual(details["highest_resident_ratio"], 0.9)
        self.assertIn(
            "OpenCL event profiling disabled", details["source"])

    def test_pipeline_residual_accepts_external_gpu_compute_floor(self):
        rows = []
        for mode, observed in (
            ("multi", 20.0),
            ("tensor", 25.0),
            ("cut", 30.0),
        ):
            rows.append({
                "ratio_pct": "90",
                "mode": mode,
                "decode_phase_runs": "10",
                "steady_median_latency_ms": str(observed),
                "decode_phase_load_ms": "20",
                "decode_phase_prepare_ms": "30",
                # Deliberately bogus CPU fallback value.
                "compute_total_ms": "0.001",
            })
        curve, details = pipeline_curve(
            rows,
            prepare_time_key="decode_phase_prepare_ms",
            compute_per_forward_ms={
                "multi": 20.0,
                "tensor": 25.0,
                "cut": 30.0,
            },
        )
        for mode in ("multi", "tensor", "cut"):
            self.assertAlmostEqual(curve[mode][0]["scale"], 1.0)
        self.assertAlmostEqual(
            details["0.9"]["cut"]["critical_stage_scale"], 1.2)

    def test_cpu_placement_profile_uses_tensor_stage_fit(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            base = root / "base"
            output = root / "cpu"
            base.mkdir()
            (base / "op_costs.json").write_text(
                '{"records": [{"backend": "CPU_Elastic"}]}')
            write_cpu_placement_cost_dir(
                output_dir=output,
                base_cost_dir=base,
                source="op12-test",
                io_launch_ms=0.1,
                io_ms_per_mib=0.2,
                prepare_launch_ms=0.3,
                prepare_ms_per_mib=0.4,
            )
            calibration = json.loads(
                (output / "calibration.json").read_text())
            self.assertEqual(
                calibration["stage_fallback_models"][
                    "CPU_Elastic:XFORM"]["ms_per_mib"],
                0.4)
            self.assertTrue((output / "op_costs.json").exists())

    def test_gpu_placement_profile_counts_combined_prepare_once(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            base = root / "base"
            output = root / "gpu"
            base.mkdir()
            (base / "op_costs.json").write_text(
                '{"records": [{"backend": "OpenCL"}]}')
            write_gpu_placement_cost_dir(
                output_dir=output,
                base_cost_dir=base,
                source="op12-gpu-test",
                io_launch_ms=0.1,
                io_ms_per_mib=0.2,
                prepare_launch_ms=0.3,
                prepare_ms_per_mib=0.4,
            )
            calibration = json.loads(
                (output / "calibration.json").read_text())
            models = calibration["stage_fallback_models"]
            self.assertEqual(models["OpenCL:LOAD"]["ms_per_mib"], 0.2)
            self.assertEqual(models["OpenCL:XFORM"]["ms_per_mib"], 0.4)
            self.assertEqual(models["OpenCL:TRANSFER"]["ms_per_mib"], 0.0)
            self.assertEqual(models["OpenCL:SYNC"]["ms_per_mib"], 0.0)
            self.assertTrue((output / "op_costs.json").exists())

    def test_physical_target_is_mapped_to_planner_resident_ratio(self):
        row = {
            "ratio_pct": "30",
            "weight_target_mib": "1331.34",
        }
        self.assertAlmostEqual(
            resident_ratio(
                row,
                planned_weight_mib=4154.9765625,
                unplanned_pinned_mib=281.8125,
            ),
            (1331.34 - 281.8125) / 4154.9765625,
        )


if __name__ == "__main__":
    unittest.main()
