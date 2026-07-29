#!/usr/bin/env python3

import tempfile
import unittest
from pathlib import Path
from subprocess import CompletedProcess
from unittest import mock

from runtime.plan.run_dynamic_budget_matrix import (
    apply_plan_residency_policy,
    backend_compute_contract_failures,
    backend_safety_mib,
    configure_unit_pipeline_stage_authority,
    cooldown_snapshot_ready,
    cpu_frequency_contract_failures,
    decode_phase_contract_failures,
    decode_wall_contract_failures,
    directory_file_manifest,
    directory_sha256,
    dynamic_run_core_config_sha256,
    dynamic_run_config_sha256,
    effective_granularity_placement_source,
    effective_granularity_policy,
    external_inference_processes,
    fixed_performance_measurement,
    frontier_contract_failures,
    granularity_profile_cut_compute_grouping,
    granularity_profile_kind,
    granularity_profile_multi_implementation,
    granularity_profile_transition_contract,
    mark_token_correctness,
    matrix_completion_failures,
    methods_need_mixed_offline_table,
    monitor_measurement_overlap,
    parse_log,
    pin_policy_requests,
    pipeline_worker_was_required,
    read_thermal_snapshot,
    read_trace,
    required_plan_budgets,
    remote_solver_log_contract,
    remote_server_granularity_policy,
    resume_environment_contract_failures,
    token_sequence,
    token_trace_timing_contract_failures,
    timed_run_contract_failures,
    unit_pipeline_stage_authority_failures,
    validate_remote_sha256,
    validate_remote_directory_contents,
    write_window_trace,
)
from runtime.plan.dynamic_budget_solver import (
    CostModel,
    parse_forced_weight_placements,
    planner_resident_budget_bytes,
)
from runtime.plan.summarize_dynamic_baseline_matrix import format_metric


class DynamicBudgetMatrixTraceTest(unittest.TestCase):
    def test_dynamic_resume_config_excludes_method_list_but_tracks_pipeline(
            self):
        class Args:
            execution_backend = "cpu"
            methods = "online,diff-tree-mixed"
            granularity_pipeline_lookahead_mib = 128

        first = dynamic_run_config_sha256(Args())
        Args.methods = "diff-tree-mixed"
        self.assertEqual(first, dynamic_run_config_sha256(Args()))
        Args.granularity_pipeline_lookahead_mib = 64
        self.assertNotEqual(first, dynamic_run_config_sha256(Args()))

    def test_dynamic_config_canonicalizes_original_trace_path(self):
        class Args:
            original_source_trace = Path("/tmp/original-trace.csv")
            expected_original_source_trace_sha256 = "a" * 64

        digest = dynamic_run_config_sha256(Args())
        self.assertEqual(len(digest), 64)
        Args.original_source_trace = Path("/tmp/other-trace.csv")
        self.assertNotEqual(digest, dynamic_run_config_sha256(Args()))

    def test_residency_pilot_core_config_omits_only_policy(self):
        class Args:
            execution_backend = "cpu"
            plan_residency_policy = "strict"
            planner_stream_reserve_mib = 16

        full = dynamic_run_config_sha256(Args())
        core = dynamic_run_core_config_sha256(Args())
        Args.plan_residency_policy = "reuse-stream"
        self.assertNotEqual(full, dynamic_run_config_sha256(Args()))
        self.assertEqual(core, dynamic_run_core_config_sha256(Args()))
        Args.planner_stream_reserve_mib = 8
        self.assertNotEqual(core, dynamic_run_core_config_sha256(Args()))

    def test_matrix_completion_fails_immediately_on_invalid_row(self):
        rows = [{
            "trace": "trace.csv",
            "method": "online",
            "status": "frequency_invalid",
            "rc": "0",
            "valid_token_trace": "True",
            "token_trace_timing_contract_valid": "True",
            "decode_wall_contract_valid": "True",
            "decode_phase_contract_valid": "True",
            "valid_measurement_isolation": "True",
            "measurement_monitor_samples": "10",
            "measurement_monitor_failures": "0",
            "measurement_llama_pid": "123",
        }]
        failures = matrix_completion_failures(
            rows,
            trace_names={"trace.csv"},
            methods={"online"},
            require_device_idle=True,
        )
        self.assertEqual(
            failures,
            ["trace.csv:online: status=frequency_invalid"],
        )

    def test_matrix_completion_requires_fixed_performance_restore(self):
        rows = [{
            "trace": "trace.csv",
            "method": "online",
            "status": "ok",
            "rc": "0",
            "valid_token_trace": "True",
            "token_trace_timing_contract_valid": "True",
            "decode_wall_contract_valid": "True",
            "decode_phase_contract_valid": "True",
            "valid_token_sequence": "True",
            "fixed_performance_restore_valid": "False",
        }]
        self.assertEqual(
            matrix_completion_failures(
                rows,
                trace_names={"trace.csv"},
                methods={"online"},
                require_device_idle=False,
                require_fixed_performance_restore=True,
            ),
            ["trace.csv:online: fixed-performance-restore"],
        )

    @mock.patch(
        "runtime.plan.run_dynamic_budget_matrix."
        "set_fixed_performance_mode")
    def test_fixed_performance_context_restores_after_exception(
            self, set_mode):
        class Args:
            fixed_performance_mode = True

        with self.assertRaisesRegex(RuntimeError, "measured failure"):
            with fixed_performance_measurement(Args()) as audit:
                self.assertTrue(
                    audit[
                        "fixed_performance_mode_enabled_for_measurement"])
                raise RuntimeError("measured failure")
        self.assertEqual(
            [call.args[1] for call in set_mode.call_args_list],
            [False, True, False],
        )
        self.assertTrue(audit["fixed_performance_restore_attempted"])
        self.assertTrue(audit["fixed_performance_restore_valid"])

    def test_single_method_matrix_allows_missing_peer_token_flag(self):
        rows = [{
            "trace": "trace.csv",
            "method": "offline-mixed",
            "status": "ok",
            "rc": "0",
            "valid_token_trace": "True",
            "token_trace_timing_contract_valid": "True",
            "decode_wall_contract_valid": "True",
            "decode_phase_contract_valid": "True",
            "valid_token_sequence": "",
            "valid_measurement_isolation": "True",
            "measurement_monitor_samples": "10",
            "measurement_monitor_failures": "0",
            "measurement_llama_pid": "123",
        }]
        self.assertEqual(
            matrix_completion_failures(
                rows,
                trace_names={"trace.csv"},
                methods={"offline-mixed"},
                require_device_idle=True,
            ),
            [],
        )

    @mock.patch(
        "runtime.plan.run_dynamic_budget_matrix.adb_shell_retry")
    def test_remote_plan_directory_requires_every_exact_file(
            self, adb_retry):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "index.json").write_text(
                '{"index":[]}\n', encoding="utf-8")
            (root / "plan_1MiB.json").write_text(
                '{"budget_mib":1}\n', encoding="utf-8")
            manifest = directory_file_manifest(root)
            adb_retry.return_value = CompletedProcess(
                args=[], returncode=0,
                stdout="".join(
                    f"{sha}\t{size}\t{path}\n"
                    for path, (size, sha) in manifest.items()
                ),
            )
            self.assertEqual(
                validate_remote_directory_contents(
                    "serial",
                    root,
                    "/remote/table",
                    label="plan table",
                    timeout_s=30,
                    retries=2,
                ),
                directory_sha256(root),
            )
            adb_retry.return_value = CompletedProcess(
                args=[], returncode=0,
                stdout="",
            )
            with self.assertRaisesRegex(
                    RuntimeError, "remote plan table content mismatch"):
                validate_remote_directory_contents(
                    "serial",
                    root,
                    "/remote/table",
                    label="plan table",
                    timeout_s=30,
                    retries=2,
                )

    def test_isolation_monitor_fails_closed_without_a_sample(self):
        class Args:
            execution_backend = "gpu"
            cpu_mask = ""
            device_idle_poll_s = 1.0
            require_device_idle = True

        class AlreadyStopped:
            @staticmethod
            def wait(_timeout):
                return True

        result = {}
        monitor_measurement_overlap(Args(), AlreadyStopped(), result)
        self.assertEqual(result["measurement_monitor_samples"], 0)
        self.assertFalse(result["valid_measurement_isolation"])

    @mock.patch(
        "runtime.plan.run_dynamic_budget_matrix.adb_shell")
    def test_process_isolation_ignores_shell_wrapper_not_real_llama(
            self, adb_shell):
        adb_shell.return_value = CompletedProcess(
            args=[], returncode=0,
            stdout=(
                "shell 100 1 sh -c cd /data/local/tmp && ./llama-cli -m m\n"
                "shell 101 100 /data/local/tmp/llama-cli -m m\n"
                "shell 102 1 sha256sum /tmp/llama-cli\n"
                "shell 103 1 /data/local/tmp/granularity-pipeline-bench\n"
            ),
        )
        processes = external_inference_processes(
            "serial", 30.0, quiet=True)
        self.assertEqual(
            processes,
            [
                "shell 101 100 /data/local/tmp/llama-cli -m m",
                "shell 103 1 /data/local/tmp/granularity-pipeline-bench",
            ],
        )

    @mock.patch(
        "runtime.plan.run_dynamic_budget_matrix.adb_shell")
    def test_process_isolation_fails_closed_on_adb_error(self, adb_shell):
        adb_shell.return_value = CompletedProcess(
            args=[], returncode=1, stdout="device offline")
        with self.assertRaisesRegex(
                RuntimeError, "failed to sample inference processes"):
            external_inference_processes(
                "serial", 30.0, quiet=True)

    def test_plan_table_hash_depends_on_paths_and_bytes_not_mtime(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "index.json").write_text(
                '{"index":[]}\n', encoding="utf-8")
            (root / "plan_1MiB.json").write_text(
                '{"budget_mib":1}\n', encoding="utf-8")
            first = directory_sha256(root)
            (root / "index.json").touch()
            self.assertEqual(directory_sha256(root), first)
            (root / "plan_1MiB.json").write_text(
                '{"budget_mib":2}\n', encoding="utf-8")
            self.assertNotEqual(directory_sha256(root), first)

    @mock.patch(
        "runtime.plan.run_dynamic_budget_matrix.adb_shell_retry")
    def test_remote_sha256_contract_accepts_exact_bytes(self, adb_retry):
        expected = "a" * 64
        adb_retry.return_value = CompletedProcess(
            args=[], returncode=0,
            stdout=f"{expected}  /data/local/tmp/model.gguf\n",
        )
        self.assertEqual(
            validate_remote_sha256(
                "serial",
                "/data/local/tmp/model.gguf",
                expected.upper(),
                label="model",
                timeout_s=30,
                retries=2,
            ),
            expected,
        )

    @mock.patch(
        "runtime.plan.run_dynamic_budget_matrix.adb_shell_retry")
    def test_remote_sha256_contract_rejects_stale_input(self, adb_retry):
        adb_retry.return_value = CompletedProcess(
            args=[], returncode=0,
            stdout=f"{'b' * 64}  /data/local/tmp/llama-cli\n",
        )
        with self.assertRaisesRegex(
                RuntimeError, "remote llama-cli SHA-256 mismatch"):
            validate_remote_sha256(
                "serial",
                "/data/local/tmp/llama-cli",
                "a" * 64,
                label="llama-cli",
                timeout_s=30,
                retries=2,
            )

    def test_effective_granularity_policy_names_real_method(self):
        self.assertEqual(
            effective_granularity_policy("offline-mixed", "diff-tree"),
            "offline",
        )
        self.assertEqual(
            effective_granularity_policy("online", "diff-tree"),
            "online",
        )
        self.assertEqual(
            effective_granularity_policy(
                "diff-tree-mixed", "diff-tree"),
            "diff-tree",
        )
        self.assertEqual(
            effective_granularity_policy("mru", "diff-tree"),
            "fixed-tensor",
        )

    def test_remote_server_policy_is_method_specific_in_joint_matrix(self):
        self.assertEqual(
            remote_server_granularity_policy("online", "diff-tree"),
            "online",
        )
        self.assertEqual(
            remote_server_granularity_policy(
                "diff-tree-mixed", "diff-tree"),
            "diff-tree",
        )
        with self.assertRaisesRegex(ValueError, "not used"):
            remote_server_granularity_policy("offline-mixed", "diff-tree")

    def test_profile_kind_uses_residual_curve_content(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "profile.json"
            path.write_text(
                '{"profiles":{"cpu":{"mode_pipeline_efficiency_curve":{'
                '"cut":[{"resident_ratio":0.3,"residual_ms":1.0}]}}}}',
                encoding="utf-8",
            )
            self.assertEqual(
                granularity_profile_kind(path, "cpu"),
                "pipeline-residual",
            )
            self.assertEqual(
                granularity_profile_kind(path, "gpu"),
                "phase-only",
            )

    def test_profile_declares_one_supported_multi_implementation(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "profile.json"
            path.write_text(
                '{"profiles":{"cpu":{"multi_implementation":'
                '"multi_fused"},"gpu":{"multi_implementation":"multi"}}}',
                encoding="utf-8",
            )
            self.assertEqual(
                granularity_profile_multi_implementation(path, "cpu"),
                "multi_fused",
            )
            self.assertEqual(
                granularity_profile_multi_implementation(path, "gpu"),
                "multi",
            )
            path.write_text(
                '{"profiles":{"gpu":{"multi_implementation":'
                '"multi_fused"}}}',
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "unsupported"):
                granularity_profile_multi_implementation(path, "gpu")
            path.write_text(
                '{"profiles":{"gpu":{"cut_compute_grouping":'
                '"fused_pair"}}}',
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                    ValueError, "explicitly encode"):
                granularity_profile_multi_implementation(path, "gpu")

    def test_profile_cut_compute_grouping_matches_backend(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "profile.json"
            path.write_text(
                '{"profiles":{"cpu":{"cut_compute_grouping":'
                '"independent_halves"},"gpu":{"cut_compute_grouping":'
                '"fused_pair"}}}',
                encoding="utf-8",
            )
            self.assertEqual(
                granularity_profile_cut_compute_grouping(path, "cpu"),
                "independent_halves",
            )
            self.assertEqual(
                granularity_profile_cut_compute_grouping(path, "gpu"),
                "fused_pair",
            )
            path.write_text(
                '{"profiles":{"gpu":{"cut_compute_grouping":'
                '"independent_halves"}}}',
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "formal runtime"):
                granularity_profile_cut_compute_grouping(path, "gpu")
            path.write_text(
                '{"profiles":{"gpu":{"multi_implementation":"multi"}}}',
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                    ValueError, "explicitly encode"):
                granularity_profile_cut_compute_grouping(path, "gpu")

    def test_profile_transition_contract_is_content_addressed(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "profile.json"
            path.write_text(
                '{"profiles":{"cpu":{"transition_fixed_ms":0.75,'
                '"transition_cost_source":"held-in frontier publish"}}}',
                encoding="utf-8",
            )
            self.assertEqual(
                granularity_profile_transition_contract(path, "cpu"),
                (0.75, "held-in frontier publish"),
            )

    def test_remote_solver_log_is_method_and_run_isolated(self):
        with tempfile.TemporaryDirectory() as temporary:
            log = Path(temporary) / "solver.log"
            log.write_text(
                '{"event":"solve","run_id":"/work/online",'
                '"working_unit_policy":"online-global-mixed",'
                '"online_placement_source":"stateful-runtime-cp"}\n'
                '{"event":"solve-cache","run_id":"/work/online",'
                '"working_unit_policy":"online-global-mixed"}\n',
                encoding="utf-8",
            )
            failures, audit = remote_solver_log_contract(
                "online", log, "/work/online", None, "stateful-cp")
            self.assertEqual(failures, [])
            self.assertEqual(audit["remote_solver_event_count"], 2)
            self.assertEqual(
                audit["remote_solver_placement_sources"],
                "stateful-runtime-cp",
            )
            failures, _ = remote_solver_log_contract(
                "online", log, "/work/online", None, "offline-table")
            self.assertIn(
                "remote solver placement source is not "
                "precomputed-offline-budget-table",
                failures,
            )
            failures, _ = remote_solver_log_contract(
                "diff-tree-mixed", log, "/work/diff")
            self.assertTrue(any(
                "run id" in failure for failure in failures))
            self.assertTrue(any(
                "diff-tree" in failure for failure in failures))
            self.assertEqual(
                remote_solver_log_contract(
                    "offline-mixed", None, "/work/offline")[0],
                [],
            )

    def test_remote_diff_solver_proves_edit_bound(self):
        with tempfile.TemporaryDirectory() as temporary:
            log = Path(temporary) / "solver.log"
            log.write_text(
                '{"event":"solve","run_id":"/work/diff",'
                '"working_unit_policy":"diff-tree-mixed",'
                '"online_placement_source":"stateful-runtime-cp",'
                '"edit_count":16}\n',
                encoding="utf-8",
            )
            failures, audit = remote_solver_log_contract(
                "diff-tree-mixed", log, "/work/diff", 16)
            self.assertEqual(failures, [])
            self.assertEqual(
                audit["remote_solver_diff_edit_count_max"], 16)
            failures, _ = remote_solver_log_contract(
                "diff-tree-mixed", log, "/work/diff", 15)
            self.assertIn(
                "Diff-tree solver exceeded configured edit bound",
                failures,
            )

    def test_effective_granularity_placement_source_is_method_specific(self):
        self.assertEqual(
            effective_granularity_placement_source(
                "offline-mixed", "stateful-cp"),
            "offline-table",
        )
        self.assertEqual(
            effective_granularity_placement_source(
                "online", "stateful-cp"),
            "stateful-cp",
        )
        self.assertEqual(
            effective_granularity_placement_source(
                "diff-tree-mixed", "stateful-cp"),
            "stateful-cp",
        )
        self.assertEqual(
            effective_granularity_placement_source(
                "mru", "stateful-cp"),
            "fixed-tensor",
        )

    def test_gpu_cut_frontier_requires_fused_dual_compute(self):
        good = {
            "frontier_cut_units_max": 12,
            "physical_cut_tensors": 7,
            "physical_cut_parts": 14,
            "cut_dual_candidates": 4,
            "cut_dual_calls": 4,
            "cut_dual_budget_fallbacks": 0,
            "cut_dual_queue_errors": 0,
            "cut_dual_image_creates": 8,
            "cut_dual_image_errors": 0,
        }
        self.assertEqual(
            backend_compute_contract_failures("gpu", good),
            [],
        )
        no_fused = dict(good, cut_dual_calls=0)
        self.assertIn(
            "tiled weights did not execute fused dual GEMV",
            backend_compute_contract_failures("gpu", no_fused),
        )
        budget_fallback = dict(good, cut_dual_budget_fallbacks=1)
        self.assertTrue(any(
            "did not fit the active weight budget" in reason
            for reason in backend_compute_contract_failures(
                "gpu", budget_fallback)
        ))
        self.assertEqual(
            backend_compute_contract_failures("cpu", no_fused),
            [],
        )
        no_cut_frontier = dict(good, frontier_cut_units_max=0)
        self.assertEqual(
            backend_compute_contract_failures("gpu", no_cut_frontier),
            [],
        )
        no_cut_frontier["cut_dual_calls"] = 0
        self.assertIn(
            "tiled weights did not execute fused dual GEMV",
            backend_compute_contract_failures("gpu", no_cut_frontier),
        )

    def test_cooldown_snapshot_requires_real_thermal_signal(self):
        class Args:
            cooldown_thermal_max_c = 35
            cooldown_thermal_status_max = 0
            min_battery_level_pct = 20

        self.assertFalse(cooldown_snapshot_ready(
            Args(), {"battery_level_pct": 90}))
        self.assertTrue(cooldown_snapshot_ready(Args(), {
            "thermal_status": 0,
            "thermal_cpu_max_c": 34.5,
            "thermal_gpu_max_c": "",
            "thermal_skin_max_c": "",
            "battery_level_pct": 90,
        }))
        self.assertFalse(cooldown_snapshot_ready(Args(), {
            "thermal_status": 0,
            "thermal_cpu_max_c": 35.5,
            "battery_level_pct": 90,
        }))

    def test_resume_requires_current_environment_contract(self):
        class Args:
            require_device_idle = True
            cooldown_thermal_max_c = 35
            cooldown_thermal_status_max = 0
            min_battery_level_pct = 20
            execution_backend = "cpu"
            min_cpu_freq_limit_khz = 1400000
            min_cpu_mean_freq_khz = 1550000
            min_cpu_median_sample_min_khz = 1500000

        good = {
            "require_device_idle_configured": "True",
            "cooldown_thermal_max_c_configured": "35",
            "cooldown_thermal_status_max_configured": "0",
            "min_battery_level_pct_configured": "20",
            "thermal_status_before": "0",
            "thermal_cpu_max_c_before": "34",
            "thermal_gpu_max_c_before": "",
            "thermal_skin_max_c_before": "",
            "battery_level_pct_before": "90",
            "cpu_freq_limit_min_khz": "1824000",
            "cpu_freq_mean_khz": "1600000",
            "cpu_freq_median_sample_min_khz": "1593600",
        }
        self.assertEqual(
            resume_environment_contract_failures(Args(), good),
            [],
        )
        del good["thermal_cpu_max_c_before"]
        self.assertIn(
            "recorded clean-start snapshot invalid",
            resume_environment_contract_failures(Args(), good),
        )

    def test_cpu_frequency_contract_fails_closed(self):
        thresholds = {
            "min_limit_khz": 1400000,
            "min_mean_khz": 1550000,
            "min_median_sample_min_khz": 1500000,
        }
        self.assertEqual(
            cpu_frequency_contract_failures({
                "cpu_freq_limit_min_khz": 1824000,
                "cpu_freq_mean_khz": 1600000,
                "cpu_freq_median_sample_min_khz": 1593600,
            }, **thresholds),
            [],
        )
        missing = cpu_frequency_contract_failures(
            {"cpu_freq_limit_min_khz": 1824000},
            **thresholds,
        )
        self.assertEqual(len(missing), 2)
        self.assertTrue(all("missing" in item for item in missing))
        low = cpu_frequency_contract_failures({
            "cpu_freq_limit_min_khz": 1300000,
            "cpu_freq_mean_khz": 1600000,
            "cpu_freq_median_sample_min_khz": 1593600,
        }, **thresholds)
        self.assertEqual(
            low,
            ["cpu_freq_limit_min_khz=1300000<1400000"],
        )

    def test_timed_run_requires_duration_exit(self):
        self.assertEqual(timed_run_contract_failures(
            {
                "bench_exit_reason": "duration",
                "bench_time_done": 1,
                "budget_decode_reset_count": 1,
                "budget_decode_reset_generated": 0,
                "budget_backend_reset_count": 1,
            },
            600.0,
        ), [])
        failures = timed_run_contract_failures(
            {
                "bench_exit_reason": "loop_condition",
                "bench_time_done": 0,
            },
            600.0,
        )
        self.assertEqual(len(failures), 4)
        self.assertEqual(timed_run_contract_failures({}, None), [])

    def test_decode_wall_timer_covers_every_eval_component(self):
        parsed = {
            "eval_runs": 3,
            "eval_ms_total": 300.0,
            "provider_get_ms_total": 6.0,
            "apply_ms_total": 3.0,
            "decode_wall_runs": 3,
            "decode_wall_ms_total": 315.0,
            "exec_timing_source": "decode-wall",
        }
        self.assertEqual(decode_wall_contract_failures(parsed), [])
        parsed["decode_wall_runs"] = 2
        self.assertIn(
            "decode wall runs=2, eval runs=3",
            decode_wall_contract_failures(parsed),
        )
        parsed["decode_wall_runs"] = 3
        parsed["decode_wall_ms_total"] = 290.0
        self.assertTrue(any(
            "shorter than contained components" in failure
            for failure in decode_wall_contract_failures(parsed)
        ))

    def test_decode_phase_contract_is_prompt_excluding(self):
        parsed = {
            "decode_phase_origin_count": 1,
            "decode_phase_quiesce_count": 2,
            "decode_phase_quiesce_sequence": "begin,end",
            "decode_phase_quiesce_begin_rc": 0,
            "decode_phase_quiesce_end_rc": 0,
            "decode_phase_origin_generated": 0,
            "decode_phase_origin_backend": "cpu",
            "decode_phase_summary_count": 1,
            "decode_phase_backend": "cpu",
            "decode_phase_runs": 3,
            "eval_runs": 3,
            "decode_wall_runs": 3,
            "decode_phase_units": 30,
            "decode_phase_nonresident_units": 12,
            "decode_phase_reloads": 0,
            "decode_phase_reload_bytes": 0,
            "decode_phase_direct_read_calls": 0,
            "decode_phase_direct_read_ms": 0.0,
            "decode_phase_direct_read_mib": 0.0,
            "decode_phase_load_calls": 0,
            "decode_phase_load_ms": 0.0,
            "decode_phase_load_mib": 0.0,
            "decode_phase_prepare_calls": 0,
            "decode_phase_prepare_ms": 0.0,
            "decode_phase_prepare_mib": 0.0,
            "decode_phase_pipeline_waits": 0,
            "decode_phase_pipeline_wait_ms": 0.0,
            "decode_phase_pipeline_residency_ms": 0.0,
            "decode_phase_pipeline_unissued_ms": 0.0,
            "decode_phase_pipeline_stage_ms": 0.0,
            "decode_phase_pipeline_retire_ms": 0.0,
            "decode_phase_evict_ms": 0.0,
            "decode_phase_resident_mib": 1024.0,
            "decode_phase_counter_scope":
                "decode-only-backend-counter-delta",
            "decode_phase_compute_available": 1,
            "decode_phase_compute_calls": 3,
            "decode_phase_compute_ms": 90.0,
        }
        self.assertEqual(
            decode_phase_contract_failures(parsed, "cpu"), [])
        parsed["decode_phase_quiesce_count"] = 1
        self.assertIn(
            "decode phase quiescence boundary count is not two",
            decode_phase_contract_failures(parsed, "cpu"),
        )
        parsed["decode_phase_quiesce_count"] = 2
        parsed["decode_phase_runs"] = 4
        self.assertTrue(any(
            "decode phase runs=4" in failure
            for failure in decode_phase_contract_failures(parsed, "cpu")
        ))

    def test_token_trace_timing_covers_duration(self):
        timing = {
            "token_trace_decode_rows": 20,
            "token_trace_t_first_sec": 0.8,
            "token_trace_t_last_sec": 599.4,
            "token_trace_first_latency_ms": 700.0,
            "token_trace_max_latency_ms": 900.0,
            "token_trace_time_monotonic": True,
            "token_trace_budget_provider_valid": True,
        }
        self.assertEqual(
            token_trace_timing_contract_failures(timing, 600.0), [])
        timing["token_trace_t_last_sec"] = 500.0
        self.assertIn(
            "does not cover",
            token_trace_timing_contract_failures(
                timing, 600.0)[0])

    def test_mixed_frontier_contract_requires_real_mixed_execution(self):
        good = {
            "frontier_apply_count": 4,
            "frontier_multi_units_max": 8,
            "frontier_tensor_units_max": 0,
            "frontier_cut_units_max": 12,
            "frontier_mixed_mode_generations": 2,
            "online_calls": 4,
        }
        self.assertEqual(
            frontier_contract_failures("offline-mixed", good), [])
        self.assertEqual(
            frontier_contract_failures("online", good), [])
        self.assertEqual(
            frontier_contract_failures("diff-tree-mixed", good), [])

        no_frontier = dict(good, frontier_apply_count=0)
        self.assertIn(
            "working-unit frontier was not applied",
            frontier_contract_failures("offline-mixed", no_frontier))
        one_mode = dict(
            good,
            frontier_tensor_units_max=0,
            frontier_cut_units_max=0,
        )
        self.assertIn(
            "mixed-granularity frontier was not exercised",
            frontier_contract_failures("diff-tree-mixed", one_mode))
        no_simultaneous_mix = dict(
            good, frontier_mixed_mode_generations=0)
        self.assertIn(
            "no applied frontier simultaneously used multiple modes",
            frontier_contract_failures(
                "diff-tree-mixed", no_simultaneous_mix))
        no_online = dict(good, online_calls=0)
        self.assertIn(
            "online planner was not exercised",
            frontier_contract_failures("online", no_online))

    def test_non_mixed_baseline_does_not_require_frontier(self):
        self.assertEqual(frontier_contract_failures("mru", {}), [])

    def test_unit_pipeline_is_the_only_recurring_stage_authority(self):
        parsed = {
            "plan_stage_defer_apply_count": 4,
            "plan_stage_defer_min": 1,
            "plan_stage_defer_max": 1,
            "plan_anchor_fired": 0,
            "plan_anchor_load_events": 0,
            "plan_anchor_transfer_events": 0,
            "plan_anchor_xform_events": 0,
        }
        self.assertEqual(
            unit_pipeline_stage_authority_failures(parsed), [])
        parsed["plan_stage_defer_min"] = 0
        self.assertIn(
            "whole-weight plan stages were issued eagerly",
            unit_pipeline_stage_authority_failures(parsed),
        )
        parsed["plan_stage_defer_min"] = 1
        parsed["plan_anchor_load_events"] = 2
        self.assertIn(
            "plan LOAD anchor fired alongside the physical unit pipeline",
            unit_pipeline_stage_authority_failures(parsed),
        )

    def test_unit_pipeline_authority_disables_weight_stage_anchors(self):
        env = {
            "LLAMA_ELASTIC_USE_INTERVAL_SCHEDULE": "1",
            "LLAMA_ELASTIC_DEFER_STAGE": "0",
            "LLAMA_ELASTIC_INTERVAL_STAGE_KINDS": "load,prepare",
        }
        configure_unit_pipeline_stage_authority(env)
        self.assertEqual(
            env["LLAMA_ELASTIC_USE_INTERVAL_SCHEDULE"], "0")
        self.assertEqual(env["LLAMA_ELASTIC_DEFER_STAGE"], "1")
        self.assertEqual(
            env["LLAMA_ELASTIC_INTERVAL_STAGE_KINDS"], "none")

    def test_dynamic_log_parses_unit_pipeline_stage_authority(self):
        with tempfile.TemporaryDirectory() as temporary:
            log = Path(temporary) / "authority.log"
            log.write_text(
                "apply_exec_plan: applied plan budget=4992MiB "
                "stage_defer=1\n"
                "apply_exec_plan: applied plan budget=4864MiB "
                "stage_defer=1\n"
                "~llama_context: elastic anchor summary: requests=200 "
                "hits=0 duplicate=0 fired=0 load=0 transfer=0 xform=0 "
                "failures=0\n"
                "__LLAMA_INNER_RC__=0\n",
                encoding="utf-8",
            )
            parsed = parse_log(log, "diff-tree-mixed")
            self.assertEqual(
                parsed["plan_stage_defer_apply_count"], 2)
            self.assertEqual(parsed["plan_stage_defer_min"], 1)
            self.assertEqual(parsed["plan_stage_defer_max"], 1)
            self.assertEqual(parsed["plan_anchor_fired"], 0)
            self.assertEqual(parsed["plan_anchor_load_events"], 0)
            self.assertEqual(parsed["plan_anchor_xform_events"], 0)

    def test_pin_policy_is_exact_and_all_is_explicit(self):
        self.assertTrue(
            pin_policy_requests("norm,token_embd,output", "output"))
        self.assertTrue(pin_policy_requests("all", "token_embd"))
        self.assertFalse(pin_policy_requests("attn_output", "output"))

    def test_dynamic_log_parses_inside_and_outside_budget_pins(self):
        with tempfile.TemporaryDirectory() as temporary:
            log = Path(temporary) / "pin.log"
            log.write_text(
                "elastic: pin token_embd inside weight budget "
                "(281.81 MiB, 1 physical unit)\n"
                "ggml_opencl elastic: pin output inside weight budget "
                "(205.49 MiB)\n"
                "ggml_opencl elastic: pin output inside weight budget "
                "(205.49 MiB)\n"
                "ggml_opencl elastic: pin unplanned output (410 MB) "
                "outside budget -> target=3278 MB\n"
                "__LLAMA_INNER_RC__=0\n",
                encoding="utf-8",
            )
            parsed = parse_log(log, "offline-mixed")
            self.assertEqual(
                parsed["pin_token_embd_inside_budget_count"], 1)
            self.assertAlmostEqual(
                parsed["pin_token_embd_inside_budget_mib"], 281.81)
            self.assertEqual(parsed["pin_output_inside_budget_count"], 2)
            self.assertAlmostEqual(
                parsed["pin_output_inside_budget_mib"], 410.98)
            self.assertEqual(parsed["pin_outside_budget_count"], 1)

    def test_failed_runtime_plan_apply_invalidates_row(self):
        with tempfile.TemporaryDirectory() as temporary:
            log = Path(temporary) / "failed-apply.log"
            log.write_text(
                "maybe_apply_plan: apply_exec_plan failed "
                "budget=3712MiB plan_budget=3712MiB rc=-2; "
                "retaining previous plan\n"
                "__LLAMA_INNER_RC__=0\n",
                encoding="utf-8",
            )
            self.assertEqual(
                parse_log(log, "diff-tree-mixed")["status"],
                "check_log",
            )

    def test_dynamic_log_parses_gpu_tiled_image_cache(self):
        with tempfile.TemporaryDirectory() as temporary:
            log = Path(temporary) / "gpu.log"
            log.write_text(
                "[elastic cut dual] enabled=1 candidates=20 calls=18 "
                "budget_fallbacks=1 shape_fallbacks=1 queue_errors=0 "
                "image_creates=12 image_cache_hits=24 image_releases=8 "
                "image_errors=0\n"
                "__LLAMA_INNER_RC__=0\n",
                encoding="utf-8",
            )
            parsed = parse_log(log, "online")
            self.assertEqual(parsed["cut_dual_calls"], 18)
            self.assertEqual(parsed["cut_dual_image_creates"], 12)
            self.assertEqual(parsed["cut_dual_image_cache_hits"], 24)
            self.assertEqual(parsed["cut_dual_image_releases"], 8)
            self.assertEqual(parsed["cut_dual_image_errors"], 0)

    def test_dynamic_log_parses_common_physical_tiles(self):
        with tempfile.TemporaryDirectory() as temporary:
            log = Path(temporary) / "physical.log"
            log.write_text(
                "[elastic granularity] backend=cpu mode=tensor units=10 "
                "cut_ops=0 fallback=0 peak_unit=2.00 MiB "
                "cut_tensors=7 cut_parts=14 wbm_total=4436.79 MiB\n"
                "__LLAMA_INNER_RC__=0\n",
                encoding="utf-8",
            )
            parsed = parse_log(log, "offline-mixed")
            self.assertEqual(parsed["physical_cut_tensors"], 7)
            self.assertEqual(parsed["physical_cut_parts"], 14)
            self.assertEqual(parsed["wbm_total_mib"], 4436.79)

    def test_dynamic_log_parses_cpu_profile_physical_tiles(self):
        with tempfile.TemporaryDirectory() as temporary:
            log = Path(temporary) / "physical-cpu.log"
            log.write_text(
                "  granularity  : mode=tensor units=10 cut_ops=0 "
                "fallback_ops=0 peak_unit=2.00 MiB tensors=225 "
                "cut_tensors=7 parts=14 wbm_total=4436.79 MiB\n"
                "__LLAMA_INNER_RC__=0\n",
                encoding="utf-8",
            )
            parsed = parse_log(log, "offline-mixed")
            self.assertEqual(parsed["physical_cut_tensors"], 7)
            self.assertEqual(parsed["physical_cut_parts"], 14)
            self.assertEqual(parsed["wbm_total_mib"], 4436.79)

    def test_backend_contract_rejects_noncommon_physical_tiles(self):
        failures = backend_compute_contract_failures(
            "cpu",
            {"physical_cut_tensors": 7, "physical_cut_parts": 7},
        )
        self.assertTrue(any("exactly two tiles" in item
                            for item in failures))

    def test_backend_contract_rejects_wrong_physical_weight_total(self):
        failures = backend_compute_contract_failures(
            "cpu",
            {
                "physical_cut_tensors": 7,
                "physical_cut_parts": 14,
                "wbm_total_mib": 4000,
            },
            4436.7890625,
        )
        self.assertTrue(any("WBM total" in item for item in failures))

    def test_cpu_multi_fused_profile_requires_real_fused_kernel(self):
        parsed = {
            "physical_cut_tensors": 7,
            "physical_cut_parts": 14,
            "wbm_total_mib": 4436.79,
            "frontier_multi_units_max": 4,
            "fused_kernel_pair_candidates": 4,
            "fused_kernel_pair_calls": 4,
            "fused_kernel_pair_errors": 0,
        }
        self.assertEqual(
            backend_compute_contract_failures(
                "cpu", parsed, 4436.7890625, True),
            [],
        )
        parsed["fused_kernel_pair_calls"] = 0
        self.assertIn(
            "Multi-fused frontier did not execute fused MUL_MAT",
            backend_compute_contract_failures(
                "cpu", parsed, 4436.7890625, True),
        )

    def test_reuse_stream_preserves_plan_residents_without_graph_cold_flush(
            self):
        env = {
            "GGML_ELASTIC_NO_AUTO_EVICT": "1",
            "LLAMA_ELASTIC_FORCE_PLAN_EVICT": "1",
            "LLAMA_ELASTIC_EVICT_DISK_WEIGHTS_PER_GRAPH": "1",
        }
        apply_plan_residency_policy(
            env, "diff-tree-mixed", "reuse-stream")
        self.assertEqual(env["GGML_ELASTIC_NO_AUTO_EVICT"], "1")
        self.assertEqual(env["LLAMA_ELASTIC_FORCE_PLAN_EVICT"], "1")
        self.assertEqual(
            env["LLAMA_ELASTIC_EVICT_DISK_WEIGHTS_PER_GRAPH"], "0")
        self.assertEqual(
            env["LLAMA_ELASTIC_PLAN_RESIDENCY_POLICY"], "reuse-stream")

    def test_cache_managed_keeps_baseline_mru_policy_untouched(self):
        env = {
            "GGML_ELASTIC_NO_AUTO_EVICT": "0",
            "LLAMA_ELASTIC_EVICT_DISK_WEIGHTS_PER_GRAPH": "0",
        }
        apply_plan_residency_policy(env, "mru", "cache-managed")
        self.assertEqual(env["GGML_ELASTIC_NO_AUTO_EVICT"], "0")
        self.assertEqual(
            env["LLAMA_ELASTIC_EVICT_DISK_WEIGHTS_PER_GRAPH"], "0")
        self.assertNotIn("LLAMA_ELASTIC_FORCE_PLAN_EVICT", env)
        self.assertEqual(
            env["LLAMA_ELASTIC_PLAN_RESIDENCY_POLICY"], "cache-managed")

    def test_summary_formats_unavailable_gpu_compute_without_crashing(self):
        self.assertEqual(format_metric(""), "N/A")
        self.assertEqual(format_metric(None), "N/A")
        self.assertEqual(format_metric(12.3456), "12.346")

    @mock.patch("runtime.plan.run_dynamic_budget_matrix.adb_shell")
    def test_device_snapshot_includes_physical_battery_level(self, adb_shell):
        adb_shell.return_value = CompletedProcess(
            args=[], returncode=0,
            stdout=(
                "Thermal Status: 0\n"
                "Temperature{mValue=41.5, mType=0, mName=CPU7, mStatus=0}\n"
                "__ELASTIC_BATTERY__\n"
                "Current Battery Service state:\n"
                "  level: 37\n"
                "__ELASTIC_POWER__\n"
                "  mWakefulness=Asleep\n"))
        snapshot = read_thermal_snapshot("serial", 30)
        self.assertEqual(snapshot["thermal_cpu_max_c"], 41.5)
        self.assertEqual(snapshot["battery_level_pct"], 37)
        self.assertEqual(snapshot["device_wakefulness"], "Asleep")

    @mock.patch("runtime.plan.run_dynamic_budget_matrix.adb_shell")
    def test_device_snapshot_prefers_current_hal_over_stale_cache(
            self, adb_shell):
        adb_shell.return_value = CompletedProcess(
            args=[], returncode=0,
            stdout=(
                "Thermal Status: 0\n"
                "Cached temperatures:\n"
                "  Temperature{mValue=89.9, mType=0, "
                "mName=CPU7, mStatus=0}\n"
                "  Temperature{mValue=43.4, mType=3, "
                "mName=skin, mStatus=0}\n"
                "  Temperature{mValue=36.0, mType=3, "
                "mName=shell_skin, mStatus=0}\n"
                "Current temperatures from HAL:\n"
                "  Temperature{mValue=31.4, mType=0, "
                "mName=CPU7, mStatus=0}\n"
                "  Temperature{mValue=31.9, mType=1, "
                "mName=GPU0, mStatus=0}\n"
                "  Temperature{mValue=29.7, mType=3, "
                "mName=skin, mStatus=0}\n"
                "Current cooling devices from HAL:\n"
                "  CoolingDevice{mValue=0, mName=gpu}\n"
                "__ELASTIC_BATTERY__\n"
                "  level: 95\n"
                "__ELASTIC_POWER__\n"
                "  mWakefulness=Asleep\n"))
        snapshot = read_thermal_snapshot("serial", 30)
        self.assertEqual(snapshot["thermal_cpu_max_c"], 31.4)
        self.assertEqual(snapshot["thermal_gpu_max_c"], 31.9)
        self.assertEqual(snapshot["thermal_skin_max_c"], 29.7)

    def test_backends_do_not_double_count_inside_budget_pin_reserve(self):
        class Args:
            execution_backend = "cpu"
            safety_mib = 283
            pinned_extra_mib = 282

        self.assertEqual(backend_safety_mib(Args()), 283)
        Args.execution_backend = "gpu"
        self.assertEqual(backend_safety_mib(Args()), 283)

    def test_stream_reserve_reduces_only_planner_residency(self):
        class Args:
            budget_mib = 4096
            kv_mib = 512
            misc_mib = 256
            safety_mib = 565
            planner_stream_reserve_mib = 16

        self.assertEqual(
            planner_resident_budget_bytes(Args()),
            (4096 - 512 - 256 - 565 - 16) * 1024 * 1024)

    def test_fully_resident_plan_does_not_require_idle_worker_affinity(self):
        parsed = {
            "load_planned": 0,
            "direct_read_calls": 0,
            "reload_calls": 0,
            "xform_planned": 0,
            "stage_xform_calls": 0,
        }
        self.assertFalse(pipeline_worker_was_required(parsed, "load"))
        self.assertFalse(pipeline_worker_was_required(parsed, "prepare"))
        parsed["direct_read_calls"] = 1
        parsed["stage_xform_calls"] = 1
        self.assertTrue(pipeline_worker_was_required(parsed, "load"))
        self.assertTrue(pipeline_worker_was_required(parsed, "prepare"))
        self.assertFalse(pipeline_worker_was_required(parsed, "copy"))
        parsed["fused_layout_parallel_calls"] = 2
        self.assertTrue(pipeline_worker_was_required(parsed, "copy"))

    def test_cpu_stage_fallback_uses_calibrated_layout_model(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "calibration.json").write_text(
                '{"stage_fallback_models": {'
                '"CPU_Elastic:LOAD": {"fixed_ms": 0.2, '
                '"ms_per_mib": 0.5}, '
                '"CPU_Elastic:XFORM": {"fixed_ms": 0.3, '
                '"bandwidth_mib_s": 250.0}}}',
                encoding="utf-8")
            model = CostModel(root)
            self.assertAlmostEqual(
                model.stage_ms(
                    "CPU_Elastic", "LOAD", "weight", 2 * 1024 * 1024),
                1.2)
            self.assertAlmostEqual(
                model.stage_ms(
                    "CPU_Elastic", "XFORM", "weight", 2 * 1024 * 1024),
                8.3)
            self.assertTrue(
                model.has_measured_stage_backend("CPU_Elastic"))
            self.assertTrue(
                model.has_measured_stage_kind("CPU_Elastic", "XFORM"))

    def test_current_cpu_profile_dump_is_parsed_per_forward(self):
        with tempfile.TemporaryDirectory() as temporary:
            log = Path(temporary) / "run.log"
            log.write_text(
                "llama_perf_context_print: prompt eval time = 100.0 ms / "
                "2 tokens (50.0 ms per token)\n"
                "llama_perf_context_print: eval time = 300.0 ms / 3 runs "
                "(100.0 ms per token)\n"
                "[elastic-bench] final exit reason=duration generated=4 "
                "n_past=6 n_remain=-1 time_done=1\n"
                "  reload_count : 8 (10485760 bytes total)\n"
                "  direct_read  : calls=8 ok=8 fail=0 total=40.0 ms "
                "MB=10.0 MB/s=250.0\n"
                "  evict_count  : 8 (10485760 bytes total) wall=3.0 ms\n"
                "[elastic unit pipeline] backend=cpu issued=8 ready=8 "
                "waits=2 wait_ms=4.5\n"
                "[elastic unit pipeline residency] total_ms=10.0 "
                "missing_units=2 missing_tensors=2 missing_mib=8.0 "
                "unissued_units=1 unissued_tensors=1 unissued_mib=2.0 "
                "unissued_ms=1.0 prepare_cap_declines=0 "
                "prepare_cap_tensors=0\n"
                "[elastic unit pipeline timing] stage_ms=3.0 "
                "retire_ms=2.0 evict_ms=1.0\n"
                "[elastic unit pipeline budget] samples=99 violations=0 "
                "plan_protection_relaxations=3 relaxed_mib=4.5 "
                "resident_peak_mib=2701.0 pinned_peak_mib=281.8 "
                "over_peak_mib=0.0\n"
                "elastic: LOAD worker affinity cpu=0 rc=0\n"
                "elastic: PREPARE worker affinity cpu=1 rc=0\n"
                "apply_exec_plan: working_unit mode=tensor cut_parts=2 "
                "multi_tensors=2 policy=diff-tree-mixed state_aware=1 "
                "predicted_ms=12.0 switch_ms=0.1 "
                "frontier_sig=0123456789abcdef mixed_units=4 "
                "multi_units=1 tensor_units=2 cut_units=1 tiles=7 "
                "publish_ms=0.750000 delta_weights=3 "
                "generation=3\n"
                "apply_exec_plan: applied frontier-only delta_weights=3 "
                "budget=3712MiB without graph invalidation\n"
                "apply_exec_plan: working_unit mode=tensor cut_parts=2 "
                "multi_tensors=2 policy=diff-tree-mixed state_aware=1 "
                "predicted_ms=11.0 switch_ms=0.2 "
                "frontier_sig=fedcba9876543210 mixed_units=4 "
                "multi_units=1 tensor_units=2 cut_units=1 tiles=7 "
                "publish_ms=0.250000 delta_weights=2 "
                "generation=4\n"
                "  granularity  : mode=tensor units=1234 cut_ops=56 "
                "fallback_ops=0 peak_unit=31.5 MiB\n"
                "  compute_pt   : 17.0 ms/graph fallback=0 "
                "total=68.0 ms calls=4 avg=17.0 ms\n"
                "  stage_xform  : calls=7 ok=7 total=6.5 ms MB=12.0\n"
                "  async_prepare: units=7/7 tensors=7/7 waits=3 "
                "wait_total=4.0 ms\n"
                "[elastic cpu layout] initial_repack_calls=2 "
                "initial_repack_ms=5.0 initial_repack_mib=6.0 "
                "repack_calls=7 repack_ok=7 repack_ms=6.0 "
                "repack_mib=11.0 materialize_calls=1 "
                "materialize_ms=0.5 materialize_mib=1.0\n"
                "[elastic cpu multi_fused] layout_pairs=4 ok=4 fallback=0 "
                "parallel=4 "
                "ms=2.0 mib=8.0 kernel_candidates=12 calls=10 "
                "fallback=2 errors=0 ms=1.0 workspace_alloc=0 reuse=0 "
                "bytes=0\n"
                "maybe_apply_plan: budget=3712MiB switched "
                "provider_get_ms=2.0 apply_ms=3.0\n"
                "llama_context::~llama_context: elastic plan timing summary: "
                "provider_calls=2 provider_get_ms=6.0 apply_calls=1 "
                "apply_ms=3.0\n"
                "llama_context::~llama_context: elastic decode wall summary: "
                "runs=3 wall_ms=315.0\n"
                "[elastic-bench] decode phase origin generated=0 "
                "backend=cpu\n"
                "[elastic-bench] decode phase quiesce "
                "boundary=begin rc=0\n"
                "[elastic-bench] decode phase quiesce "
                "boundary=end rc=0\n"
                "[elastic decode phase summary] backend=cpu runs=3 "
                "units=123 nonresident_units=24 "
                "reloads=8 reload_bytes=10485760 "
                "direct_read_calls=8 direct_read_us=40000 "
                "direct_read_bytes=10485760 load_calls=8 load_us=42000 "
                "load_bytes=10485760 prepare_calls=7 prepare_us=6500 "
                "prepare_bytes=12582912 compute_available=1 "
                "compute_calls=4 compute_us=68000 pipeline_waits=2 "
                "pipeline_wait_us=4500 pipeline_residency_us=10000 "
                "pipeline_unissued_us=1000 pipeline_stage_us=3000 "
                "pipeline_retire_us=2000 evict_us=1000 "
                "resident_bytes=2832203776\n"
                "__LLAMA_INNER_RC__=0\n",
                encoding="utf-8")
            parsed = parse_log(log, "static-min")
            self.assertEqual(parsed["status"], "ok")
            self.assertEqual(parsed["forward_runs"], 4)
            self.assertEqual(parsed["generated_tokens"], 4)
            self.assertEqual(parsed["bench_exit_reason"], "duration")
            self.assertEqual(parsed["bench_time_done"], 1)
            self.assertEqual(parsed["provider_get_calls"], 2)
            self.assertEqual(parsed["provider_get_ms_total"], 6.0)
            self.assertEqual(parsed["provider_get_ms_max"], 2.0)
            self.assertEqual(parsed["apply_count"], 1)
            self.assertEqual(parsed["apply_ms_total"], 3.0)
            self.assertEqual(parsed["apply_ms_max"], 3.0)
            self.assertEqual(parsed["exec_ms_per_token"], 105.0)
            self.assertEqual(
                parsed["exec_component_sum_ms_per_token"], 103.0)
            self.assertEqual(parsed["exec_timing_source"], "decode-wall")
            self.assertEqual(
                decode_phase_contract_failures(parsed, "cpu"), [])
            self.assertEqual(
                parsed["decode_phase_counter_scope"],
                "decode-only-backend-counter-delta")
            self.assertEqual(parsed["decode_phase_direct_read_ms"], 40.0)
            self.assertEqual(parsed["decode_phase_prepare_ms"], 6.5)
            self.assertEqual(parsed["decode_phase_compute_ms"], 68.0)
            self.assertEqual(parsed["direct_read_calls"], 8)
            self.assertEqual(parsed["direct_read_mb_per_forward"], 2.5)
            self.assertEqual(parsed["reload_bytes"], 10485760)
            self.assertEqual(parsed["evict_bytes"], 10485760)
            self.assertEqual(parsed["pipeline_waits"], 2)
            self.assertEqual(parsed["pipeline_wait_ms"], 4.5)
            self.assertEqual(parsed["pipeline_stage_ms"], 3.0)
            self.assertEqual(parsed["pipeline_retire_ms"], 2.0)
            self.assertEqual(parsed["pipeline_evict_ms"], 1.0)
            self.assertEqual(parsed["pipeline_residency_ms"], 10.0)
            self.assertEqual(parsed["pipeline_unissued_ms"], 1.0)
            self.assertEqual(parsed["pipeline_missing_mib"], 8.0)
            self.assertEqual(parsed["pipeline_unissued_mib"], 2.0)
            self.assertEqual(parsed["pipeline_budget_samples"], 99)
            self.assertEqual(parsed["pipeline_budget_violations"], 0)
            self.assertEqual(
                parsed["pipeline_plan_protection_relaxations"], 3)
            self.assertEqual(
                parsed["pipeline_plan_protection_relaxed_mib"], 4.5)
            self.assertEqual(parsed["pipeline_resident_peak_mib"], 2701.0)
            self.assertEqual(parsed["pipeline_pinned_peak_mib"], 281.8)
            self.assertEqual(parsed["pipeline_over_budget_peak_mib"], 0.0)
            self.assertEqual(parsed["pipeline_load_affinity_cpu"], 0)
            self.assertEqual(parsed["pipeline_load_affinity_rc"], 0)
            self.assertEqual(parsed["pipeline_prepare_affinity_cpu"], 1)
            self.assertEqual(parsed["pipeline_prepare_affinity_rc"], 0)
            self.assertEqual(parsed["frontier_apply_count"], 2)
            self.assertRegex(
                parsed["frontier_trace_sha256"], r"^[0-9a-f]{64}$")
            self.assertRegex(
                parsed["frontier_transition_trace_sha256"],
                r"^[0-9a-f]{64}$")
            self.assertRegex(
                parsed["frontier_shape_trace_sha256"],
                r"^[0-9a-f]{64}$")
            self.assertEqual(parsed["frontier_unique_states"], 2)
            self.assertEqual(parsed["frontier_unique_shape_states"], 1)
            self.assertEqual(parsed["frontier_state_changes"], 1)
            self.assertEqual(parsed["frontier_shape_state_changes"], 0)
            self.assertEqual(
                parsed["frontier_mixed_mode_generations"], 2)
            self.assertEqual(parsed["frontier_multi_units_last"], 1)
            self.assertEqual(parsed["frontier_tensor_units_last"], 2)
            self.assertEqual(parsed["frontier_cut_units_last"], 1)
            self.assertEqual(
                parsed["frontier_delta_weights_total"], 5)
            self.assertEqual(
                parsed["frontier_delta_weights_max"], 3)
            self.assertEqual(
                parsed["frontier_publish_ms_total"], 1.0)
            self.assertEqual(
                parsed["frontier_publish_ms_max"], 0.75)
            self.assertEqual(
                parsed["frontier_transition_publish_count"], 1)
            self.assertEqual(
                parsed["frontier_transition_publish_ms_total"], 0.25)
            self.assertEqual(
                parsed["frontier_transition_publish_ms_max"], 0.25)
            self.assertEqual(
                parsed["frontier_transition_delta_weights_total"], 2)
            self.assertEqual(
                parsed["frontier_transition_delta_weights_max"], 2)
            self.assertEqual(
                parsed["frontier_only_apply_count"], 1)
            self.assertEqual(parsed["granularity_units"], 1234)
            self.assertEqual(parsed["granularity_cut_ops"], 56)
            self.assertEqual(parsed["cpu_delegate_compute_calls"], 4)
            self.assertEqual(parsed["cpu_delegate_compute_ms"], 68.0)
            self.assertEqual(parsed["cpu_delegate_compute_avg_ms"], 17.0)
            self.assertEqual(parsed["stage_xform_calls"], 7)
            self.assertEqual(parsed["stage_xform_ms"], 6.5)
            self.assertEqual(parsed["async_prepare_wait_ms"], 4.0)
            self.assertEqual(parsed["cpu_layout_initial_repack_ms"], 5.0)
            self.assertEqual(parsed["cpu_layout_repack_ms"], 6.0)
            self.assertEqual(parsed["cpu_layout_materialize_ms"], 0.5)
            self.assertEqual(parsed["fused_layout_pair_calls"], 4)
            self.assertEqual(parsed["fused_layout_parallel_calls"], 4)
            self.assertEqual(parsed["fused_layout_pair_ms"], 2.0)
            self.assertEqual(
                parsed["fused_kernel_pair_candidates"], 12)
            self.assertEqual(parsed["fused_kernel_pair_calls"], 10)
            self.assertEqual(
                parsed["fused_kernel_pair_fallbacks"], 2)
            self.assertEqual(parsed["fused_kernel_pair_errors"], 0)
            self.assertEqual(parsed["fused_kernel_pair_ms"], 1.0)
            self.assertEqual(parsed["frontier_predicted_ms_last"], 11.0)
            self.assertAlmostEqual(
                parsed["frontier_switch_ms_total"], 0.3)

    def test_forced_weight_placement_is_strict_and_repeatable(self):
        self.assertEqual(
            parse_forced_weight_placements([
                "output.weight=cpu",
                "token_embd.weight=disk_cpu",
            ]),
            {
                "output.weight": "cpu",
                "token_embd.weight": "disk_cpu",
            })
        with self.assertRaises(ValueError):
            parse_forced_weight_placements([
                "output.weight=cpu",
                "output.weight=disk_cpu",
            ])

    def test_native_budget_mb_is_preserved_without_mapping(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "native.csv"
            source.write_text(
                "time_sec,budget_mb\n"
                "130,5079.9\n"
                "131,3752.1\n"
                "132,6198.6\n",
                encoding="utf-8")
            self.assertEqual(
                read_trace(source),
                [(130.0, 5079.9), (131.0, 3752.1), (132.0, 6198.6)])
            window = write_window_trace(
                source, root / "out",
                window_sec=2.0,
                stride_sec=1.0,
                replay_speedup=1.0,
                bucket_mib=128,
                window_start_sec=130.0)
            self.assertEqual(window.min_mib, 3752.1)
            self.assertEqual(window.max_mib, 6198.6)
            lines = window.local.read_text(encoding="utf-8").splitlines()
            self.assertEqual(lines[0], "t_sec,mem_available_mb")
            self.assertIn("0,5079.9", lines)
            self.assertIn("1,3752.1", lines)
            self.assertIn("2,6198.6", lines)

    def test_token_sequence_hashes_decode_tokens_not_prompt(self):
        with tempfile.TemporaryDirectory() as temporary:
            trace = Path(temporary) / "tokens.csv"
            trace.write_text(
                "n_tokens,token_ids\n"
                "3,10;11;12\n"
                "1,21\n"
                "1,22\n"
                "1,23\n",
                encoding="utf-8")
            full_count, full_hash = token_sequence(trace)
            prefix_count, prefix_hash = token_sequence(trace, 2)
            self.assertEqual(full_count, 3)
            self.assertEqual(prefix_count, 2)
            self.assertIsNotNone(full_hash)
            self.assertIsNotNone(prefix_hash)
            self.assertNotEqual(full_hash, prefix_hash)

    def test_token_mismatch_invalidates_all_compared_methods(self):
        rows = [
            {
                "trace": "trace.csv",
                "method": "static-max",
                "status": "ok",
                "token_prefix_count": 2,
                "token_prefix_sha256": "same",
            },
            {
                "trace": "trace.csv",
                "method": "diff-tree-mixed",
                "status": "ok",
                "token_prefix_count": 2,
                "token_prefix_sha256": "different",
            },
        ]
        mark_token_correctness(
            rows, {"static-max", "diff-tree-mixed"}, 2)
        self.assertTrue(all(
            row["valid_token_sequence"] is False for row in rows))
        self.assertTrue(all(
            row["status"] == "token_mismatch" for row in rows))

    def test_token_mismatch_status_recovers_after_resume_replacement(self):
        rows = [
            {
                "trace": "trace.csv",
                "method": "static-max",
                "status": "token_mismatch",
                "token_prefix_count": 2,
                "token_prefix_sha256": "same",
            },
            {
                "trace": "trace.csv",
                "method": "diff-tree-mixed",
                "status": "ok",
                "token_prefix_count": 2,
                "token_prefix_sha256": "same",
            },
        ]
        mark_token_correctness(
            rows, {"static-max", "diff-tree-mixed"}, 2)
        self.assertTrue(all(
            row["valid_token_sequence"] is True for row in rows))
        self.assertTrue(all(row["status"] == "ok" for row in rows))

    def test_diff_tree_requires_offline_mixed_targets(self):
        self.assertTrue(methods_need_mixed_offline_table(
            "static-min,diff-tree-mixed"))
        self.assertTrue(methods_need_mixed_offline_table(
            "static-min,offline-mixed"))
        self.assertEqual(
            required_plan_budgets(
                ["diff-tree-mixed"], [], [3840, 3968], mixed=True),
            [3840, 3968])


if __name__ == "__main__":
    unittest.main()
