#!/usr/bin/env python3

import unittest
from subprocess import CompletedProcess
from unittest import mock

from scripts.elastic.run_granularity_model_sweep import (
    external_inference_processes,
    fixed_run_config_sha256,
    mark_fixed_token_correctness,
    parse_current_temperatures,
    parse_log,
    pin_policy_requests,
    remote_sha256,
    validate_pipeline_run,
)


class GranularitySweepPinAuditTest(unittest.TestCase):
    def test_fixed_gpu_rejects_fused_pair_budget_fallback(self):
        class Args:
            backend = "gpu"
            pipeline = "on"
            require_device_idle = False
            pin = ""
            weight_mib = 100.0
            cut_parts = 2
            gpu_unit_sync = "none"
            cut_dual_compute = "fused"
            max_compute_call_ms = 0.0
            fixed_warmup_decode_tokens = 0
            compute_recovery_ratio = 1.0
            min_cpu_freq_limit_khz = 0
            min_cpu_mean_freq_khz = 0
            min_cpu_median_sample_min_khz = 0
            pipeline_lookahead_mib = 128
            pipeline_copy_cpu = -1
            require_real_batch_io = False

        row = {
            "returncode": 0,
            "token_trace_pull_returncode": 0,
            "budget_trace_sha256": "trace",
            "remote_budget_trace_sha256": "trace",
            "unit_pipeline_issued": 1,
            "async_load_units_enqueued": 1,
            "async_load_units_completed": 1,
            "async_prepare_units_enqueued": 1,
            "async_prepare_units_completed": 1,
            "pipeline_window_max_mib": 1.0,
            "pipeline_window_oversize": 0,
            "compute_calls": 1,
            "compute_avg_call_ms": 1.0,
            "pipeline_budget_samples": 1,
            "pipeline_budget_violations": 0,
            "pipeline_plan_protection_relaxations": 0,
            "pipeline_resident_peak_mib": 90.0,
            "pipeline_pinned_peak_mib": 10.0,
            "pipeline_over_budget_peak_mib": 0.0,
            "wbm_total_mib": 100.0,
            "physical_cut_tensors": 1,
            "physical_cut_parts": 2,
            "soa_pool_hits": 1,
            "gpu_unit_boundaries": 1,
            "gpu_unit_flushes": 0,
            "gpu_unit_finishes": 0,
            "gpu_unit_sync_errors": 0,
            "cut_dual_calls": 1,
            "cut_dual_budget_fallbacks": 0,
            "cut_dual_queue_errors": 0,
            "cut_dual_image_errors": 0,
            "cut_dual_image_creates": 1,
            "cut_dual_image_releases": 0,
        }
        valid, reasons = validate_pipeline_run(
            Args(), "tensor", dict(row))
        self.assertTrue(valid, reasons)
        row["cut_dual_budget_fallbacks"] = 1
        valid, reasons = validate_pipeline_run(
            Args(), "tensor", row)
        self.assertFalse(valid)
        self.assertTrue(any(
            "both tiles did not fit" in reason for reason in reasons))

    def test_gpu_prepare_breakdown_excludes_host_source_load_interval(self):
        parsed = parse_log(
            "detail soa_pool_lookup    calls=2 total=1.000 ms avg=0.500 ms MB=2.0\n"
            "detail parent_alloc       calls=2 total=2.000 ms avg=1.000 ms MB=2.0\n"
            "detail staging_alloc      calls=2 total=3.000 ms avg=1.500 ms MB=2.0\n"
            "detail host_src           calls=2 total=100.000 ms avg=50.000 ms MB=2.0\n"
            "detail write_enqueue      calls=2 total=4.000 ms avg=2.000 ms MB=2.0\n"
            "detail subbuffer          calls=2 total=5.000 ms avg=2.500 ms MB=2.0\n"
            "detail barrier_enqueue    calls=2 total=6.000 ms avg=3.000 ms MB=2.0\n"
            "detail convert_enqueue    calls=2 total=7.000 ms avg=3.500 ms MB=2.0\n"
            "detail transpose_enqueue  calls=2 total=8.000 ms avg=4.000 ms MB=2.0\n"
            "detail final_wait_enqueue calls=2 total=9.000 ms avg=4.500 ms MB=2.0\n"
            "detail mark_resident      calls=2 total=10.000 ms avg=5.000 ms MB=2.0\n"
            "detail cpu_xform          calls=2 total=11.000 ms avg=5.500 ms MB=2.0\n"
        )
        self.assertEqual(parsed["gpu_load_host_source_ms"], 100.0)
        self.assertEqual(parsed["gpu_prepare_host_total_ms"], 66.0)
        self.assertEqual(parsed["gpu_prepare_calls"], 2)
        self.assertEqual(parsed["gpu_prepare_mib"], 2.0)

    def test_fixed_resume_config_fingerprint_changes_with_pipeline(self):
        class Args:
            backend = "cpu"
            modes = ["multi", "tensor", "cut"]
            pipeline = "on"
            pipeline_lookahead_mib = 128

        first = fixed_run_config_sha256(Args())
        self.assertEqual(first, fixed_run_config_sha256(Args()))
        Args.pipeline_lookahead_mib = 64
        self.assertNotEqual(first, fixed_run_config_sha256(Args()))

    def test_fixed_isolation_requires_observing_measured_process(self):
        class Args:
            pin = ""
            pipeline = "off"
            require_device_idle = True

        base = {
            "returncode": 0,
            "token_trace_pull_returncode": 0,
            "budget_trace_sha256": "trace-hash",
            "remote_budget_trace_sha256": "trace-hash",
        }
        valid, reasons = validate_pipeline_run(Args(), "tensor", dict(base))
        self.assertFalse(valid)
        self.assertIn(
            "measurement isolation monitor had no samples", reasons)
        self.assertIn(
            "measurement isolation monitor did not observe llama-cli",
            reasons,
        )

        valid, reasons = validate_pipeline_run(
            Args(),
            "tensor",
            {
                **base,
                "measurement_monitor_samples": 1,
                "measurement_llama_pid": 123,
            },
        )
        self.assertTrue(valid, reasons)

    @mock.patch(
        "scripts.elastic.run_granularity_model_sweep.adb")
    def test_isolation_matches_executable_and_keeps_same_path_competitor(
            self, adb):
        adb.return_value = CompletedProcess(
            args=[], returncode=0,
            stdout=(
                "shell 100 1 sh -c ./llama-cli -m model.gguf\n"
                "shell 101 100 /data/local/tmp/llama-cli -m model.gguf\n"
                "shell 102 1 sha256sum /tmp/llama-cli\n"
                "shell 103 1 /data/local/tmp/granularity-pipeline-bench\n"
            ),
        )
        self.assertEqual(
            external_inference_processes("serial"),
            [
                "shell 101 100 /data/local/tmp/llama-cli -m model.gguf",
                "shell 103 1 /data/local/tmp/granularity-pipeline-bench",
            ],
        )

    @mock.patch(
        "scripts.elastic.run_granularity_model_sweep.adb")
    def test_fixed_isolation_fails_closed_on_adb_error(self, adb):
        adb.return_value = CompletedProcess(
            args=[], returncode=1, stdout="device offline")
        with self.assertRaisesRegex(
                RuntimeError, "failed to sample inference processes"):
            external_inference_processes("serial")

    @mock.patch(
        "scripts.elastic.run_granularity_model_sweep.adb")
    def test_remote_input_hash_is_parsed_exactly(self, adb):
        expected = "c" * 64
        adb.return_value = CompletedProcess(
            args=[], returncode=0,
            stdout=f"{expected}  /data/local/tmp/model.gguf\n",
        )
        self.assertEqual(
            remote_sha256(
                "serial", "/data/local/tmp/model.gguf", 30),
            expected,
        )

    def test_fixed_thermal_gate_does_not_use_stale_cache(self):
        stale = (
            "Cached temperatures:\n"
            "Temperature{mValue=30.0, mType=0, mName=CPU7, mStatus=0}\n"
        )
        self.assertEqual(
            parse_current_temperatures(stale),
            (None, None, None),
        )
        current = (
            stale
            + "Current temperatures from HAL:\n"
            "Temperature{mValue=34.0, mType=0, mName=CPU7, mStatus=0}\n"
            "Temperature{mValue=35.0, mType=1, mName=GPU0, mStatus=0}\n"
            "Temperature{mValue=30.0, mType=3, mName=skin, mStatus=0}\n"
            "Current cooling devices from HAL:\n"
            "Temperature{mValue=99.0, mType=0, mName=not-a-sensor,"
            " mStatus=0}\n"
        )
        self.assertEqual(
            parse_current_temperatures(current),
            (34.0, 35.0, 30.0),
        )

    def test_parse_inside_budget_pin_for_cpu_and_cut_opencl_parts(self):
        parsed = parse_log(
            "elastic: pin token_embd inside weight budget "
            "(281.81 MiB, 1 physical unit)\n"
            "ggml_opencl elastic: pin output inside weight budget "
            "(205.49 MiB)\n"
            "ggml_opencl elastic: pin output inside weight budget "
            "(205.49 MiB)\n")
        self.assertEqual(parsed["pin_token_embd_inside_budget_count"], 1)
        self.assertAlmostEqual(
            parsed["pin_token_embd_inside_budget_mib"], 281.81)
        self.assertEqual(parsed["pin_output_inside_budget_count"], 2)
        self.assertAlmostEqual(parsed["pin_output_inside_budget_mib"], 410.98)
        self.assertEqual(parsed["pin_outside_budget_count"], 0)

    def test_fixed_latency_requires_full_decode_wall_clock(self):
        class Args:
            pin = ""
            pipeline = "off"
            require_device_idle = False

        parsed = parse_log(
            "llama_perf_context_print: eval time = 300.0 ms / 3 runs "
            "(100.0 ms per token)\n"
            "llama_context::~llama_context: elastic decode wall summary: "
            "runs=3 wall_ms=315.0\n"
            "[elastic-bench] decode phase quiesce boundary=begin rc=0\n"
            "[elastic-bench] decode phase origin generated=0 backend=cpu\n"
            "[elastic-bench] decode phase quiesce boundary=end rc=0\n"
            "[elastic decode phase summary] backend=cpu runs=3 units=30 "
            "nonresident_units=12 "
            "reloads=2 reload_bytes=1048576 direct_read_calls=2 "
            "direct_read_us=1000 direct_read_bytes=1048576 "
            "load_calls=2 load_us=1200 load_bytes=1048576 "
            "prepare_calls=2 prepare_us=900 prepare_bytes=1048576 "
            "compute_available=1 compute_calls=3 compute_us=300000 "
            "pipeline_waits=1 pipeline_wait_us=100 "
            "pipeline_residency_us=200 pipeline_unissued_us=0 "
            "pipeline_stage_us=50 pipeline_retire_us=50 evict_us=20 "
            "resident_bytes=1048576\n"
        )
        row = {
            "returncode": 0,
            "token_trace_pull_returncode": 0,
            "budget_trace_sha256": "trace",
            "remote_budget_trace_sha256": "trace",
            "fixed_decode_tokens_total": 3,
            **parsed,
        }
        valid, reasons = validate_pipeline_run(Args(), "tensor", row)
        self.assertTrue(valid, reasons)
        self.assertTrue(row["decode_wall_contract_valid"])
        self.assertEqual(row["latency_timing_source"], "decode-wall")

        row["decode_phase_quiesce_end_rc"] = -1
        valid, reasons = validate_pipeline_run(Args(), "tensor", row)
        self.assertFalse(valid)
        self.assertIn("decode phase end quiescence failed", reasons)
        row["decode_phase_quiesce_end_rc"] = 0

        row["decode_wall_runs"] = 2
        valid, reasons = validate_pipeline_run(Args(), "tensor", row)
        self.assertFalse(valid)
        self.assertIn("decode wall runs=2, eval runs=3", reasons)

    def test_requested_pin_must_be_observed(self):
        class Args:
            pin = "token_embd,output"
            pipeline = "off"

        row = {
            "returncode": 0,
            "token_trace_pull_returncode": 0,
            **parse_log(
                "elastic: pin token_embd inside weight budget "
                "(281.81 MiB, 1 physical unit)\n"),
        }
        valid, reasons = validate_pipeline_run(Args(), "tensor", row)
        self.assertFalse(valid)
        self.assertIn(
            "requested output pin was not observed inside the weight budget",
            reasons,
        )

    def test_partial_cut_pin_is_invalid(self):
        class Args:
            pin = "token_embd,output"
            pipeline = "off"
            expected_pin_token_embd_mib = 281.8125
            expected_pin_output_mib = 410.9765625

        row = {
            "returncode": 0,
            "token_trace_pull_returncode": 0,
            **parse_log(
                "elastic: pin token_embd inside weight budget "
                "(281.81 MiB, 1 physical unit)\n"
                "ggml_opencl elastic: pin output inside weight budget "
                "(205.49 MiB)\n"),
        }
        valid, reasons = validate_pipeline_run(Args(), "cut", row)
        self.assertFalse(valid)
        self.assertTrue(any(
            "output pinned 205.49 MiB, expected 410.98 MiB" in reason
            for reason in reasons
        ))

    def test_outside_budget_pin_is_invalid_even_when_requested(self):
        class Args:
            pin = "all"
            pipeline = "off"

        row = {
            "returncode": 0,
            "token_trace_pull_returncode": 0,
            **parse_log(
                "elastic: pin token_embd inside weight budget "
                "(281.81 MiB, 1 physical unit)\n"
                "ggml_opencl elastic: pin output inside weight budget "
                "(410.98 MiB)\n"
                "ggml_opencl elastic: pin unplanned output (410 MB) "
                "outside budget -> target=3278 MB\n"),
        }
        valid, reasons = validate_pipeline_run(Args(), "multi", row)
        self.assertFalse(valid)
        self.assertTrue(any("enlarged the weight budget" in reason
                            for reason in reasons))

    def test_pin_policy_is_token_exact(self):
        self.assertTrue(pin_policy_requests("norm,token_embd,output", "output"))
        self.assertTrue(pin_policy_requests("all", "token_embd"))
        self.assertFalse(pin_policy_requests("norm,attn_output", "output"))

    def test_parse_opencl_plan_protection_audit(self):
        parsed = parse_log(
            "[elastic unit pipeline budget] samples=99 violations=0 "
            "plan_protection_relaxations=2 relaxed_mib=15.75 "
            "resident_peak_mib=2701.0 pinned_peak_mib=692.8 "
            "over_peak_mib=0.0\n")
        self.assertEqual(parsed["pipeline_budget_samples"], 99)
        self.assertEqual(parsed["pipeline_budget_violations"], 0)
        self.assertEqual(
            parsed["pipeline_plan_protection_relaxations"], 2)
        self.assertEqual(parsed["pipeline_resident_peak_mib"], 2701.0)
        self.assertEqual(parsed["pipeline_pinned_peak_mib"], 692.8)
        self.assertEqual(parsed["pipeline_over_budget_peak_mib"], 0.0)

    def test_parse_gpu_tiled_image_cache_audit(self):
        parsed = parse_log(
            "[elastic cut dual] enabled=1 candidates=20 calls=18 "
            "budget_fallbacks=1 shape_fallbacks=1 queue_errors=0 "
            "image_creates=12 image_cache_hits=24 image_releases=8 "
            "image_errors=0\n")
        self.assertEqual(parsed["cut_dual_calls"], 18)
        self.assertEqual(parsed["cut_dual_image_creates"], 12)
        self.assertEqual(parsed["cut_dual_image_cache_hits"], 24)
        self.assertEqual(parsed["cut_dual_image_releases"], 8)
        self.assertEqual(parsed["cut_dual_image_errors"], 0)

    def test_parse_common_dynamic_capable_physical_tiles(self):
        parsed = parse_log(
            "[elastic granularity] backend=opencl mode=tensor units=10 "
            "cut_ops=0 fallback=0 peak_unit=2.00 MiB "
            "cut_tensors=7 cut_parts=14 wbm_total=4436.79 MiB\n")
        self.assertEqual(parsed["physical_cut_tensors"], 7)
        self.assertEqual(parsed["physical_cut_parts"], 14)
        self.assertAlmostEqual(parsed["wbm_total_mib"], 4436.79)

    def test_parse_cpu_profile_physical_tiles(self):
        parsed = parse_log(
            "  granularity  : mode=tensor units=10 cut_ops=0 "
            "fallback_ops=0 peak_unit=2.00 MiB tensors=225 "
            "cut_tensors=7 parts=14 wbm_total=4436.79 MiB\n")
        self.assertEqual(parsed["physical_cut_tensors"], 7)
        self.assertEqual(parsed["physical_cut_parts"], 14)
        self.assertAlmostEqual(parsed["wbm_total_mib"], 4436.79)

    def test_fixed_correctness_compares_the_complete_token_sequence(self):
        rows = [
            {
                "repeat": 0,
                "ratio_pct": 30,
                "mode": "multi",
                "token_id_count": 12,
                "token_ids_sha256": "full-a",
                "token_prefix_sha256": "shared-prefix",
            },
            {
                "repeat": 0,
                "ratio_pct": 30,
                "mode": "tensor",
                "token_id_count": 12,
                "token_ids_sha256": "full-b",
                "token_prefix_sha256": "shared-prefix",
            },
        ]
        mark_fixed_token_correctness(rows, {"multi", "tensor"})
        self.assertFalse(rows[0]["valid_token_sequence"])
        self.assertFalse(rows[1]["valid_token_sequence"])


if __name__ == "__main__":
    unittest.main()
