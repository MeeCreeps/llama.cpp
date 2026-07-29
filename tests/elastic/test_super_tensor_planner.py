#!/usr/bin/env python3

import unittest
from unittest.mock import patch

from runtime.plan.super_tensor_planner import (
    Decision,
    GranularityCostModel,
    _edited_pipeline_end,
    _partition_prefix_suffix,
    apply_local_edit,
    attach_working_unit,
    diff_tree_partition,
    fixed_partition,
    local_edits,
    mode_boundary_count,
    normalize_weights,
    offline_partition,
    online_partition,
    partition_to_working_unit,
    refine_partition,
    simulate_partition,
    working_unit_to_partition,
)


def model_meta():
    weights = []
    ops = []
    for index, size in enumerate((10, 10, 8, 8)):
        weights.append({
            "weight_id": index,
            "name": f"blk.0.w{index}.weight",
            "layer": 0,
            "byte_size": size * 1024 * 1024,
            "shape": [4096, 512],
        })
        ops.append({
            "op_id": index,
            "weight_id": index,
            "name": f"blk.0.w{index}.weight",
        })
    return {"weights": weights, "ops": ops}


def plan(resident: bool):
    return {
        "weights": [{
            "weight_id": index,
            "name": f"blk.0.w{index}.weight",
            "byte_size": size * 1024 * 1024,
            "location": "cpu" if resident else "disk",
        } for index, size in enumerate((10, 10, 8, 8))],
        "ops": [],
    }


class SuperTensorPlannerTest(unittest.TestCase):
    def test_gpu_fused_cut_waits_for_both_prepares(self):
        profile = {
            "profiles": {"gpu": {
                "io_bandwidth_mib_s": 1000.0,
                "prepare_bandwidth_mib_s": 1000.0,
                "compute_ms_per_mib": 1.0,
                "io_request_ms": 0.0,
                "prepare_launch_ms": 0.0,
                "compute_launch_ms": 3.0,
                "cut_compute_grouping": "fused_pair",
            }},
        }
        cost = GranularityCostModel(profile, "gpu")
        weights = normalize_weights(model_meta(), plan(False), "gpu")
        by_id = {weight.weight_id: weight for weight in weights}
        halves = cost.decision_costs(Decision("cut", (0,)), by_id)
        self.assertEqual(len(halves), 2)
        self.assertEqual(halves[0].compute_ms, 0.0)
        self.assertEqual(halves[1].compute_ms, 13.0)
        state = simulate_partition(
            [Decision("cut", (0,))], weights, cost)
        # io=10, prepare=10, then one 13 ms fused GEMV. Compute cannot
        # start after only the first half has been prepared.
        self.assertEqual(state.compute_end_ms, 28.0)

    def test_cpu_cut_keeps_independent_half_compute(self):
        profile = {
            "profiles": {"cpu": {
                "compute_ms_per_mib": 1.0,
                "compute_launch_ms": 3.0,
                "cut_compute_grouping": "independent_halves",
            }},
        }
        cost = GranularityCostModel(profile, "cpu")
        weights = normalize_weights(model_meta(), plan(True), "cpu")
        by_id = {weight.weight_id: weight for weight in weights}
        halves = cost.decision_costs(Decision("cut", (0,)), by_id)
        self.assertEqual(
            [half.compute_ms for half in halves], [8.0, 8.0])

    def test_local_edits_preserve_unaffected_runtime_unit_ids(self):
        cost = GranularityCostModel({}, "cpu")
        weights = normalize_weights(model_meta(), plan(False), "cpu")

        def unit_ids(decisions):
            working_unit = partition_to_working_unit(
                decisions,
                weights,
                cost,
                policy="test",
                predicted_ms=0.0,
            )
            result = {}
            for unit in working_unit["units"]:
                for tile in unit["tiles"]:
                    result.setdefault(
                        int(tile["weight_id"]), []).append(
                            int(unit["unit_id"]))
            return {
                weight_id: tuple(ids)
                for weight_id, ids in result.items()
            }

        tensor = [
            Decision("tensor", (weight.weight_id,))
            for weight in weights
        ]
        cut_first = [
            Decision("cut", (0,)),
            *tensor[1:],
        ]
        merge_first_pair = [
            Decision("multi", (0, 1)),
            *tensor[2:],
        ]
        tensor_ids = unit_ids(tensor)
        cut_ids = unit_ids(cut_first)
        multi_ids = unit_ids(merge_first_pair)
        self.assertNotEqual(tensor_ids[0], cut_ids[0])
        self.assertNotEqual(tensor_ids[0], multi_ids[0])
        self.assertNotEqual(tensor_ids[1], multi_ids[1])
        for weight_id in (1, 2, 3):
            self.assertEqual(tensor_ids[weight_id], cut_ids[weight_id])
        for weight_id in (2, 3):
            self.assertEqual(
                tensor_ids[weight_id], multi_ids[weight_id])

    def test_offline_changes_with_residency(self):
        profile = {
            "source": "synthetic-test",
            "profiles": {"cpu": {
                "io_request_ms": 0.01,
                "prepare_launch_ms": 0.01,
                "compute_launch_ms": 0.08,
                "compute_ms_per_mib": 0.2,
                "mode_prepare_scale": {
                    "multi": 1.0, "tensor": 1.0, "cut": 0.35},
                "mode_compute_scale": {
                    "multi": 0.70, "tensor": 1.0, "cut": 1.45},
            }},
        }
        cost = GranularityCostModel(profile, "cpu")
        low_weights = normalize_weights(model_meta(), plan(False), "cpu")
        high_weights = normalize_weights(model_meta(), plan(True), "cpu")
        low, _ = offline_partition(low_weights, cost)
        high, _ = offline_partition(high_weights, cost)
        self.assertIn("cut", [decision.kind for decision in low])
        self.assertIn("multi", [decision.kind for decision in high])

    def test_ratio_curve_separates_work_volume_from_phase_rate(self):
        profile = {
            "profiles": {"cpu": {
                "io_bandwidth_mib_s": 1000.0,
                "prepare_bandwidth_mib_s": 1000.0,
                "io_request_ms": 0.0,
                "prepare_launch_ms": 0.0,
                "compute_launch_ms": 0.0,
                "compute_ms_per_mib": 0.0,
                "mode_io_scale": {
                    "multi": 1.0, "tensor": 1.0, "cut": 1.0},
                "mode_io_work_curve": {
                    "cut": [
                        {"resident_ratio": 0.25, "scale": 0.5},
                        {"resident_ratio": 0.75, "scale": 1.0},
                    ],
                },
            }},
        }
        cost = GranularityCostModel(profile, "cpu")
        placement = plan(False)
        # Exactly half of model bytes are resident.
        placement["weights"][0]["location"] = "cpu"
        placement["weights"][2]["location"] = "cpu"
        weights = normalize_weights(model_meta(), placement, "cpu")
        self.assertAlmostEqual(weights[0].plan_resident_ratio, 0.5)
        by_id = {weight.weight_id: weight for weight in weights}
        tensor = cost.decision_costs(Decision("tensor", (1,)), by_id)
        cut = cost.decision_costs(Decision("cut", (1,)), by_id)
        # Interpolation gives 0.75x useful I/O work for Cut. It remains
        # distinct from mode_io_scale (time per processed MiB), which is 1.
        self.assertAlmostEqual(sum(unit.io_ms for unit in tensor), 10.0)
        self.assertAlmostEqual(sum(unit.io_ms for unit in cut), 7.5)

    def test_per_weight_cost_override_is_applied(self):
        cost = GranularityCostModel({
            "profiles": {"cpu": {
                "io_request_ms": 0.0,
                "weights": {
                    "blk.0.w0.weight": {"io_ms": 123.0},
                },
            }},
        }, "cpu")
        weights = normalize_weights(model_meta(), plan(False), "cpu")
        by_id = {weight.weight_id: weight for weight in weights}
        units = cost.decision_costs(Decision("tensor", (0,)), by_id)
        self.assertAlmostEqual(units[0].io_ms, 123.0)

    def test_mode_unit_overhead_is_separate_from_kernel_rate(self):
        cost = GranularityCostModel({
            "profiles": {"cpu": {
                "compute_ms_per_mib": 0.0,
                "compute_launch_ms": 0.0,
                "mode_unit_overhead_ms": {
                    "multi": 0.1, "tensor": 0.0, "cut": 0.02,
                },
            }},
        }, "cpu")
        weights = normalize_weights(model_meta(), plan(True), "cpu")
        by_id = {weight.weight_id: weight for weight in weights}
        multi = cost.decision_costs(Decision("multi", (0, 1)), by_id)
        tensor = cost.decision_costs(Decision("tensor", (0,)), by_id)
        cut = cost.decision_costs(Decision("cut", (0,)), by_id)
        self.assertAlmostEqual(multi[0].compute_ms, 0.1)
        self.assertAlmostEqual(tensor[0].compute_ms, 0.0)
        self.assertEqual(len(cut), 2)
        self.assertTrue(all(
            abs(unit.compute_ms - 0.02) < 1e-9 for unit in cut))

    def test_mode_boundary_cost_is_separate_from_pipeline_stages(self):
        common = {
            "io_bandwidth_mib_s": 1.0e12,
            "prepare_bandwidth_mib_s": 1.0e12,
            "compute_ms_per_mib": 0.0,
            "io_request_ms": 0.0,
            "prepare_launch_ms": 0.0,
            "compute_launch_ms": 0.0,
        }
        base = GranularityCostModel({
            "profiles": {"cpu": dict(common, mode_boundary_ms=0.0)},
        }, "cpu")
        penalized = GranularityCostModel({
            "profiles": {"cpu": dict(common, mode_boundary_ms=3.0)},
        }, "cpu")
        weights = normalize_weights(model_meta(), plan(True), "cpu")
        decisions = [
            Decision("tensor", (0,)),
            Decision("cut", (1,)),
            Decision("tensor", (2,)),
            Decision("tensor", (3,)),
        ]
        self.assertEqual(mode_boundary_count(decisions), 2)
        base_state = simulate_partition(decisions, weights, base)
        penalized_state = simulate_partition(
            decisions, weights, penalized)
        self.assertAlmostEqual(
            penalized_state.compute_end_ms -
            base_state.compute_end_ms,
            6.0,
        )
        self.assertAlmostEqual(
            penalized_state.io_end_ms, base_state.io_end_ms)
        self.assertAlmostEqual(
            penalized_state.prepare_end_ms, base_state.prepare_end_ms)
        # A Tensor fallback inside an otherwise coarse Multi frontier keeps
        # whole-tensor physical semantics and is not a Cut boundary.
        whole_tensor_decisions = [
            Decision("multi", (0, 1)),
            Decision("tensor", (2,)),
            Decision("tensor", (3,)),
        ]
        whole_base = simulate_partition(
            whole_tensor_decisions, weights, base)
        whole_penalized = simulate_partition(
            whole_tensor_decisions, weights, penalized)
        self.assertAlmostEqual(
            whole_penalized.compute_end_ms,
            whole_base.compute_end_ms,
        )

    def test_refinement_removes_unprofitable_cut_boundaries(self):
        cost = GranularityCostModel({
            "profiles": {"cpu": {
                "io_bandwidth_mib_s": 1.0e12,
                "prepare_bandwidth_mib_s": 1.0e12,
                "compute_ms_per_mib": 0.0,
                "io_request_ms": 0.0,
                "prepare_launch_ms": 0.0,
                "compute_launch_ms": 0.0,
                "cut_boundary_ms": 5.0,
            }},
        }, "cpu")
        weights = normalize_weights(model_meta(), plan(True), "cpu")
        initial = [
            Decision("tensor", (0,)),
            Decision("cut", (1,)),
            Decision("tensor", (2,)),
            Decision("tensor", (3,)),
        ]
        refined, state = refine_partition(
            initial, weights, cost, max_edits=2)
        self.assertEqual(
            [decision.kind for decision in refined],
            ["tensor", "tensor", "tensor", "tensor"],
        )
        self.assertEqual(mode_boundary_count(refined), 0)
        self.assertAlmostEqual(state.compute_end_ms, 0.0)

    def test_pipeline_residual_does_not_rescale_resident_compute(self):
        cost = GranularityCostModel({
            "profiles": {"cpu": {
                "compute_ms_per_mib": 1.0,
                "compute_launch_ms": 0.0,
                "mode_compute_scale": {
                    "multi": 0.8, "tensor": 1.0, "cut": 1.2,
                },
                "mode_pipeline_efficiency_curve": {
                    "multi": [
                        {"resident_ratio": 1.0, "scale": 1.5},
                    ],
                    "tensor": [
                        {"resident_ratio": 1.0, "scale": 1.0},
                    ],
                    "cut": [
                        {"resident_ratio": 1.0, "scale": 1.4},
                    ],
                },
            }},
        }, "cpu")
        weights = normalize_weights(model_meta(), plan(True), "cpu")
        by_id = {weight.weight_id: weight for weight in weights}
        multi = cost.decision_costs(Decision("multi", (0, 1)), by_id)[0]
        tensor = cost.decision_costs(Decision("tensor", (0,)), by_id)[0]
        # The residual may correct non-resident I/O/layout exposure, but a
        # fully resident decision must retain the measured kernel-rate scale.
        self.assertAlmostEqual(multi.compute_ms, 16.0)
        self.assertAlmostEqual(tensor.compute_ms, 10.0)
        self.assertEqual(multi.io_ms, 0.0)
        self.assertEqual(multi.prepare_ms, 0.0)

    def test_plan_roundtrip_preserves_mixed_partition(self):
        profile = {
            "profiles": {"cpu": {
                "mode_prepare_scale": {
                    "multi": 1.0, "tensor": 1.0, "cut": 0.3},
                "mode_compute_scale": {
                    "multi": 0.65, "tensor": 1.0, "cut": 1.35},
            }},
        }
        cost = GranularityCostModel(profile, "cpu")
        output = attach_working_unit(
            model_meta(), plan(False), cost, policy="offline")
        weights = normalize_weights(model_meta(), output, "cpu")
        restored = working_unit_to_partition(
            output["working_unit"], weights)
        self.assertTrue(restored)
        self.assertEqual(
            {weight_id for decision in restored
             for weight_id in decision.weight_ids},
            {0, 1, 2, 3})
        for unit in output["working_unit"]["units"]:
            self.assertGreaterEqual(unit["unit_id"], 0)
            self.assertTrue(unit["tiles"])

    def test_fixed_references_share_tiles_but_not_units(self):
        cost = GranularityCostModel({}, "cpu")
        weights = normalize_weights(model_meta(), plan(False), "cpu")
        multi, _ = fixed_partition(weights, cost, "multi")
        tensor, _ = fixed_partition(weights, cost, "tensor")
        cut, _ = fixed_partition(weights, cost, "cut")
        self.assertLess(len(multi), len(tensor))
        self.assertEqual(len(tensor), len(weights))
        self.assertEqual(
            sum(2 if decision.kind == "cut" else 1 for decision in cut),
            2 * len(weights))
        for policy in ("fixed-multi", "fixed-tensor", "fixed-cut"):
            output = attach_working_unit(
                model_meta(), plan(False), cost, policy=policy)
            covered = {
                int(tile["weight_id"])
                for unit in output["working_unit"]["units"]
                for tile in unit["tiles"]
            }
            self.assertEqual(covered, {0, 1, 2, 3})

    def test_disk_variants_are_nonresident_and_3d_is_not_cut(self):
        meta = model_meta()
        meta["weights"][3]["shape"] = [4096, 512, 8]
        placement = plan(True)
        placement["weights"][0]["location"] = "disk_cpu"
        placement["weights"][1]["location"] = "disk_gpu"
        weights = normalize_weights(meta, placement, "cpu")
        self.assertFalse(weights[0].resident)
        self.assertFalse(weights[1].resident)
        self.assertTrue(weights[2].resident)
        self.assertFalse(weights[3].eligible_cut)

    def test_gpu_cut_matches_backend_materialization_rules(self):
        meta = model_meta()
        meta["weights"][0]["quant_name"] = "F16"
        meta["weights"][1]["name"] = "token_embd.weight"
        meta["weights"][2]["quant_name"] = "Q4_0"
        weights = normalize_weights(meta, plan(False), "gpu")
        self.assertFalse(weights[0].eligible_cut)
        self.assertFalse(weights[1].eligible_cut)
        self.assertTrue(weights[2].eligible_cut)

    def test_fusion_flags_require_equal_2d_layouts(self):
        meta = model_meta()
        # The pair remains one scheduling unit, but the fused kernel cannot
        # consume unequal physical layouts.
        meta["weights"][1]["shape"] = [2048, 1024]
        cost = GranularityCostModel({}, "cpu")
        output = attach_working_unit(
            meta, plan(False), cost, policy="fixed-multi")
        first = output["working_unit"]["units"][0]
        self.assertEqual(
            {tile["weight_id"] for tile in first["tiles"]}, {0, 1})
        self.assertFalse(first["fuse_layout"])
        self.assertFalse(first["fuse_compute"])

    def test_selected_cpu_multi_fused_publishes_both_fusion_flags(self):
        cost = GranularityCostModel({
            "profiles": {
                "cpu": {"multi_implementation": "multi_fused"},
            },
        }, "cpu")
        output = attach_working_unit(
            model_meta(), plan(False), cost, policy="fixed-multi")
        first = output["working_unit"]["units"][0]
        self.assertEqual(
            {tile["weight_id"] for tile in first["tiles"]}, {0, 1})
        self.assertTrue(first["fuse_layout"])
        self.assertTrue(first["fuse_compute"])

    def test_diff_tree_is_local_and_guarded(self):
        profile = {
            "profiles": {"cpu": {
                "transition_fixed_ms": 0.01,
                "transition_ms_per_mib": 0.001,
                "hysteresis_ms": 0.01,
                "mode_prepare_scale": {
                    "multi": 1.0, "tensor": 1.0, "cut": 0.25},
                "mode_compute_scale": {
                    "multi": 0.8, "tensor": 1.0, "cut": 1.1},
            }},
        }
        cost = GranularityCostModel(profile, "cpu")
        weights = normalize_weights(model_meta(), plan(False), "cpu")
        current = [Decision("tensor", (weight.weight_id,))
                   for weight in weights]
        target, _ = offline_partition(weights, cost)
        chosen, _, diagnostics = diff_tree_partition(
            current, target, weights, cost,
            horizon_tokens=16, max_edits=1)
        self.assertLessEqual(len(diagnostics["edits"]), 1)
        if diagnostics["edits"]:
            self.assertLess(
                diagnostics["offline_distance_after"],
                diagnostics["offline_distance_before"])
        else:
            self.assertEqual(
                diagnostics["offline_distance_after"],
                diagnostics["offline_distance_before"])
        self.assertNotEqual(chosen, [])

        guarded, _, guarded_diag = diff_tree_partition(
            current, target, weights,
            GranularityCostModel({
                "profiles": {"cpu": {
                    "transition_fixed_ms": 10_000.0,
                    "hysteresis_ms": 10_000.0,
                }}}, "cpu"),
            horizon_tokens=1, max_edits=4)
        self.assertEqual(guarded, current)
        self.assertEqual(guarded_diag["edits"], [])

    def test_zero_wait_feedback_keeps_telemetry_diagnostic_only(self):
        cost = GranularityCostModel({
            "profiles": {"cpu": {
                "wait_feedback_weight": 0.0,
                "transition_fixed_ms": 0.0,
                "hysteresis_ms": 0.0,
                "mode_prepare_scale": {
                    "multi": 1.0, "tensor": 1.0, "cut": 0.2},
                "mode_compute_scale": {
                    "multi": 0.8, "tensor": 1.0, "cut": 1.2},
            }}
        }, "cpu")
        weights = normalize_weights(model_meta(), plan(False), "cpu")
        current, _ = fixed_partition(weights, cost, "tensor")
        target, _ = offline_partition(weights, cost)
        telemetry_replays = 0

        def tracked(decisions, candidate_weights, candidate_cost,
                    wait_ratio=0.0):
            nonlocal telemetry_replays
            if wait_ratio > 0.0:
                telemetry_replays += 1
            return simulate_partition(
                decisions, candidate_weights, candidate_cost, wait_ratio)

        with patch(
            "runtime.plan.super_tensor_planner.simulate_partition",
            side_effect=tracked,
        ):
            _, _, diagnostics = diff_tree_partition(
                current,
                target,
                weights,
                cost,
                telemetry={
                    "units": 1000,
                    "pipeline_wait_us": 1_000_000,
                    "prepare_us": 1_000_000,
                },
                horizon_tokens=16,
                max_edits=4,
            )

        self.assertGreater(diagnostics["wait_ratio"], 0.0)
        self.assertFalse(diagnostics["wait_feedback_active"])
        self.assertEqual(telemetry_replays, 0)

    def test_diff_tree_scores_cut_multi_transition_as_atomic_prefix(self):
        cost = GranularityCostModel({
            "profiles": {"cpu": {
                "compute_ms_per_mib": 1.0,
                "io_request_ms": 0.0,
                "prepare_launch_ms": 0.0,
                "compute_launch_ms": 0.0,
                "transition_fixed_ms": 0.5,
                "transition_ms_per_mib": 0.01,
                "hysteresis_ms": 0.0,
                # Tensor is an intentionally poor intermediate. The complete
                # Multi -> Tensor -> Cut path is still profitable.
                "mode_compute_scale": {
                    "multi": 1.0, "tensor": 10.0, "cut": 0.5},
            }}
        }, "cpu")
        weights = normalize_weights(model_meta(), plan(True), "cpu")
        current, _ = fixed_partition(weights, cost, "multi")
        target, _ = fixed_partition(weights, cost, "cut")
        chosen, _, diagnostics = diff_tree_partition(
            current,
            target,
            weights,
            cost,
            horizon_tokens=8,
            max_edits=6,
        )

        self.assertEqual(chosen, target)
        self.assertEqual(diagnostics["offline_distance_after"], 0)
        self.assertGreater(diagnostics["horizon_net_gain_ms"], 0.0)
        # Four unique weights (36 MiB total) are regrouped atomically. The
        # fixed 0.5 ms switch cost is charged once, not once per Cut<-Tensor
        # <-Multi tree edge.
        self.assertAlmostEqual(
            diagnostics["switch_cost_ms"], 0.5 + 36 * 0.01)
        self.assertAlmostEqual(
            diagnostics["edits"][-1]["prefix_switch_cost_ms"],
            diagnostics["switch_cost_ms"])
        self.assertTrue(any(
            edit["steady_gain_ms"] < 0.0
            for edit in diagnostics["edits"]))

    def test_diff_tree_caches_costs_across_iff_prefixes(self):
        cost = GranularityCostModel({
            "profiles": {"cpu": {
                "compute_ms_per_mib": 1.0,
                "io_request_ms": 0.0,
                "prepare_launch_ms": 0.0,
                "compute_launch_ms": 0.0,
                "transition_fixed_ms": 0.0,
                "transition_ms_per_mib": 0.0,
                "hysteresis_ms": 0.0,
                "mode_compute_scale": {
                    "multi": 1.0, "tensor": 2.0, "cut": 0.5},
            }}
        }, "cpu")
        weights = normalize_weights(model_meta(), plan(True), "cpu")
        current, _ = fixed_partition(weights, cost, "multi")
        target, _ = fixed_partition(weights, cost, "cut")

        with patch.object(
            cost,
            "decision_costs",
            wraps=cost.decision_costs,
        ) as decision_costs:
            chosen, _, diagnostics = diff_tree_partition(
                current,
                target,
                weights,
                cost,
                horizon_tokens=8,
                max_edits=6,
            )

        self.assertEqual(chosen, target)
        self.assertEqual(diagnostics["offline_distance_after"], 0)
        # Four Tensor, four Cut and at most three adjacent Multi decisions
        # exist for this graph. Prefix replay must not re-evaluate any one of
        # those immutable decisions on every IFF-tree edge.
        self.assertLessEqual(decision_costs.call_count, 11)

    def test_diff_tree_bootstraps_at_refined_offline_partition(self):
        cost = GranularityCostModel({
            "profiles": {"cpu": {
                "mode_prepare_scale": {
                    "multi": 1.0, "tensor": 1.0, "cut": 0.2},
                "mode_compute_scale": {
                    "multi": 0.7, "tensor": 1.0, "cut": 1.2},
            }}
        }, "cpu")
        output = attach_working_unit(
            model_meta(), plan(False), cost,
            policy="diff-tree", current_working_unit=None,
            max_edits=1)
        diagnostics = output["cost_model"]["working_unit"]
        self.assertTrue(diagnostics["bootstrap_offline"])
        self.assertEqual(diagnostics["edit_count"], 0)
        self.assertFalse(diagnostics["edits_truncated"])
        self.assertEqual(output["working_unit"]["switch_cost_ms"], 0.0)
        weights = normalize_weights(model_meta(), output, "cpu")
        offline, _ = offline_partition(weights, cost)
        self.assertEqual(
            working_unit_to_partition(
                output["working_unit"], weights),
            offline)

    def test_diff_tree_reuses_precomputed_offline_target(self):
        cost = GranularityCostModel({
            "profiles": {"cpu": {
                "mode_prepare_scale": {
                    "multi": 1.0, "tensor": 1.0, "cut": 0.2},
                "mode_compute_scale": {
                    "multi": 0.7, "tensor": 1.0, "cut": 1.2},
            }}
        }, "cpu")
        offline_plan = attach_working_unit(
            model_meta(), plan(False), cost, policy="offline")
        with patch(
            "runtime.plan.super_tensor_planner.cached_offline_partition",
            side_effect=AssertionError("online beam search must be skipped"),
        ):
            output = attach_working_unit(
                model_meta(), plan(False), cost,
                policy="diff-tree", current_working_unit=None,
                offline_working_unit=offline_plan["working_unit"])
        diagnostics = output["cost_model"]["working_unit"]
        self.assertEqual(
            diagnostics["offline_target_source"],
            "precomputed-budget-table")
        weights = normalize_weights(model_meta(), output, "cpu")
        self.assertEqual(
            working_unit_to_partition(
                output["working_unit"], weights),
            working_unit_to_partition(
                offline_plan["working_unit"], weights))

    def test_online_can_replace_the_full_frontier(self):
        cost = GranularityCostModel({
            "profiles": {"cpu": {
                "transition_fixed_ms": 0.0,
                "transition_ms_per_mib": 0.0,
                "hysteresis_ms": 0.0,
                "mode_prepare_scale": {
                    "multi": 1.0, "tensor": 1.0, "cut": 0.05},
                "mode_compute_scale": {
                    "multi": 1.0, "tensor": 1.0, "cut": 1.01},
            }}
        }, "cpu")
        weights = normalize_weights(model_meta(), plan(False), "cpu")
        current, _ = fixed_partition(weights, cost, "tensor")
        target, _ = offline_partition(weights, cost)
        chosen, _, diagnostics = online_partition(
            current, target, weights, cost, horizon_tokens=32)
        self.assertTrue(diagnostics["accepted"])
        self.assertEqual(chosen, target)
        self.assertEqual(diagnostics["policy"], "online-global-mixed")

    def test_online_skips_edit_replay_for_constant_switch_cost(self):
        cost = GranularityCostModel({
            "profiles": {"cpu": {
                "transition_fixed_ms": 0.02,
                "transition_ms_per_mib": 0.0,
                "hysteresis_ms": 0.0,
                "wait_feedback_weight": 0.0,
                "mode_prepare_scale": {
                    "multi": 1.0, "tensor": 1.0, "cut": 0.05},
                "mode_compute_scale": {
                    "multi": 1.0, "tensor": 1.0, "cut": 1.01},
            }}
        }, "cpu")
        weights = normalize_weights(model_meta(), plan(False), "cpu")
        current, _ = fixed_partition(weights, cost, "tensor")
        target, _ = offline_partition(weights, cost)
        with patch(
            "runtime.plan.super_tensor_planner.diff_tree_partition",
            side_effect=AssertionError(
                "constant-cost Online must not replay local edits"),
        ):
            chosen, _, diagnostics = online_partition(
                current, target, weights, cost, horizon_tokens=32)
        self.assertEqual(chosen, target)
        self.assertTrue(diagnostics["constant_switch_fast_path"])
        self.assertGreater(diagnostics["optimizer_edit_count"], 0)

    def test_online_scores_slow_tensor_intermediate_atomically(self):
        cost = GranularityCostModel({
            "profiles": {"cpu": {
                "compute_ms_per_mib": 1.0,
                "io_request_ms": 0.0,
                "prepare_launch_ms": 0.0,
                "compute_launch_ms": 0.0,
                "transition_fixed_ms": 0.0,
                "hysteresis_ms": 0.0,
                "mode_compute_scale": {
                    "multi": 1.0, "tensor": 10.0, "cut": 0.5},
            }}
        }, "cpu")
        weights = normalize_weights(model_meta(), plan(True), "cpu")
        current, _ = fixed_partition(weights, cost, "multi")
        target, _ = fixed_partition(weights, cost, "cut")
        chosen, _, diagnostics = online_partition(
            current, target, weights, cost, horizon_tokens=8)

        self.assertEqual(chosen, target)
        self.assertTrue(diagnostics["accepted"])
        self.assertEqual(diagnostics["offline_distance_after"], 0)
        self.assertGreater(diagnostics["optimizer_edit_count"], 0)

    def test_attach_online_reports_global_mixed_policy(self):
        cost = GranularityCostModel({
            "profiles": {"cpu": {
                "transition_fixed_ms": 0.0,
                "hysteresis_ms": 0.0,
                "mode_prepare_scale": {
                    "multi": 1.0, "tensor": 1.0, "cut": 0.05},
                "mode_compute_scale": {
                    "multi": 1.0, "tensor": 1.0, "cut": 1.01},
            }}
        }, "cpu")
        current = attach_working_unit(
            model_meta(), plan(False), cost, policy="fixed-tensor")
        output = attach_working_unit(
            model_meta(), plan(False), cost, policy="online",
            current_working_unit=current["working_unit"],
            horizon_tokens=32)
        self.assertEqual(
            output["working_unit"]["policy"], "online-global-mixed")
        self.assertEqual(
            output["cost_model"]["working_unit"]["policy"],
            "online-global-mixed")

    def test_online_reuses_precomputed_global_frontier(self):
        cost = GranularityCostModel({
            "profiles": {"cpu": {
                "transition_fixed_ms": 0.0,
                "hysteresis_ms": 0.0,
                "mode_prepare_scale": {
                    "multi": 1.0, "tensor": 1.0, "cut": 0.05},
                "mode_compute_scale": {
                    "multi": 1.0, "tensor": 1.0, "cut": 1.01},
            }}
        }, "cpu")
        offline_plan = attach_working_unit(
            model_meta(), plan(False), cost, policy="offline")
        current = attach_working_unit(
            model_meta(), plan(False), cost, policy="fixed-tensor")
        with patch(
            "runtime.plan.super_tensor_planner.cached_offline_partition",
            side_effect=AssertionError(
                "precomputed Online must not repeat the beam search"),
        ):
            output = attach_working_unit(
                model_meta(), plan(False), cost, policy="online",
                current_working_unit=current["working_unit"],
                offline_working_unit=offline_plan["working_unit"],
                horizon_tokens=32)
        diagnostics = output["cost_model"]["working_unit"]
        self.assertEqual(
            diagnostics["offline_target_source"],
            "precomputed-budget-table")
        self.assertEqual(
            diagnostics["policy"], "online-global-mixed")

    def test_max_plus_local_score_matches_full_pipeline(self):
        cost = GranularityCostModel({
            "profiles": {"cpu": {
                "io_request_ms": 0.07,
                "prepare_launch_ms": 0.11,
                "compute_launch_ms": 0.13,
            }}
        }, "cpu")
        weights = normalize_weights(model_meta(), plan(False), "cpu")
        decisions, _ = fixed_partition(weights, cost, "tensor")
        by_id = {weight.weight_id: weight for weight in weights}
        prefix, suffix = _partition_prefix_suffix(
            decisions, by_id, cost)
        for edit in local_edits(decisions, by_id, cost):
            fast = _edited_pipeline_end(
                edit, prefix, suffix, by_id, cost)
            exact = simulate_partition(
                apply_local_edit(decisions, edit), weights, cost
            ).compute_end_ms
            self.assertAlmostEqual(fast, exact, places=9)


if __name__ == "__main__":
    unittest.main()
