import os
from pathlib import Path
import shlex
import subprocess
import unittest


REPO_ROOT = Path(__file__).resolve().parents[2]
AUTOCHAIN = REPO_ROOT / "scripts/elastic/run_op12_v21_autochain.sh"


def expanded_runner_args(
    script: str,
    overrides: dict[str, str],
) -> list[str]:
    """Expand one shell wrapper without touching a device.

    The wrappers honor ``ELASTIC_PYTHON``.  Using ``/bin/echo`` exposes the
    exact matrix-runner argv after all shell defaults and v21 overrides have
    been applied.
    """
    env = os.environ.copy()
    env.update({
        "ELASTIC_PYTHON": "/bin/echo",
        "ELASTIC_ADB_SERIAL": "5ae7a43d",
        "ELASTIC_EXPECTED_BINARY_SHA256": "b" * 64,
        "ELASTIC_EXPECTED_MODEL_SHA256": "a" * 64,
        "ELASTIC_PLAN_RESIDENCY_POLICY": "strict",
        "ELASTIC_PLANNER_STREAM_RESERVE_MIB": "128",
        "ELASTIC_GRANULARITY_PLACEMENT_SOURCE": "stateful-cp",
        "ELASTIC_PIPELINE_COPY_CPU": "1",
    })
    env.update(overrides)
    proc = subprocess.run(
        ["bash", script],
        cwd=REPO_ROOT,
        env=env,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    return shlex.split(proc.stdout.strip())


def option_rows(argv: list[str]) -> list[tuple[str, str | None]]:
    rows: list[tuple[str, str | None]] = []
    index = 1  # matrix-runner script path
    while index < len(argv):
        option = argv[index]
        if option.startswith("--"):
            if index + 1 < len(argv) and not argv[index + 1].startswith("--"):
                rows.append((option, argv[index + 1]))
                index += 2
            else:
                rows.append((option, None))
                index += 1
        else:
            rows.append(("<positional>", option))
            index += 1
    return rows


def common_runtime_contract(argv: list[str]) -> list[tuple[str, str | None]]:
    intentionally_distinct = {
        "--artifact-root",
        "--granularity-profile",
    }
    return sorted(
        row for row in option_rows(argv)
        if row[0] not in intentionally_distinct
    )


class Op12V21AutochainContractTests(unittest.TestCase):
    def test_cpu_diff_before_and_now_share_runtime_contract(self):
        common = {
            "ELASTIC_COST_DIR": "/tmp/cpu-cost",
            "ELASTIC_CPU_THREADS": "5",
            "ELASTIC_CPU_MASK": "0xf8",
            "ELASTIC_PIPELINE_LOAD_CPU": "0",
            "ELASTIC_PIPELINE_PREPARE_CPU": "2",
            "ELASTIC_COOLDOWN_THERMAL_MAX_C": "35",
            "ELASTIC_MIN_CPU_FREQ_LIMIT_KHZ": "1400000",
            "ELASTIC_MIN_CPU_MEAN_FREQ_KHZ": "1550000",
            "ELASTIC_MIN_CPU_MEDIAN_SAMPLE_MIN_KHZ": "1500000",
            "ELASTIC_MIN_BATTERY_PCT": "20",
        }
        diff_now = expanded_runner_args(
            "scripts/elastic/run_op13_cpu_full10min_difftree.sh",
            common | {
                "ELASTIC_METHODS": "diff-tree-mixed",
                "ELASTIC_ARTIFACT_ROOT": "/tmp/cpu-now",
                "ELASTIC_GRANULARITY_PROFILE": "/tmp/full.json",
            },
        )
        diff_before = expanded_runner_args(
            "scripts/elastic/run_op13_cpu_full10min_diff_before.sh",
            common | {
                "ELASTIC_ARTIFACT_ROOT": "/tmp/cpu-before",
                "ELASTIC_GRANULARITY_PROFILE": "/tmp/phase-only.json",
            },
        )
        self.assertEqual(
            common_runtime_contract(diff_now),
            common_runtime_contract(diff_before),
        )
        rows = option_rows(diff_now)
        self.assertIn(("--bench-seconds", "600"), rows)
        self.assertIn(("--execution-backend", "cpu"), rows)
        self.assertIn(("--overlap-model", "pipeline"), rows)
        self.assertIn(("--max-pipeline-budget-violations", "0"), rows)
        self.assertIn(
            ("--extra-env", "GGML_ELASTIC_PIN=token_embd,output"),
            rows,
        )

    def test_gpu_diff_before_and_now_share_runtime_contract(self):
        common = {
            "ELASTIC_COST_DIR": "/tmp/gpu-cost",
            "ELASTIC_COOLDOWN_THERMAL_MAX_C": "42",
            "ELASTIC_MIN_BATTERY_PCT": "20",
            "ELASTIC_PIPELINE_GRAPH_LOOKAHEAD": "1",
        }
        diff_now = expanded_runner_args(
            "scripts/elastic/run_op12_gpu_full10min_difftree.sh",
            common | {
                "ELASTIC_METHODS": "diff-tree-mixed",
                "ELASTIC_ARTIFACT_ROOT": "/tmp/gpu-now",
                "ELASTIC_GRANULARITY_PROFILE": "/tmp/full.json",
            },
        )
        diff_before = expanded_runner_args(
            "scripts/elastic/run_op12_gpu_full10min_diff_before_v21.sh",
            common | {
                "ELASTIC_ARTIFACT_ROOT": "/tmp/gpu-before",
                "ELASTIC_GRANULARITY_PROFILE": "/tmp/phase-only.json",
            },
        )
        self.assertEqual(
            common_runtime_contract(diff_now),
            common_runtime_contract(diff_before),
        )
        rows = option_rows(diff_now)
        self.assertIn(("--bench-seconds", "600"), rows)
        self.assertIn(("--execution-backend", "gpu"), rows)
        self.assertIn(("--overlap-model", "pipeline"), rows)
        self.assertIn(("--max-pipeline-budget-violations", "0"), rows)
        self.assertIn(
            ("--extra-env", "GGML_ELASTIC_CUT_DUAL_COMPUTE=fused"),
            rows,
        )

    def test_cpu_consolidates_before_gpu_and_outputs_are_isolated(self):
        text = AUTOCHAIN.read_text(encoding="utf-8")
        cpu_summary = text.index("--backend cpu")
        gpu_start = text.index(
            "bash scripts/elastic/run_op12_gpu_pin_contract_smoke_v21.sh")
        gpu_summary = text.index("--backend gpu")
        self.assertLess(cpu_summary, gpu_start)
        self.assertLess(gpu_start, gpu_summary)
        self.assertIn('--output-dir "$final_root/cpu"', text)
        self.assertIn('--output-dir "$final_root/gpu"', text)


if __name__ == "__main__":
    unittest.main()
