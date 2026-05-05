import csv
import json
import os
import py_compile
import subprocess
import tarfile
import tempfile
import unittest
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
BIN = REPO / "bin"


class LabToolsSmokeTests(unittest.TestCase):
    def run_cmd(self, args, **kwargs):
        env = os.environ.copy()
        env["PATH"] = f"{BIN}:{env.get('PATH', '')}"
        return subprocess.run(
            args,
            cwd=REPO,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=True,
            **kwargs,
        )

    def test_scripts_parse(self):
        for script in sorted(BIN.iterdir()):
            if not script.is_file():
                continue
            first = script.read_text(errors="replace").splitlines()[0]
            if "python" in first:
                py_compile.compile(str(script), doraise=True)
            elif "bash" in first or "sh" in first:
                self.run_cmd(["bash", "-n", str(script)])

    def test_matrix_and_suite_configs_validate(self):
        self.run_cmd(["bin/lab-validate", "matrix", "config/pipelines/research-matrix.yaml"])
        self.run_cmd(["bin/lab-validate", "suite-config", "config/suites/baseline.yaml"])
        self.run_cmd(["bin/lab-validate", "suite-config", "config/suites/forest-uvm.yaml"])
        self.run_cmd(["bin/lab-validate", "suite-config", "config/suites/memory-kernels.yaml"])

    def test_pipeline_dry_runs(self):
        out = self.run_cmd(["bin/lab-pipeline", "list"]).stdout
        self.assertIn("forest-uvm-access", out)
        out = self.run_cmd([
            "bin/lab-pipeline",
            "plan",
            "consumer-accelerator-baseline",
            "--profile",
            "cpu",
            "--dry-run",
        ]).stdout
        self.assertIn("bench-suite-config", out)
        out = self.run_cmd([
            "bin/lab-pipeline",
            "plan",
            "forest-uvm-access",
            "--profile",
            "cuda",
            "--sweep",
            "--dry-run",
        ]).stdout
        self.assertIn("uvm_pattern-ls", out)
        self.assertIn("uvm_pattern-hchi", out)
        out = self.run_cmd([
            "bin/lab-pipeline",
            "plan",
            "memory-hierarchy-pim",
            "--profile",
            "cuda",
            "--dry-run",
        ]).stdout
        self.assertIn("cuda-gemv-gemv_size", out)
        self.assertIn("gcn_rows-16384", out)
        out = self.run_cmd([
            "bin/lab-pipeline",
            "plan",
            "memory-hierarchy-pim",
            "--profile",
            "cuda",
            "--sweep",
            "--dry-run",
        ]).stdout
        commands = [line for line in out.splitlines() if line.startswith("  bench-suite-config ")]
        self.assertEqual(len(commands), 30)
        self.assertIn("cuda-gcn-gcn_rows-16384-gcn_degree-8-gcn_feature_dim-16", out)
        self.assertNotIn("gemv_size-1024-spmv_rows", out)

    def test_bench_suite_config_dry_run(self):
        out = self.run_cmd(["bin/bench-suite-config", "--dry-run", "config/suites/forest-uvm.yaml"]).stdout
        self.assertIn("Dry-run: yes", out)
        self.assertIn("LAB_WORKLOADS=cuda-uvm", out)
        self.assertIn("LAB_UVM_PATTERN=ls", out)
        named = self.run_cmd(["bin/bench-suite-config", "--dry-run", "forest-uvm.yaml"]).stdout
        self.assertIn("config/suites/forest-uvm.yaml", named)
        self.assertIn("LAB_WORKLOADS=cuda-uvm", named)
        out = self.run_cmd(["bin/lab-host-acceptance", "--dry-run"]).stdout
        self.assertIn("Acceptance dry-run", out)
        self.assertIn("doctor", out)
        out = self.run_cmd([
            "bin/lab-acceptance-collect",
            "--dry-run",
            "--profile",
            "cuda",
            "--run",
            "--uvm-profile",
            "--require-gpu-name",
            "RTX 5060",
            "--min-gpu-memory-mib",
            "8000",
        ]).stdout
        self.assertIn("DRY lab-host-acceptance --run --uvm-profile", out)
        self.assertIn("--require-gpu-name RTX 5060 --min-gpu-memory-mib 8000", out)
        self.assertIn("DRY lab-acceptance-bundle --check-bundle", out)
        out = self.run_cmd([
            "bin/lab-remote-acceptance",
            "user@rtx-host",
            "--dry-run",
            "--profile",
            "cuda",
            "--run",
            "--uvm-profile",
            "--require-gpu-name",
            "RTX 5060",
            "--min-gpu-memory-mib",
            "8000",
        ]).stdout
        self.assertIn("DRY remote collect: lab-acceptance-collect --profile cuda --run --uvm-profile --require-gpu-name RTX\\ 5060 --min-gpu-memory-mib 8000", out)
        self.assertIn("DRY local verify: lab-acceptance-bundle --check-bundle", out)
        out = self.run_cmd([
            "bin/lab-remote-acceptance",
            "user@intel-host",
            "--dry-run",
            "--profile",
            "cpu",
            "--require-opencl-device",
            "Intel",
        ]).stdout
        self.assertIn("DRY remote collect: lab-acceptance-collect --profile cpu --require-opencl-device Intel", out)
        with tempfile.TemporaryDirectory() as td:
            artifact = Path(td)
            for name in ["profile", "doctor", "matrix-validate", "baseline-config", "pipeline-cpu-plan", "rtx-smoke-dry", "forest-uvm-config", "memory-kernels-config", "memory-kernels-sweep-plan", "rtx-smoke-run", "uvm-profile-small"]:
                (artifact / f"{name}.log").write_text("ok\n")
            (artifact / "rtx-smoke-dry.log").write_text(
                "== NVIDIA device ==\n"
                "GPU 0: NVIDIA GeForce RTX 5060 (UUID: GPU-test)\n"
                "name, driver_version, cuda_version, memory.total [MiB], power.limit [W], persistence_mode\n"
                "NVIDIA GeForce RTX 5060, 580.88, 13.0, 8188 MiB, 145.00 W, Enabled\n"
            )
            profiles = artifact / "profiles"
            profiles.mkdir()
            (profiles / "uvm-ls-512mb-test.nsys-rep").write_text("fake nsys report\n")
            steps = [
                {"name": name, "status": "ok", "exit_code": "0", "log": f"{name}.log", "command": name}
                for name in ["profile", "doctor", "matrix-validate", "baseline-config", "pipeline-cpu-plan", "rtx-smoke-dry", "forest-uvm-config", "memory-kernels-config", "memory-kernels-sweep-plan", "rtx-smoke-run", "uvm-profile-small"]
            ]
            (artifact / "acceptance.json").write_text(json.dumps({
                "schema_version": 1,
                "profile": "cuda",
                "status": "ok",
                "failures": 0,
                "steps": steps,
            }))
            out = self.run_cmd([
                "bin/lab-acceptance-verify",
                str(artifact),
                "--expect-profile",
                "cuda",
                "--require-run",
                "--require-uvm-profile",
                "--require-gpu-name",
                "RTX 5060",
                "--min-gpu-memory-mib",
                "8000",
            ]).stdout
            self.assertIn("ok acceptance", out)
            bundle_dir = artifact.parent / f"{artifact.name}-bundles"
            out = self.run_cmd([
                "bin/lab-acceptance-bundle",
                str(artifact),
                "--out",
                str(bundle_dir),
                "--expect-profile",
                "cuda",
                "--require-run",
                "--require-uvm-profile",
                "--require-gpu-name",
                "RTX 5060",
                "--min-gpu-memory-mib",
                "8000",
            ]).stdout
            bundle_line = next(line for line in out.splitlines() if line.startswith("bundle="))
            bundle_path = Path(bundle_line.split("=", 1)[1])
            self.assertTrue(bundle_path.exists())
            self.assertTrue(Path(str(bundle_path) + ".sha256").exists())
            out = self.run_cmd([
                "bin/lab-acceptance-bundle",
                "--check-bundle",
                str(bundle_path),
                "--expect-profile",
                "cuda",
                "--require-run",
                "--require-uvm-profile",
                "--require-gpu-name",
                "RTX 5060",
                "--min-gpu-memory-mib",
                "8000",
            ]).stdout
            self.assertIn("ok acceptance bundle", out)
            extract_dir = artifact / "extract"
            extract_dir.mkdir()
            with tarfile.open(bundle_path, "r:gz") as tar:
                tar.extractall(extract_dir)
            extracted_artifacts = list(extract_dir.glob("*/acceptance/*"))
            self.assertEqual(len(extracted_artifacts), 1)
            out = self.run_cmd([
                "bin/lab-acceptance-verify",
                str(extracted_artifacts[0]),
                "--expect-profile",
                "cuda",
                "--require-run",
                "--require-uvm-profile",
                "--require-gpu-name",
                "RTX 5060",
                "--min-gpu-memory-mib",
                "8000",
            ]).stdout
            self.assertIn("ok acceptance", out)
            cpu_artifact = artifact / "cpu-artifact"
            cpu_artifact.mkdir()
            for name in ["profile", "doctor", "matrix-validate", "baseline-config", "pipeline-cpu-plan", "cpu-quick-config"]:
                (cpu_artifact / f"{name}.log").write_text("ok\n")
            (cpu_artifact / "doctor.log").write_text(
                "== OpenCL devices ==\n"
                "Platform #0: Intel(R) OpenCL HD Graphics\n"
                "`-- Device #0: Intel(R) UHD Graphics 770\n"
            )
            cpu_steps = [
                {"name": name, "status": "ok", "exit_code": "0", "log": f"{name}.log", "command": name}
                for name in ["profile", "doctor", "matrix-validate", "baseline-config", "pipeline-cpu-plan", "cpu-quick-config"]
            ]
            (cpu_artifact / "acceptance.json").write_text(json.dumps({
                "schema_version": 1,
                "profile": "cpu",
                "status": "ok",
                "failures": 0,
                "steps": cpu_steps,
            }))
            out = self.run_cmd([
                "bin/lab-acceptance-verify",
                str(cpu_artifact),
                "--expect-profile",
                "cpu",
                "--require-opencl-device",
                "Intel",
            ]).stdout
            self.assertIn("ok acceptance", out)

    def test_install_dry_run_and_handoff_are_portable(self):
        with tempfile.TemporaryDirectory() as td:
            env = os.environ.copy()
            env["PATH"] = f"{BIN}:{env.get('PATH', '')}"
            env["HOME"] = td
            subprocess.run(
                ["bash", "bin/lab-tools-install", "--dry-run"],
                cwd=REPO,
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                check=True,
            )
            self.assertFalse((Path(td) / "bin").exists())
            self.assertFalse((Path(td) / ".config" / "lab").exists())
        handoff = (BIN / "lab-handoff").read_text()
        remote = (BIN / "lab-remote-acceptance").read_text()
        self.assertIn("lab-tools-install", handoff)
        self.assertIn("lab-acceptance-bundle", handoff)
        self.assertIn("lab-acceptance-collect", handoff)
        self.assertIn("lab-remote-acceptance", handoff)
        self.assertIn("*.tar.gz|*.tgz", handoff)
        self.assertIn("export PATH=", handoff)
        self.assertIn("hashlib", handoff)
        self.assertNotIn("sha256sum", handoff)
        self.assertIn('PATH="$SCRIPT_DIR:$HOME/bin:$PATH"', remote)

    def test_summary_stats_and_validation_on_fixture_suite(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            suite = root / "suite"
            suite.mkdir()
            run_dir = root / "runs" / "r1"
            run_dir.mkdir(parents=True)
            gcn_run_dir = root / "runs" / "gcn1"
            gcn_run_dir.mkdir(parents=True)
            (suite / "execution-order.csv").write_text(
                "order,workload,repeat\n"
                "1,opencl,1\n"
                "2,cuda-gcn,1\n"
            )
            (suite / "opencl-run-1.log").write_text(
                "Kernel time: 0.125 s\n"
                "Device approx bandwidth: 12.5 GB/s\n"
                "Max error: 0\n"
                f"Run complete: {run_dir}\n"
            )
            (suite / "cuda-gcn-run-1.log").write_text(
                "Kernel: cuda-gcn-agg\n"
                "Kernel time: 0.250 s\n"
                "Device approx bandwidth: 25.0 GB/s\n"
                "Device approx GFLOP/s: 50.0\n"
                "Max error: 0\n"
                f"Run complete: {gcn_run_dir}\n"
            )
            (run_dir / "result.json").write_text(json.dumps({
                "started_at": "2026-01-01T00:00:00+00:00",
                "ended_at": "2026-01-01T00:00:01+00:00",
                "exit_code": 0,
                "status": "ok",
                "duration_s": 1.0,
                "energy_j": {"pkg_energy_j": 2.0},
                "avg_power_w": {"pkg_power_w": 2.0},
                "thermal_event_count": 0,
                "system_pinned": "yes",
            }))
            (run_dir / "manifest.json").write_text(json.dumps({
                "schema_version": 1,
                "system": {},
                "tool_versions": {},
                "file_sha256": {},
            }))
            (gcn_run_dir / "result.json").write_text(json.dumps({
                "started_at": "2026-01-01T00:00:00+00:00",
                "ended_at": "2026-01-01T00:00:01+00:00",
                "exit_code": 0,
                "status": "ok",
                "duration_s": 1.0,
            }))
            (gcn_run_dir / "manifest.json").write_text(json.dumps({
                "schema_version": 1,
                "system": {},
                "tool_versions": {},
                "file_sha256": {},
            }))
            self.run_cmd(["bin/summarize-suite", str(suite)])
            env = os.environ.copy()
            env["PATH"] = f"{BIN}:{env.get('PATH', '')}"
            env["LAB_BOOTSTRAP_N"] = "100"
            subprocess.run(
                ["bin/suite-stats", str(suite)],
                cwd=REPO,
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                check=True,
            )
            self.run_cmd(["bin/render-suite-report", str(suite)])
            self.run_cmd(["bin/lab-validate", "suite-dir", str(suite)])
            handoff_target = root / "suite-handoff.tar.gz"
            handoff_out = self.run_cmd([
                "bin/lab-handoff",
                str(suite),
                "--out",
                str(handoff_target),
                "--no-include-runs",
            ]).stdout
            handoff_path = Path(handoff_out.splitlines()[0])
            self.assertTrue(handoff_path.exists())
            self.assertTrue(Path(str(handoff_path) + ".sha256").exists())
            with tarfile.open(handoff_path, "r:gz") as tar:
                names = tar.getnames()
            self.assertTrue(any(name.endswith("/bin/lab-acceptance-bundle") for name in names))
            self.assertTrue(any(name.endswith("/bin/lab-acceptance-collect") for name in names))
            self.assertTrue(any(name.endswith("/bin/lab-remote-acceptance") for name in names))
            with (suite / "summary.csv").open(newline="") as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(rows[0]["workload"], "opencl-vector")
            self.assertEqual(rows[0]["phase"], "cold")
            self.assertEqual(rows[0]["pkg_energy_j"], "2.0")
            self.assertEqual(rows[1]["workload"], "cuda-gcn")
            self.assertEqual(rows[1]["cuda_gcn_gflops"], "50.0")
            self.assertEqual(rows[1]["cuda_gemm_gflops"], "")
            with (suite / "stats.csv").open(newline="") as f:
                stats = list(csv.DictReader(f))
            self.assertTrue(any(r["metric"] == "cuda_gcn_gflops" and r["workload"] == "cuda-gcn" for r in stats))
            report = (suite / "report.md").read_text()
            self.assertIn("CUDA-GCN GF/s", report)
            self.assertIn("| cuda-gcn | 1 | cold | ok |", report)


if __name__ == "__main__":
    unittest.main()
