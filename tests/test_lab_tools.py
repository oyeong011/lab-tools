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
            "--run",
            "--require-provenance",
            "--require-opencl-device",
            "Intel",
        ]).stdout
        self.assertIn("DRY remote collect: lab-acceptance-collect --profile cpu --run --require-provenance --require-opencl-device Intel", out)
        out = self.run_cmd([
            "bin/lab-acceptance-collect",
            "--dry-run",
            "--profile",
            "apple",
            "--run",
            "--require-apple-chip",
            "Apple M1",
        ]).stdout
        self.assertIn("DRY lab-acceptance-verify <acceptance_dir> --expect-profile apple --require-run --require-apple-chip Apple M1", out)
        out = self.run_cmd([
            "bin/lab-acceptance-matrix",
            "--dry-run",
            "--target",
            "ubuntu-rtx5060-8gb",
        ]).stdout
        self.assertIn("DRY ubuntu-rtx5060-8gb: lab-acceptance-bundle --check-bundle '<bundle.tar.gz>' --expect-profile cuda --require-run --require-uvm-profile --require-provenance --require-gpu-name 'RTX 5060' --min-gpu-memory-mib 7600 --require-compute-cap 12.0 --require-cuda-sm 120", out)
        out = self.run_cmd(["bash", "-c", "LAB_CUDA_ARCH=sm_120 bin/lab-cuda-arch-flags --verbose"]).stdout
        self.assertIn("cuda_compute_cap=12.0", out)
        self.assertIn("cuda_sm=120", out)
        with tempfile.TemporaryDirectory() as td:
            artifact = Path(td)
            for name in ["profile", "doctor", "matrix-validate", "baseline-config", "pipeline-cpu-plan", "rtx-smoke-dry", "forest-uvm-config", "memory-kernels-config", "memory-kernels-sweep-plan", "rtx-smoke-run", "uvm-profile-small"]:
                (artifact / f"{name}.log").write_text("ok\n")
            (artifact / "rtx-smoke-dry.log").write_text(
                "== NVIDIA device ==\n"
                "GPU 0: NVIDIA GeForce RTX 5060 (UUID: GPU-test)\n"
                "== CUDA toolchain ==\n"
                "cuda_compute_cap=12.0\n"
                "cuda_sm=120\n"
                "cuda_nvcc_flags=-gencode=arch=compute_120,code=sm_120 -gencode=arch=compute_120,code=compute_120\n"
                "index, name, compute_cap, driver_version, cuda_version, memory.total [MiB]\n"
                "0, NVIDIA GeForce RTX 5060, 12.0, 580.88, 13.0, 8188 MiB\n"
                "name, driver_version, cuda_version, memory.total [MiB], power.limit [W], persistence_mode\n"
                "NVIDIA GeForce RTX 5060, 580.88, 13.0, 8188 MiB, 145.00 W, Enabled\n"
            )
            profiles = artifact / "profiles"
            profiles.mkdir()
            (profiles / "uvm-ls-512mb-test.nsys-rep").write_text("fake nsys report\n")
            stats = profiles / "uvm-ls-512mb-test-stats"
            stats.mkdir()
            (stats / "um_sum.csv").write_text("Metric,Value\nCPU Page Faults,1\n")
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
                "--require-compute-cap",
                "12.0",
                "--require-cuda-sm",
                "120",
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
                "--require-compute-cap",
                "12.0",
                "--require-cuda-sm",
                "120",
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
                "--require-compute-cap",
                "12.0",
                "--require-cuda-sm",
                "120",
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
                "--require-compute-cap",
                "12.0",
                "--require-cuda-sm",
                "120",
            ]).stdout
            self.assertIn("ok acceptance", out)
            cpu_artifact = artifact / "cpu-artifact"
            cpu_artifact.mkdir()
            for name in ["profile", "doctor", "matrix-validate", "baseline-config", "pipeline-cpu-plan", "cpu-quick-config", "cpu-smoke-run"]:
                (cpu_artifact / f"{name}.log").write_text("ok\n")
            (cpu_artifact / "doctor.log").write_text(
                "== OpenCL devices ==\n"
                "Platform #0: Intel(R) OpenCL HD Graphics\n"
                "`-- Device #0: Intel(R) UHD Graphics 770\n"
            )
            cpu_steps = [
                {"name": name, "status": "ok", "exit_code": "0", "log": f"{name}.log", "command": name}
                for name in ["profile", "doctor", "matrix-validate", "baseline-config", "pipeline-cpu-plan", "cpu-quick-config", "cpu-smoke-run"]
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
                "--require-run",
                "--require-opencl-device",
                "Intel",
            ]).stdout
            self.assertIn("ok acceptance", out)
            bad_cpu = artifact / "bad-cpu-artifact"
            bad_cpu.mkdir()
            for name in ["profile", "doctor", "matrix-validate", "baseline-config", "pipeline-cpu-plan", "cpu-quick-config", "cpu-smoke-run"]:
                (bad_cpu / f"{name}.log").write_text("ok\n")
            (bad_cpu / "doctor.log").write_text(
                "== System tools ==\n"
                "warn command clinfo missing\n"
                "Intel package text only\n"
                "== OpenCL devices ==\n"
                "warn no OpenCL device detected\n"
            )
            (bad_cpu / "acceptance.json").write_text(json.dumps({
                "schema_version": 1,
                "profile": "cpu",
                "status": "ok",
                "failures": 0,
                "steps": cpu_steps,
            }))
            env = os.environ.copy()
            env["PATH"] = f"{BIN}:{env.get('PATH', '')}"
            result = subprocess.run(
                ["bin/lab-acceptance-verify", str(bad_cpu), "--expect-profile", "cpu", "--require-opencl-device", "Intel"],
                cwd=REPO,
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("OpenCL device requirement not met", result.stdout)
            apple_artifact = artifact / "apple-artifact"
            apple_artifact.mkdir()
            for name in ["profile", "doctor", "matrix-validate", "baseline-config", "pipeline-cpu-plan", "apple-smoke-dry", "apple-smoke-run"]:
                (apple_artifact / f"{name}.log").write_text("ok\n")
            (apple_artifact / "apple-smoke-dry.log").write_text(
                "== System ==\n"
                "Darwin test-host arm64\n"
                "Apple M1\n"
                "Chipset Model: Apple M1\n"
            )
            apple_steps = [
                {"name": name, "status": "ok", "exit_code": "0", "log": f"{name}.log", "command": name}
                for name in ["profile", "doctor", "matrix-validate", "baseline-config", "pipeline-cpu-plan", "apple-smoke-dry", "apple-smoke-run"]
            ]
            (apple_artifact / "acceptance.json").write_text(json.dumps({
                "schema_version": 1,
                "profile": "apple",
                "status": "ok",
                "failures": 0,
                "steps": apple_steps,
            }))
            out = self.run_cmd([
                "bin/lab-acceptance-verify",
                str(apple_artifact),
                "--expect-profile",
                "apple",
                "--require-run",
                "--require-apple-chip",
                "Apple M1",
            ]).stdout
            self.assertIn("ok acceptance", out)
            apple_bundle_dir = artifact.parent / "apple-bundles"
            out = self.run_cmd([
                "bin/lab-acceptance-bundle",
                str(apple_artifact),
                "--out",
                str(apple_bundle_dir),
                "--expect-profile",
                "apple",
                "--require-run",
                "--require-apple-chip",
                "Apple M1",
            ]).stdout
            apple_bundle = Path(next(line for line in out.splitlines() if line.startswith("bundle=")).split("=", 1)[1])
            self.assertTrue(apple_bundle.exists())
            matrix_config = artifact / "matrix.json"
            matrix_config.write_text(json.dumps({
                "schema_version": 1,
                "targets": [
                    {
                        "name": "apple-m1",
                        "profile": "apple",
                        "require_run": True,
                        "require_apple_chip": ["Apple M1"],
                    },
                    {
                        "name": "apple-m4",
                        "profile": "apple",
                        "require_run": True,
                        "require_apple_chip": ["Apple M4"],
                    },
                ],
            }))
            out = self.run_cmd([
                "bin/lab-acceptance-matrix",
                "--config",
                str(matrix_config),
                "--bundle-dir",
                str(apple_bundle_dir),
                "--target",
                "apple-m1",
            ]).stdout
            self.assertIn("complete=yes", out)
            stage_dir = artifact / "drive-stage"
            out = self.run_cmd([
                "bin/lab-acceptance-stage",
                "--config",
                str(matrix_config),
                "--source-dir",
                str(apple_bundle_dir),
                "--out",
                str(stage_dir),
            ]).stdout
            self.assertIn("stage_dir=", out)
            self.assertTrue((stage_dir / apple_bundle.name).exists())
            self.assertTrue((stage_dir / f"{apple_bundle.name}.sha256").exists())
            stage_manifest = json.loads((stage_dir / "STAGE-MANIFEST.json").read_text())
            self.assertEqual(stage_manifest["mode"], "latest-passing-per-target")
            self.assertFalse(stage_manifest["complete"])
            self.assertIn("apple-m4", stage_manifest["missing_targets"])
            env = os.environ.copy()
            env["PATH"] = f"{BIN}:{env.get('PATH', '')}"
            result = subprocess.run(
                [
                    "bin/lab-acceptance-matrix",
                    "--config",
                    str(matrix_config),
                    "--bundle-dir",
                    str(apple_bundle_dir),
                    "--target",
                    "apple-m4",
                ],
                cwd=REPO,
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("missing_targets=apple-m4", result.stdout)

    def test_install_dry_run_and_handoff_are_portable(self):
        with tempfile.TemporaryDirectory() as td:
            env = os.environ.copy()
            env["PATH"] = f"{BIN}:{env.get('PATH', '')}"
            env["HOME"] = td
            result = subprocess.run(
                ["bash", "bin/lab-tools-install", "--dry-run"],
                cwd=REPO,
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                check=True,
            )
            self.assertIn(".config/lab/acceptance", result.stdout)
            self.assertFalse((Path(td) / "bin").exists())
            self.assertFalse((Path(td) / ".config" / "lab").exists())
            subprocess.run(
                ["bash", "bin/lab-tools-install"],
                cwd=REPO,
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                check=True,
            )
            installed_env = env.copy()
            installed_env["PATH"] = f"{Path(td) / 'bin'}:{installed_env.get('PATH', '')}"
            installed_env["LAB_PROFILE"] = "cpu"
            yaml_parent = subprocess.check_output(
                ["python3", "-c", "import pathlib, yaml; print(pathlib.Path(yaml.__file__).parents[1])"],
                cwd=REPO,
                text=True,
            ).strip()
            installed_env["PYTHONPATH"] = f"{yaml_parent}:{installed_env.get('PYTHONPATH', '')}"
            subprocess.run(
                ["lab-host-acceptance", "--dry-run"],
                cwd="/tmp",
                env=installed_env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                check=True,
            )
            subprocess.run(
                ["lab-validate", "matrix", str(Path(td) / ".config" / "lab" / "pipelines" / "research-matrix.yaml")],
                cwd="/tmp",
                env=installed_env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                check=True,
            )
            subprocess.run(
                ["bench-suite-config", "--dry-run", "baseline.yaml"],
                cwd="/tmp",
                env=installed_env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                check=True,
            )
        handoff = (BIN / "lab-handoff").read_text()
        remote = (BIN / "lab-remote-acceptance").read_text()
        self.assertIn("lab-tools-install", handoff)
        self.assertIn("lab-acceptance-bundle", handoff)
        self.assertIn("lab-acceptance-collect", handoff)
        self.assertIn("lab-acceptance-matrix", handoff)
        self.assertIn("lab-acceptance-stage", handoff)
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
            self.assertTrue(any(name.endswith("/bin/lab-acceptance-matrix") for name in names))
            self.assertTrue(any(name.endswith("/bin/lab-acceptance-stage") for name in names))
            self.assertTrue(any(name.endswith("/bin/lab-remote-acceptance") for name in names))
            self.assertTrue(any(name.endswith("/config/lab/acceptance/required-hosts.json") for name in names))
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
