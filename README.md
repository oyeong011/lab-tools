# lab-tools

Reproducibility-oriented benchmarking framework for CPU and accelerator workloads on commodity Linux and Apple Silicon macOS. Designed for multi-host operation: a *dev/methodology* host, Apple Silicon smoke hosts, Intel iGPU hosts, and CUDA execution hosts with NVIDIA GPUs.

## Profiles

| Profile | Hardware target | What runs |
|---|---|---|
| `cpu`  | CPU + iGPU OpenCL only (no NVIDIA stack) | bench-cpu, bench-standard-cpu, bench-opencl, bench-opencl-gemm |
| `cuda` | CPU + iGPU OpenCL + NVIDIA GPU + CUDA toolkit | everything in `cpu` plus bench-cuda-vector-add, bench-cuda-gemm, optional bench-cuda-uvm-access |
| `apple` | macOS Apple Silicon M1/M4 | lab-apple-smoke, bench-apple-metal, optional bench-apple-mps |

## Research pipeline

`lab-pipeline` adds a reviewer-facing planning layer on top of the existing
suite runner. It maps research themes to supported environments, suite configs,
workloads, tunables, metrics, and claim boundaries.

```bash
lab-pipeline list
lab-pipeline show forest-uvm-access
lab-pipeline review
lab-pipeline plan consumer-accelerator-baseline --profile cpu
lab-pipeline plan forest-uvm-access --profile cuda --sweep
```

The default matrix is installed to
`~/.config/lab/pipelines/research-matrix.yaml`. Current status:

| Track | Status | Purpose |
|---|---|---|
| `consumer-accelerator-baseline` | wired | CPU/OpenCL/CUDA baseline performance, energy, and reproducibility |
| `forest-uvm-access` | wired | Forest-inspired CUDA managed-memory access-pattern probe |
| `memory-hierarchy-pim` | partial | CUDA GEMV/SpMV microbenchmarks for bandwidth-dominated memory kernels |
| `edge-ai-cnn-transformer` | planned | CNN/Transformer accelerator workloads |
| `multi-tenant-migration-storage` | planned | tensor migration, UVM, storage oversubscription |
| `cluster-communication` | planned | NCCL/PXN rail and topology experiments |
| `security-counter-cache` | planned | defensive cache/counter characterization |

Important claim boundary: `bench-cuda-uvm-access` uses `cudaMallocManaged` and
Forest-style access classes (`ls`, `hchi`, `hcli`, `lc`) to probe real hardware
UVM symptoms. It is not an implementation of Forest's modified UVM driver,
access-time tracker, heterogeneous TBNp, or pseudo-LRU eviction. Page-fault,
migration, re-fault, and thrashing claims require CUPTI/Nsight/driver counters
or simulator instrumentation in addition to this harness.

Validate configs and generated suites with:

```bash
lab-validate matrix ~/.config/lab/pipelines/research-matrix.yaml
lab-validate suite-config ~/.config/lab/suites/forest-uvm.yaml
lab-validate suite-dir <suite_dir>
```

Host acceptance artifact:

```bash
lab-host-acceptance                 # readiness logs under ~/lab/_acceptance
lab-host-acceptance --run           # also run host-specific smoke benchmark
lab-host-acceptance --run --uvm-profile   # CUDA hosts: also capture a small Nsight UVM profile
lab-acceptance-verify <acceptance_dir> --expect-profile cuda --require-run --require-uvm-profile
lab-acceptance-bundle <acceptance_dir> --expect-profile cuda --require-run --require-uvm-profile
lab-acceptance-bundle --check-bundle <bundle.tar.gz> --expect-profile cuda --require-run --require-uvm-profile
lab-acceptance-collect --profile cuda --run --uvm-profile --require-provenance --require-gpu-name "RTX 5060" --min-gpu-memory-mib 7600 --require-compute-cap 12.0 --require-cuda-sm 120
lab-remote-acceptance user@rtx-host --profile cuda --run --uvm-profile --require-provenance --require-gpu-name "RTX 5060" --min-gpu-memory-mib 7600 --require-compute-cap 12.0 --require-cuda-sm 120
lab-remote-acceptance user@intel-host --profile cpu --run --require-provenance --require-opencl-device Intel
lab-acceptance-collect --profile apple --run --require-provenance --require-apple-chip "Apple M1"
lab-acceptance-matrix --bundle-dir ~/lab/_acceptance_bundles
lab-acceptance-matrix --bundle-dir ~/lab/_acceptance_bundles --next-commands
lab-acceptance-stage --out ~/lab/_drive_stage/lab-acceptance-bundles
lab-acceptance-import ~/lab/_drive_stage/lab-acceptance-bundles
```

`lab-acceptance-matrix` checks all collected bundles against
`config/acceptance/required-hosts.json`: MacBook M1, MacBook M4, Ubuntu Intel
iGPU, Ubuntu RTX 5060 8GB, and Ubuntu RTX 5080 16GB. Bundles are plain
`.tar.gz` files with `.sha256` sidecars, so they can be moved by `scp`, USB, or
Google Drive as long as the sidecar is kept with the bundle.
`lab-acceptance-stage` copies the latest passing bundle per matrix target,
the acceptance config used for verification, and `STAGE-MANIFEST.json`. The
resulting folder can be uploaded to Google Drive or copied to USB without
dragging along stale bundles or relying on this Mac's local config path.
`lab-acceptance-import` copies a staged folder back into
`~/lab/_acceptance_bundles`, verifies sidecar hashes, and prints the matrix
result using the staged config.
`lab-remote-acceptance` also installs PyYAML in the remote user's Python
environment when needed, because YAML parsing is required before acceptance can
run.

RTX host smoke and UVM mechanism profiling:

```bash
lab-rtx-smoke                 # dry readiness check
lab-rtx-smoke --run           # executes small CUDA UVM/GEMV/SpMV/GCN probes
lab-uvm-profile --pattern hchi --mb 12288 --passes 2
LAB_CUDA_ARCH=sm_120 bench-cuda-gemv
```

`lab-uvm-profile` uses NVIDIA Nsight Systems Unified Memory CPU/GPU page-fault
tracing. NVIDIA documents these as high-overhead tracing options, so keep them
out of normal timing suites and use them to explain mechanisms after locating
interesting UVM cases.

Memory-kernel sweep:

```bash
lab-pipeline plan memory-hierarchy-pim --profile cuda --sweep
bench-suite-config ~/.config/lab/suites/memory-kernels.yaml
```

Apple Silicon smoke:

```bash
lab-apple-smoke
LAB_APPLE_ELEMENTS=1048576 lab-apple-smoke --run
LAB_APPLE_ELEMENTS=1048576 lab-apple-smoke --run --run-mps   # optional PyTorch MPS smoke
```

The profile auto-detects from `nvidia-smi` + `nvcc`. Override:

```bash
lab-profile set cuda          # persistent (writes ~/.config/lab/profile)
lab-profile clear             # back to auto-detect
LAB_PROFILE=cpu lab-doctor    # transient
```

`lab-doctor` and `bench-suite` honor the profile — on `cpu`, CUDA tools are listed as info/optional and CUDA workloads are skipped automatically. On `cuda`, CUDA tools become required and bench-suite includes `cuda-vector` + `cuda-gemm` workloads.

## Install on a fresh host

```bash
git clone <THIS-REPO-URL> ~/lab-tools
cd ~/lab-tools
python3 -m pip install --user PyYAML  # needed for YAML suites, lab-pipeline, and acceptance validation
bash bin/lab-tools-install            # copies into ~/bin, ~/.config/lab, ~/notes
export PATH="$HOME/bin:$PATH"         # if ~/bin is not already in your shell PATH
lab-doctor                            # sanity check
lab-host-acceptance                   # reproducible host readiness artifact
lab-acceptance-verify ~/lab/_acceptance/<dir>
lab-acceptance-matrix --dry-run       # see required cross-host gates
```

For the current completion status and the remaining RTX hardware gate, see
`notes/completion-audit.md`.

After install, on Intel CPUs:

```bash
sudo lab-pin-system enable-rapl       # one-shot per boot for energy measurement
```

For a measurement campaign:

```bash
sudo lab-pin-system pin               # governor=performance, no_turbo=1, ASLR=0
bench-suite-config baseline.yaml      # full suite with stats + reports
sudo lab-pin-system restore           # back to powersave/turbo on
```

## What a suite produces

Every `bench-suite` run produces a directory under `~/lab/<experiment>/suites/<id>/`:

- `summary.csv` — one row per run, all metrics + duration + RAPL energy + (cuda profile) NVML energy + phase + thermal events
- `stats.csv` — per (workload, phase=all/cold/steady, metric): n, mean, median, SD, CV%, ±1.96σ/√n, 95% bootstrap CI, MAD, outlier count, quality grade
- `report.md` — Markdown report with results + statistics tables
- `method.md` — paper-ready §Methodology section auto-filled from manifest.json
- `reproducibility.md` — ACM-style artifact checklist (auto-scored against artifacts present)
- `execution-order.csv` — randomized (workload, repeat) order with seed for reproducibility
- per-run dirs with `manifest.json` (system snapshot + sha256 of all sources), `monitor.csv` (1–2 Hz thermal/load/RAPL/NVML samples), `result.json` (energy_j, avg_power_w, thermal events, system_pinned, gpu_max_temp_c)

## Cross-host workflow

```bash
# Host A (cpu profile): build a baseline suite
sudo lab-pin-system pin
bench-suite-config baseline.yaml
sudo lab-pin-system restore

# Package it for Host B
lab-handoff <suite_dir>
# -> ~/lab/_handoffs/<experiment>-<id>-handoff-<date>.tar.zst

# Transfer to Host B and re-run with CUDA workloads added
scp ~/lab/_handoffs/*.tar.zst hostB:~/
ssh hostB
git clone <THIS-REPO-URL> ~/lab-tools && cd ~/lab-tools && bash bin/lab-tools-install
lab-profile set cuda
sudo lab-pin-system pin
bench-suite-config baseline.yaml      # auto-includes cuda-vector + cuda-gemm
sudo lab-pin-system restore

# Compare cross-host (works on either side)
suite-compare <hostA_suite_dir> <hostB_suite_dir> --md compare.md
```

## Statistical methods (already wired in)

- Mean, median, SD, CV%, parametric ±1.96σ/√n
- 95% bootstrap percentile CI over 10000 resamples (seed-fixed via `LAB_BOOTSTRAP_SEED`)
- MAD-based outlier flagging (Iglewicz & Hoaglin)
- Phase split: `cold` = first run of each workload by execution order; `steady` = rest
- Suite A vs B: Mann-Whitney U two-sided + Cliff's δ + Romano (2006) effect-size thresholds (negligible/small/medium/large)

## Container infrastructure

- `~/.config/lab/containers/Containerfile.cpu` — Ubuntu 24.04 + clinfo/OpenCL/OpenBLAS/sysbench/python
- `~/.config/lab/containers/Containerfile.cuda` — `nvidia/cuda:13.1.0-devel-ubuntu24.04` + same toolchain (Blackwell sm_120 ready)

Both containers include PyYAML for suite parsing and SciPy for Mann-Whitney
statistics in `suite-compare`.

```bash
lab-container-build cpu          # cpu profile
lab-container-build cuda         # cuda profile
lab-container-run -- bench-standard-cpu
```

## Editing scripts

Active source-of-truth is `~/bin/` and `~/.config/lab/`. After editing, run `lab-tools-sync` to copy changes back into this repo and commit. Other hosts pull and run `lab-tools-install`.

## Out of scope

This framework targets commodity Linux CPU/iGPU, Apple Silicon smoke testing,
and consumer-tier NVIDIA CUDA. It does *not* attempt to support:

- Multi-GPU / large LLM training / H100 baselines
- AMD ROCm or Intel oneAPI/SYCL (stubs only)
- Distributed/HPC measurement, except planned NCCL/PXN scaffolding
- Windows or non-Apple-Silicon macOS execution

For workloads beyond the local hardware envelope, use `lab-handoff` to package a suite and run on cloud-by-hour GPUs (RunPod / Lambda / Vast.ai) or shared cluster resources (KISTI Nurion, NIPA AI 바우처, university GPU clusters).
