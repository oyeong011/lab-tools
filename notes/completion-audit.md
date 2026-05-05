# Completion Audit

Date: 2026-05-05

This audit maps the original lab-tools objective to concrete repository
artifacts and current verification evidence. It is intentionally conservative:
the project is not complete until at least one RTX CUDA host produces and passes
the required acceptance bundle.

## Objective

Build a portable experiment pipeline for the lab accelerator research workflow,
covering:

- Apple Silicon MacBook M1/M4.
- Ubuntu Intel CPU/iGPU desktop.
- Ubuntu RTX 5060 8GB desktop.
- Ubuntu RTX 5080 16GB desktop.
- Forest GPU UVM reference PDF integration.
- Development, validation, tests, GitHub push, and adversarial-review hardening.
- Usability from other computers without depending on this Mac's local state.

## Prompt-To-Artifact Checklist

| Requirement | Artifact or command | Evidence | Status |
|---|---|---|---|
| Portable install on another computer | `bin/lab-tools-install` | Fresh clone install dry-run passes; dry-run has no side effects | Done |
| Host profile detection | `bin/lab-profile`, `bin/lab-doctor` | Profiles cover `cpu`, `cuda`, `apple`; doctor lists required commands | Done |
| Research matrix | `config/pipelines/research-matrix.yaml` | `lab-validate matrix config/pipelines/research-matrix.yaml` passes | Done |
| Forest PDF-derived UVM path | `forest-uvm-access`, `config/cuda-uvm-access.cu`, `bin/lab-uvm-profile` | Access classes `ls`, `hchi`, `hcli`, `lc`; UVM Nsight profile path documented | Done |
| RTX CUDA smoke path | `bin/lab-rtx-smoke`, CUDA wrappers | Dry-run/config validation passes locally; real run requires RTX host | Blocked |
| RTX GPU identity and VRAM gate | `lab-acceptance-verify --require-gpu-name --min-gpu-memory-mib` | Verifier parses `rtx-smoke-*` logs for `nvidia-smi` GPU name and `MiB` memory | Ready |
| RTX compute capability and codegen gate | `lab-cuda-arch-flags`, `--require-compute-cap 12.0`, `--require-cuda-sm 120` | CUDA wrappers emit/use detected `-gencode`; verifier parses `rtx-smoke-*` toolchain logs | Ready |
| Ubuntu Intel iGPU identity and run gate | `lab-acceptance-verify --require-run --require-opencl-device Intel` | Verifier requires `cpu-smoke-run` plus an Intel `Device #` line in `lab-doctor` OpenCL output | Ready |
| Apple Silicon identity gate | `lab-acceptance-verify --require-apple-chip` | Verifier parses `lab-apple-smoke` system output | Ready |
| Tool provenance binding | `lab-host-acceptance` `tool_provenance`, `--require-provenance` | Acceptance records git metadata and SHA256 hashes for tool/config payloads | Ready |
| CUDA memory kernels | `config/cuda-memory-kernels.cu`, `bench-cuda-gemv`, `bench-cuda-spmv`, `bench-cuda-gcn` | Unit tests cover summary/report/stat columns and sweep generation | Done for code, blocked for RTX runtime |
| Apple Silicon path | `bin/lab-apple-smoke`, `bench-apple-metal`, `bench-apple-mps` | Apple collect and acceptance bundle passed on this Mac | Done |
| Acceptance artifact generation | `bin/lab-host-acceptance` | Local Apple acceptance artifact generated under `~/lab/_acceptance` | Done |
| Acceptance verification | `bin/lab-acceptance-verify` | Verifies required steps, logs, status, profile-specific run gates, hardware identity gates, and copied Nsight UVM reports when required | Done |
| Transfer bundle | `bin/lab-acceptance-bundle` | Creates `.tar.gz` plus `.sha256`; `--check-bundle` verifies sidecar, internal hashes, and acceptance | Done |
| One-command host collection | `bin/lab-acceptance-collect` | Apple actual run passed; CUDA dry-run shows required commands | Done |
| Whole matrix bundle audit | `bin/lab-acceptance-matrix`, `config/acceptance/required-hosts.json` | Fresh clone dry-run and unit test cover target mapping; real completion still requires missing bundles | Ready |
| Drive/USB bundle staging | `bin/lab-acceptance-stage` | Copies latest passing bundle per matrix target plus `.sha256`, writes `STAGE-MANIFEST.json`, and supports strict full-matrix gating | Ready |
| SSH automation | `bin/lab-remote-acceptance` | Fresh clone remote dry-run passes; actual run requires SSH target | Ready |
| Suite handoff | `bin/lab-handoff` | Includes acceptance tools; uses Python SHA256; test opens produced tarball | Done |
| GitHub push | `origin/main` | Verify with `git status --short --branch` and matching `git rev-parse HEAD origin/main` | Done |
| CI gate | `.github/workflows/ci.yml` | Verify latest run with `gh run list --repo oyeong011/lab-tools --branch main --limit 1` | Done |
| Fresh clone reproducibility | `git clone https://github.com/oyeong011/lab-tools` | Fresh clone unittest, install dry-run, acceptance-matrix dry-run, and remote dry-run pass | Done |
| RTX 5060/5080 real hardware validation | `lab-remote-acceptance <target> --profile cuda --run --uvm-profile` | No RTX SSH target or bundle has been provided; current host lacks `nvidia-smi`, `nvcc`, and `nsys` | Missing |

## Current Evidence Snapshot

- Current commit evidence should be checked with `git rev-parse HEAD` and
  `git status --short --branch` after each push.
- Current CI evidence should be checked with
  `gh run list --repo oyeong011/lab-tools --branch main --limit 1`.
- Current host: Darwin arm64 Apple Silicon.
- Current host has no `nvidia-smi`, `nvcc`, or `nsys` on `PATH`.
- A local Apple acceptance collect run produced and verified an acceptance
  bundle under `~/lab/_acceptance_bundles`.
- No Google Drive desktop sync folder was found under
  `~/Library/CloudStorage` or `/Volumes` on this Mac during the latest check.
  The current local acceptance bundle directory is small; when RTX Nsight
  bundles become large, use `lab-acceptance-stage --out <Drive folder>` or move
  each `.tar.gz` together with its `.sha256` sidecar to Drive and run
  `lab-acceptance-matrix --bundle-dir <Drive folder>`.

## Required RTX Completion Command

If SSH access is available from this Mac:

```bash
lab-remote-acceptance user@rtx-host --profile cuda --run --uvm-profile --require-provenance --require-gpu-name "RTX 5060" --min-gpu-memory-mib 7600 --require-compute-cap 12.0 --require-cuda-sm 120
```

For the Ubuntu Intel iGPU host:

```bash
lab-remote-acceptance user@intel-host --profile cpu --run --require-provenance --require-opencl-device Intel
```

For Apple Silicon:

```bash
lab-acceptance-collect --profile apple --run --require-provenance --require-apple-chip "Apple M1"
```

Use `"Apple M4"` on the M4 MacBook.

If commands are run directly on the RTX host:

```bash
cd ~/lab-tools
git pull
bash bin/lab-tools-install
export PATH="$HOME/bin:$PATH"
lab-acceptance-collect --profile cuda --run --uvm-profile --require-provenance --require-gpu-name "RTX 5060" --min-gpu-memory-mib 7600 --require-compute-cap 12.0 --require-cuda-sm 120
```

The returned bundle must then pass:

```bash
lab-acceptance-bundle --check-bundle <bundle.tar.gz> --expect-profile cuda --require-run --require-uvm-profile --require-provenance --require-gpu-name "RTX 5060" --min-gpu-memory-mib 7600 --require-compute-cap 12.0 --require-cuda-sm 120
```

For the RTX 5080 16GB host, use `--require-gpu-name "RTX 5080"` and a memory
gate such as `--min-gpu-memory-mib 15000`.

After all host bundles are copied into one directory:

```bash
lab-acceptance-matrix --bundle-dir ~/lab/_acceptance_bundles
```

This command is the overall acceptance gate for portability across the explicit
host matrix. It should report `complete=yes` before the project is considered
fully validated.

## Adversarial Review Hardening

Three hostile CPU/GPU/memory-systems review passes identified weak evidence
gates. The current repository addresses the actionable code-level issues:

- UVM acceptance now requires a non-empty `.nsys-rep` and non-empty Nsight UVM
  stats CSV (`um_sum`, `um_total_sum`, or `um_cpu_page_faults_sum`).
- RTX acceptance now requires compute capability `12.0`, `sm_120` codegen
  evidence, and tool/config provenance hashes.
- CUDA wrappers use `lab-cuda-arch-flags` so Blackwell hosts compile with
  explicit `-gencode` flags instead of relying only on nvcc defaults.
- Intel OpenCL acceptance now checks `Device #` lines in the OpenCL device
  section instead of accepting arbitrary `Intel` text elsewhere in logs.
- Host acceptance resolves installed `~/.config/lab` configs and no longer
  depends on running from the repository checkout.
- Memory-hierarchy/PIM metrics were narrowed to fields the harness actually
  emits; DRAM row-buffer and activation-count claims remain guarded as external
  trace/simulator work.
- The Forest source reference no longer hardcodes a local absolute PDF path;
  the portable summary lives in repository notes.

## Decision

Do not mark the overall goal complete yet. The repository work is in a portable
and CI-passing state, but the explicit Intel iGPU, M4, RTX 5060, and RTX 5080
acceptance requirements are unverified until those actual hosts produce passing
acceptance bundles.
