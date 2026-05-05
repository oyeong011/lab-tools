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
| CUDA memory kernels | `config/cuda-memory-kernels.cu`, `bench-cuda-gemv`, `bench-cuda-spmv`, `bench-cuda-gcn` | Unit tests cover summary/report/stat columns and sweep generation | Done for code, blocked for RTX runtime |
| Apple Silicon path | `bin/lab-apple-smoke`, `bench-apple-metal`, `bench-apple-mps` | Apple collect and acceptance bundle passed on this Mac | Done |
| Acceptance artifact generation | `bin/lab-host-acceptance` | Local Apple acceptance artifact generated under `~/lab/_acceptance` | Done |
| Acceptance verification | `bin/lab-acceptance-verify` | Verifies required steps, logs, status, and profile-specific run gates | Done |
| Transfer bundle | `bin/lab-acceptance-bundle` | Creates `.tar.gz` plus `.sha256`; `--check-bundle` verifies sidecar, internal hashes, and acceptance | Done |
| One-command host collection | `bin/lab-acceptance-collect` | Apple actual run passed; CUDA dry-run shows required commands | Done |
| SSH automation | `bin/lab-remote-acceptance` | Fresh clone remote dry-run passes; actual run requires SSH target | Ready |
| Suite handoff | `bin/lab-handoff` | Includes acceptance tools; uses Python SHA256; test opens produced tarball | Done |
| GitHub push | `origin/main` | Local `HEAD` equals `origin/main` | Done |
| CI gate | `.github/workflows/ci.yml` | Latest main CI success: `25361654664` | Done |
| Fresh clone reproducibility | `git clone https://github.com/oyeong011/lab-tools` | Fresh clone unittest, install dry-run, remote dry-run pass | Done |
| RTX 5060/5080 real hardware validation | `lab-remote-acceptance <target> --profile cuda --run --uvm-profile` | No RTX SSH target or bundle has been provided; current host lacks `nvidia-smi`, `nvcc`, and `nsys` | Missing |

## Current Evidence Snapshot

- Latest verified commit: `dfa9ff90e12491885e6a6dd1e991ce4e8790919a`.
- Latest CI: <https://github.com/oyeong011/lab-tools/actions/runs/25361654664>.
- Current host: Darwin arm64 Apple Silicon.
- Current host has no `nvidia-smi`, `nvcc`, or `nsys` on `PATH`.
- A local Apple acceptance collect run produced and verified an acceptance
  bundle under `~/lab/_acceptance_bundles`.

## Required RTX Completion Command

If SSH access is available from this Mac:

```bash
lab-remote-acceptance user@rtx-host --profile cuda --run --uvm-profile --require-gpu-name "RTX 5060" --min-gpu-memory-mib 7600
```

If commands are run directly on the RTX host:

```bash
cd ~/lab-tools
git pull
bash bin/lab-tools-install
export PATH="$HOME/bin:$PATH"
lab-acceptance-collect --profile cuda --run --uvm-profile --require-gpu-name "RTX 5060" --min-gpu-memory-mib 7600
```

The returned bundle must then pass:

```bash
lab-acceptance-bundle --check-bundle <bundle.tar.gz> --expect-profile cuda --require-run --require-uvm-profile --require-gpu-name "RTX 5060" --min-gpu-memory-mib 7600
```

For the RTX 5080 16GB host, use `--require-gpu-name "RTX 5080"` and a memory
gate such as `--min-gpu-memory-mib 15000`.

## Decision

Do not mark the overall goal complete yet. The repository work is in a portable
and CI-passing state, but the explicit RTX 5060/5080 acceptance requirement is
unverified until an actual CUDA host produces a passing acceptance bundle.
