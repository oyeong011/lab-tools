# Research Pipeline Guide

This repo is a reproducibility harness first. It can run several CPU/OpenCL/CUDA
microbenchmark campaigns today, and it provides a matrix for expanding toward
the lab's broader CPU, GPU, memory, accelerator, and systems publication space.

## Tested Platform Matrix

| Platform | Status | Profile | Notes |
|---|---|---|---|
| Ubuntu + Intel CPU/iGPU | wired | `cpu` | CPU, OpenCL, RAPL, report/statistics pipeline |
| Ubuntu + RTX 5060 8GB | wired | `cuda` | CUDA baseline and managed-memory probe; Blackwell container is CUDA 13.1 based |
| Ubuntu + RTX 5080 16GB | wired | `cuda` | Same CUDA path; larger UVM sweeps are practical |
| MacBook M1/M4 | smoke | `apple` | `lab-apple-smoke` runs Metal vector-add and can attempt optional PyTorch MPS vector-add; package energy collection still needs powermetrics integration |
| Multi-node GPU cluster | planned | `cluster` | Needs NCCL tests, topology capture, rail binding, and launch integration |

## Wired Experiments

### Consumer Accelerator Baseline

Runs CPU toy kernels, STREAM, OpenBLAS, sysbench, OpenCL vector/GEMM, and CUDA
vector/GEMM. Use this for sanity checks, reproducibility plumbing, and
consumer-device comparison. Do not treat cross-host results as a single-variable
accelerator comparison unless host identity is explicitly modeled.

### Forest-Inspired UVM Access Probe

`bench-cuda-uvm-access` allocates managed memory with `cudaMallocManaged` and
exercises four access-pattern classes inspired by Forest:

| Pattern | Meaning | Probe behavior |
|---|---|---|
| `ls` | linear streaming | dense sequential pass over the object |
| `hchi` | high coverage, high intensity | scattered accesses across the full object |
| `hcli` | high coverage, low intensity | sparse page-stride touches across the object |
| `lc` | low coverage | dense accesses over a small region |

Generate a sweep:

```bash
lab-pipeline plan forest-uvm-access --profile cuda --sweep
```

Then run selected suite configs from the generated `runbook.md`.

Claim boundary: this probe can show timing, bandwidth, energy, and thermal
symptoms under different managed-memory access patterns and allocation sizes.
It cannot by itself prove page thrashing, TBNp policy quality, or Forest-style
driver/hardware mechanisms. Those claims need CUPTI/Nsight/driver counters or
simulator instrumentation.

Use Nsight Systems UVM tracing when a mechanism explanation is needed:

```bash
lab-uvm-profile --pattern hchi --mb 12288 --passes 2
```

The generated report and CSV summaries are stored under `~/lab/_profiles`.

### CUDA GEMV/SpMV/GCN Memory Kernels

`bench-cuda-gemv`, `bench-cuda-spmv`, and `bench-cuda-gcn` provide the first
wired path for bandwidth-dominated matrix-vector and synthetic graph neighbor
aggregation kernels. They are still microbenchmarks, but they are closer to
PIM/GEMV/SpMV/GCN papers than the original GEMM-only suite.

```bash
bench-suite-config memory-kernels
lab-pipeline plan memory-hierarchy-pim --profile cuda --sweep
```

### Apple Silicon Metal Smoke

`bench-apple-metal` runs a Swift/Metal vector-add benchmark on Apple Silicon.
This is a smoke path, not yet a full peer to the Linux `bench-suite` flow.

```bash
lab-apple-smoke
LAB_APPLE_ELEMENTS=1048576 lab-apple-smoke --run
LAB_APPLE_ELEMENTS=1048576 lab-apple-smoke --run --run-mps
```

## Publication Coverage Map

| Publication family | Representative user-provided papers | Pipeline track | Status |
|---|---|---|---|
| Edge AI accelerator arithmetic and CNN/Transformer workloads | LogFlex, FINEA, embedded CNN platform analysis, HALO, TM-Training | `edge-ai-cnn-transformer` | planned |
| Embedded GPU optimization and multi-tenant inference | VitBit, TLP Balancer, Adaptive Kernel Merge/Fusion, SSFFT | `consumer-accelerator-baseline`, `multi-tenant-migration-storage` | partial/planned |
| GPU UVM, tensor migration, oversubscription | MOST, Beyond VABlock, Forest reference PDF | `forest-uvm-access`, `multi-tenant-migration-storage` | partial/planned |
| Memory hierarchy, DRAM row policy, GPU memory controller | row-buffer activation-count papers, Warped-MC, data-cache analysis | `memory-hierarchy-pim` | partial |
| PIM, GEMV, SpMV, sparse-dense kernels | SparsePIM, GEMV GPU/PIM address mapping, HyMM, GCN aggregation | `memory-hierarchy-pim` | partial |
| Cluster communication | NCCL PXN rail-optimized networks | `cluster-communication` | planned |
| Storage and near-data processing | Coldmap, FLIXR, GraphSSD, Summarizer | not wired | unsupported today |
| Security and side channels | Vizard, RT-Sniper, RoCC/RISC-V side-channel papers, GhostLeg, CacheRewinder | `security-counter-cache` | planned defensive-only |

## Reviewer Rules

- Separate build, warmup, measured invocation, and analysis. Current wrappers
  rebuild only when sources are newer than binaries; use warmups before
  measured repeats.
- Preserve raw artifacts and validate them with `lab-validate suite-dir`.
- Report cold and steady phases separately.
- Report confidence intervals and effect sizes; avoid claims from single runs.
- Treat host identity, driver version, clock/power state, topology, and cooling
  as experimental factors.
- Do not claim UVM mechanism behavior without mechanism counters.
