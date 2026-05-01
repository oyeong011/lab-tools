# Ubuntu Lab Profile

This machine is organized as a local CPU and AI accelerator experiment workstation.

## Directories

- `~/lab`: experiment workspaces
- `~/lab/_hardware`: timestamped hardware reports
- `~/lab/_benchmarks`: baseline benchmark outputs
- `~/src`: cloned source repositories
- `~/bin`: personal commands
- `~/notes`: notes and operating procedures
- `~/sandbox`: disposable scratch work

## Commands

- `lab list`: show experiments
- `lab new NAME`: create `~/lab/NAME` with `src`, `data`, `notes`, `tmp`, `env`, `logs`, `results`, and `benchmarks`
- `lab go NAME`: open or switch to tmux session `lab-NAME`
- `lab report`: save a hardware report
- `hw-report`: save a hardware report
- `ai-accel-check`: inspect CPU/GPU/AI accelerator runtime readiness
- `bench-cpu`: run a small CPU baseline benchmark
- `bench-stream`: run a STREAM-style memory bandwidth baseline
- `bench-openblas`: run an OpenBLAS DGEMM CPU baseline
- `bench-sysbench`: run a sysbench CPU baseline
- `bench-standard-cpu`: run STREAM, OpenBLAS, and sysbench together
- `bench-opencl`: compile and run a small OpenCL vector-add kernel
- `bench-opencl-gemm`: compile and run a small OpenCL matrix multiplication benchmark
- `nvidia-check`: report NVIDIA GPU, driver, `nvidia-smi`, and CUDA compiler readiness
- `sensor-snapshot`: record temperature, CPU frequency, load, and memory
- `thermal-check`: fail if the machine is already above a temperature threshold
- `lab-safe-run EXPERIMENT -- COMMAND`: run a command with timeout, low priority, logs, and thermal precheck
- `monitor-run EXPERIMENT -- COMMAND`: same as `lab-safe-run`, with emphasis on `monitor.csv`
- `bench-suite EXP REPEATS`: run repeated CPU/OpenCL baselines with hardware and sensor records
- `bench-suite-config quick.yaml`: run a declarative YAML suite config
- `summarize-suite SUITE_DIR`: extract CPU/OpenCL metrics into `summary.csv`
- `render-suite-report SUITE_DIR`: generate a Markdown report from `summary.csv`
- `suite-stats SUITE_DIR`: generate statistics, CV, and quality flags in `stats.csv`
- `lab-plot latest`: generate PNG plots from the latest suite summary and monitor logs
- `lab-container-build cpu`: build the rootless Podman CPU experiment image
- `lab-container-run -- COMMAND`: run a command in the CPU experiment container with `~/lab` mounted
- `lab-nvidia-ready`: check NVIDIA driver, CUDA compiler, container CDI, and ONNX Runtime GPU readiness
- `lab-manifest RUN_DIR EXP WORKDIR -- COMMAND`: record run provenance as `manifest.json`
- `lab-doctor`: run a preflight check for tools, OpenCL, files, and temperature
- `lab-index`: collect all suite summaries into `~/lab/suite-index.csv`
- `lab-archive latest`: compress a suite with checksum under `~/lab/_archives`
- `lab-backup`: back up lab tools, notes, hardware reports, and key suite artifacts
- `lab-clean --days 30`: dry-run cleanup for old raw run directories
- `take NAME`: create an experiment and cd into it
- `t`: open or attach tmux session `main`
- `tls`: list tmux sessions
- `reload`: reload bash settings

## Tmux

- `Ctrl-a`: tmux prefix
- `Ctrl-Left`: previous session
- `Ctrl-Right`: next session
- `Ctrl-Up`: create new session
- `Ctrl-Down`: choose session
- `Alt-Arrow`: move between panes
- Mouse is enabled for pane/window selection and copy-mode selection.

## Current Hardware Profile

- CPU: Intel Core i7-7700, 4 cores / 8 threads
- GPU/accelerator visible now: Intel HD Graphics 630 via `i915` and `/dev/dri/renderD128`
- OpenCL: Intel OpenCL Graphics runtime installed and HD Graphics 630 detected
- Discrete NVIDIA/AMD/Coral/other accelerator: not currently detected on PCIe or USB
- Future NVIDIA GPU path: install via Ubuntu Additional Drivers, then verify with `nvidia-check`
- Container isolation: rootless Podman image `localhost/lab-cpu:24.04` is built for CPU baselines
- Memory: 15 GiB
- Disk free on `/`: about 350 GiB

## Next Candidates

- After NVIDIA GPU installation, add NVIDIA Container Toolkit/CDI and a CUDA/ONNX Runtime GPU container image.
- Add a dotfiles Git repository for repeatable setup.
- Add backup scripts for `~/lab` and `~/notes`.
- Add per-language templates under `~/.config/lab/templates`.

## Safety Policy

- Prefer `lab-safe-run` for experiments.
- Default timeout is 10 minutes.
- `LAB_WARMUPS` controls warm-up runs, which are logged but excluded from `summary.csv`.
- `LAB_REPEATS` or the second `bench-suite` argument controls measured runs.
- Suite configs live under `~/.config/lab/suites/*.yaml`.
- Runs use lower CPU and I/O priority by default.
- Logs are stored under `~/lab/<experiment>/runs/<timestamp>`.
- Sensor snapshots are captured before and after each run.
- Continuous monitor samples are stored in `monitor.csv` for each run.
- Each run stores `manifest.json` and `result.json`.
- Runtime thermal abort uses exit code 98; thermal precheck failure uses exit code 99.
- Use `lab-index` to find previous suites.
- Use `lab-plot latest` to inspect summary and temperature/load graphs.
- Use `lab-container-run -- bench-standard-cpu` for isolated CPU baseline checks.
- Use `lab-nvidia-ready` after installing a future NVIDIA GPU.
- Use `lab-backup` before risky changes or after important experiments.
- Use `lab-clean` in dry-run mode first; add `--apply` only after checking the deletion list.
- Avoid kernel parameters, overclocking, and external driver repositories until baseline data is stable.
