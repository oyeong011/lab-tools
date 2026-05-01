# lab-tools

Local CPU / accelerator experiment tooling for this Ubuntu machine.

## Install Back Into Home

This repository is a tracked copy of the active files. The active files live in:

- `~/bin`
- `~/.config/lab`
- `~/notes`

Use `lab-tools-sync` from the active environment to refresh this repo after edits.

## Common Commands

```bash
	lab-doctor
	lab-container-build cpu
	lab-container-run -- bench-standard-cpu
	bench-suite-config quick.yaml
	bench-suite-config baseline.yaml
	lab-index
	lab-archive latest
	lab-backup
```

## Safety

Runs should go through `lab-safe-run`, `monitor-run`, or `bench-suite`.
Each run records `manifest.json`, `result.json`, `monitor.csv`, sensor snapshots, stdout, and stderr.
