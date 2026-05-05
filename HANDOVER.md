# lab-tools 인수 인계

이 문서는 `lab-tools`를 처음 받아 운영하는 사람이 한 시간 안에 측정을 시작하고, 한 주 안에 페이퍼 데이터를 모을 수 있도록 쓴 운영 안내서입니다. 설계 의도와 함정을 모두 적었습니다.

## 0. 한 문단 요약

`lab-tools`는 두 호스트로 운영되는 재현성 지향 벤치마킹 프레임워크입니다.

- **호스트 A (cpu 프로파일):** Intel i7-7700 + HD 630. NVIDIA GPU 없음. 측정 방법론·재현성 개발 환경.
- **호스트 B (cuda 프로파일):** Intel Core Ultra 5 235 + RTX 5060 8GB(Blackwell sm_120) + CUDA 13.1. CUDA 실행 환경.

같은 코드, 같은 통계, 같은 리포트 형식을 양쪽에서 쓰되 `lab-profile`이 호스트 능력을 자동 감지해서 워크로드만 자동으로 갈라집니다. summary.csv 스키마는 두 프로파일에 통합되어 있어 `suite-compare`가 호스트 간 직접 비교를 별도 변환 없이 처리합니다.

설계 목표 청중: 재현성/방법론 학회(ACM REP, SC Reproducibility Appendix, BenchCouncil), 엣지·소비자 하드웨어 측정 워크숍. **메인 컨퍼런스 가속기 트랙(MLSys/SC/ISCA 본진)은 8GB Blackwell로는 어림없음** — 그건 클라우드/HPC 자원으로 보내야 함.

## 1. 첫 1시간

### 새 호스트 설치
```bash
git clone https://github.com/oyeong011/lab-tools.git ~/lab-tools
cd ~/lab-tools
bash bin/lab-tools-install        # ~/bin, ~/.config/lab, ~/notes 로 복사
lab-doctor                        # 빨간 줄 없는지 확인
```

### 프로파일 확인 (자동 감지)
```bash
lab-profile          # 이 호스트의 활성 프로파일 출력
lab-profile detect   # 자동 감지 결과만 (override 무시)
```

NVIDIA GPU + nvcc가 모두 있으면 `cuda`, 아니면 `cpu`. 강제 변경:
```bash
lab-profile set cuda     # 영구 (~/.config/lab/profile에 저장)
lab-profile clear        # 영구 해제 → 자동 감지로 복귀
LAB_PROFILE=cpu lab-doctor   # 일회성 환경변수 override
```

### Intel RAPL 에너지 측정 활성화 (Intel CPU만, 부팅 후 1회)
```bash
sudo lab-pin-system enable-rapl
```
이 단계를 빠뜨리면 **모든 에너지 컬럼이 비어 있음**. result.json의 `energy_j`가 `{}`이면 이게 원인.

### 시범 캠페인
```bash
sudo lab-pin-system pin
bench-suite-config quick.yaml     # 30초~1분
sudo lab-pin-system restore
ls ~/lab/cpu-ai-kernel-baseline/suites/  # 만들어진 디렉터리 확인
```

### 범용 연구 파이프라인 확인
```bash
lab-pipeline list
lab-pipeline review
lab-pipeline plan consumer-accelerator-baseline --profile cpu --dry-run
lab-pipeline plan forest-uvm-access --profile cuda --sweep --dry-run
lab-validate matrix ~/.config/lab/pipelines/research-matrix.yaml
```

`forest-uvm-access`는 Forest 논문에서 중요한 managed memory access
pattern/oversubscription 축을 실제 CUDA UVM 마이크로프로브로 준비한
경로입니다. 단, 이것은 Forest의 UVM driver/hardware 변경 구현이 아니므로
page fault, migration, thrashing 주장은 CUPTI/Nsight/driver counter 또는
시뮬레이터 계측과 함께 써야 합니다.

## 2. 캠페인 산출물

`bench-suite`(또는 `bench-suite-config`)를 한 번 돌리면 `~/lab/<experiment>/suites/<id>/`에 다음이 생깁니다:

| 파일 | 내용 |
|---|---|
| `summary.csv` | 행=run, 열=메트릭 + duration + RAPL 4도메인 에너지 + (cuda) NVML 에너지 + phase + throttle count |
| `stats.csv` | (workload × phase=all/cold/steady × metric) → n, mean, median, SD, CV%, ±1.96σ/√n, 95% bootstrap CI(10000 resamples), MAD, outlier count, quality grade |
| `report.md` | 결과 표 + 통계 표 + execution order |
| `method.md` | 페이퍼 §Methodology 단락 초안 (CPU/GPU/커널/통계 방법 자동 채움) |
| `reproducibility.md` | ACM 아티팩트 체크리스트 17개 항목 자동 채점 |
| `execution-order.csv` | 셔플 시드 기반 (order, workload, repeat) — 재현 가능 |

각 run 디렉터리 (`runs/<run_id>/`):

| 파일 | 내용 |
|---|---|
| `manifest.json` | 시스템 스냅샷 (lscpu, governor, kernel cmdline, OpenCL 디바이스 목록, 패키지 버전, 모든 소스 sha256) |
| `monitor.csv` | 1-2Hz 샘플 (timestamp, max_temp, CPU scaling, load1, mem, RAPL 4도메인 raw uj, NVML 4컬럼) |
| `result.json` | 종료 코드, duration, energy_j(도메인별), avg_power_w, thermal_event_count, gpu_max_temp_c, system_pinned |
| `stdout.log`, `stderr.log` | 실행 로그 |
| `sensors-before.txt` / `sensors-after.txt` | 시작·종료 시점 lm-sensors 스냅샷 |
| `thermal-before.txt` / `thermal-after.txt` | 온도 가드 결과 |

## 3. 캠페인 한 사이클

```bash
sudo lab-pin-system pin            # governor=performance, no_turbo=1, ASLR=0
bench-suite-config baseline.yaml   # 약 10–30분, 모든 산출물 자동 생성
sudo lab-pin-system restore        # 원상복구
```

**캠페인 두 개 비교 (변경 X가 의미 있는가?):**
```bash
suite-compare <suite_A_dir> <suite_B_dir> --md ~/notes/compare-$(date +%F).md
# Mann-Whitney U + Cliff's δ + Romano(2006) 효과 크기 분류
# verdict 컬럼: ns / stat_sig_negligible / significant / indeterminate
```

n<2 메트릭은 자동으로 `magnitude=indeterminate, verdict=indeterminate`. 실수로 의미를 부여하지 않게.

## 4. 두 호스트 흐름

### 코드 변경 전파 (호스트 A → B)
```bash
# 호스트 A에서 ~/bin 또는 ~/.config/lab 수정 후
lab-tools-sync         # ~/bin → ~/lab-tools 복사 + git commit
cd ~/lab-tools && git push

# 호스트 B에서
cd ~/lab-tools && git pull
bash bin/lab-tools-install
```

### 결과 데이터 전파 (호스트 B → A 또는 그 반대)
```bash
# 만든 호스트에서 패키징
lab-handoff <suite_dir>            # ~/lab/_handoffs/...tar.zst (sha256 동봉)

# 받는 호스트로 전송
scp ~/lab/_handoffs/*.tar.zst other:~/

# 받는 호스트에서 풀고 비교
tar --zstd -xf *.tar.zst -C ~/inbox/
suite-compare ~/lab/<my_suite> ~/inbox/<bundle>/suite --md ~/notes/cross-host.md
```

호스트 A의 OpenCL-on-HD630 vs 호스트 B의 OpenCL-on-Xe vs 호스트 B의 CUDA-on-RTX5060 — 같은 SGEMM 커널을 세 디바이스에서 비교하는 건 페이퍼화 가능한 영역(consumer-tier 가속기 cross-comparison은 발표된 데이터가 적음).

## 5. 자주 쓰는 명령

| 목적 | 명령 |
|---|---|
| 시스템 점검 | `lab-doctor` |
| 하드웨어 보고서 | `hw-report` (저장: `~/lab/_hardware/`) |
| 단일 워크로드 안전 실행 | `lab-safe-run <experiment> -- <command>` |
| 단일 캠페인 실행 | `bench-suite-config <name>` (configs: `~/.config/lab/suites/`) |
| 모든 suite 인덱스 | `lab-index` → `~/lab/suite-index.csv` |
| 오래된 raw runs 정리 | `lab-clean --days 30` (확인) → `--days 30 --apply` (실제 삭제) |
| 백업 | `lab-backup` |
| 컨테이너 빌드 | `lab-container-build cpu` 또는 `lab-container-build cuda` |
| 컨테이너 안에서 실행 | `lab-container-run -- <command>` |
| GPU 상태 확인 | `nvml-check` (cuda 프로파일에서) |
| iGPU/OpenCL 상태 확인 | `ai-accel-check`, `clinfo -l` |

## 6. 파일/디렉터리 지도

```
~/bin/                          모든 실행 스크립트 (소스 오브 트루스)
~/.config/lab/                  설정 + 커널 소스
  ├── *.c, *.cu                 OpenCL/CUDA 커널 소스
  ├── suites/                   YAML/env 캠페인 설정
  ├── containers/               Containerfile.cpu, Containerfile.cuda
  └── profile                   영구 프로파일 override (선택)
~/lab/                          실험 작업 공간 (git에 안 올라감)
  ├── <experiment>/             실험 디렉터리 (lab new로 생성)
  │   ├── runs/                 단일 실행 결과
  │   └── suites/               멀티-반복 캠페인
  ├── _benchmarks/              cpu/, opencl/, cuda/ 빌드 산출물
  ├── _hardware/                hw-report 출력
  ├── _archives/                lab-archive 결과
  └── _handoffs/                lab-handoff 결과
~/lab-tools/                    git repo (소스 미러, push/pull 대상)
~/notes/                        문서, 비교 리포트
```

## 7. 함정 / 주의사항 (실제 겪을 만한 것들)

- **부팅마다 enable-rapl 필요:** `/sys/class/powercap/intel-rapl:*/energy_uj`는 root-only. `lab-pin-system enable-rapl`이 권한을 풀어줌. 부팅하면 다시 막힘.
- **`lab-pin-system pin/restore`는 sudo 필요:** 비밀번호 없는 sudo가 안 되면 직접 입력. 캠페인 후 restore 잊으면 다음 부팅까지 governor=performance/ASLR=0 유지(보통 무해, 그러나 일반 데스크탑 사용 시 발열 증가).
- **컨테이너 안에서는 RAPL 안 됨:** podman rootless 권한 분리. 호스트에서 직접 실행할 때만 에너지가 채워짐. 컨테이너는 빌드 환경 격리/재현용.
- **워크로드 키 ↔ 로그 파일 ↔ summary.csv 매핑:** bench-suite의 키는 `opencl`이지만 summary.csv 컬럼은 `opencl-vector`. CUDA는 `cuda-vector`로 통일. summarize-suite의 `phase_for` 함수가 매핑을 처리. 새 워크로드 추가 시 이 매핑 의식할 것.
- **bench-suite는 항상 셔플함:** 워크로드 실행 순서가 매번 다름. 같은 시드로 재현하려면 `LAB_SHUFFLE_SEED=42`. `execution-order.csv`가 항상 기록되므로 사후에 순서 확인 가능.
- **CV% 5% 임계는 보수적:** OpenCL on iGPU 같은 노이즈 큰 워크로드는 자주 `unstable`. 거짓말이 아니라 정직한 표시.
- **n<3 → low_n:** 통계 검정력 부족 표시. 의미 있는 결론을 내고 싶으면 repeats를 5–10으로.
- **NVML 샘플링 비용:** `nvidia-smi` 호출 ~50–100ms. 1Hz는 OK, 10Hz는 측정에 끼어들 수 있음.
- **CUDA 13.1 + Blackwell sm_120:** 빌드 시 `nvcc -arch=sm_120` 필요한 경우가 있을 수 있음 (현재 `-O2`만 쓰고 자동 fatbin에 맡김 — 문제 생기면 명시).
- **호스트 B의 Arrow Lake P-state:** 일부 sysfs 경로가 i7-7700과 다를 수 있음. lab-pin-system이 graceful degrade하지만 lscpu/`/sys/devices/system/cpu/cpu*/cpufreq/scaling_governor` 출력 다르면 첫 캠페인 후 manifest.json을 한 번 사람 눈으로 확인.

## 8. 확장 가이드

### 새 벤치마크 추가
1. `~/.config/lab/<new>.c` (or `.cu`) 작성
2. `~/bin/bench-<new>` 셸 래퍼 작성 (기존 `bench-opencl-gemm` 참고)
3. `bench-suite`의 `WORKLOAD_CMD`/`WORKLOAD_LABEL`에 등록
4. `summarize-suite`에 새 워크로드 로그 파싱 블록 추가 (정규식으로 stdout 메트릭 뽑아 summary.csv 행 생성)
5. CSV 컬럼이 추가되면:
   - `summarize-suite`의 헤더 `printf` 한 줄 + 행 출력 `printf` 컬럼 수
   - `suite-stats`의 `METRICS` 리스트
   - `suite-compare`의 `METRICS` + (낮을수록 좋으면) `LOWER_IS_BETTER`
6. `lab-tools-sync`의 bin 파일 목록에 새 스크립트 등록
7. `lab-tools-sync && git push` → 다른 호스트에서 `git pull && bash bin/lab-tools-install`

### 새 메트릭만 추가 (기존 워크로드의)
- 기존 bench-* 의 stdout에 출력만 더하면, summarize-suite의 sed 정규식 한 줄로 끝
- result.json 새 필드는 lab-safe-run의 python heredoc에 추가
- summary.csv 컬럼 추가 시 위 5번 항목들 동일하게 갱신

### 새 프로파일 (예: ROCm)
- `lab-profile`의 `detect()`에 분기 추가 (`rocminfo` 검사)
- `Containerfile.rocm` 추가
- `bench-rocm-*` 스크립트 + `~/.config/lab/rocm-*.cpp`
- `lab-doctor`의 `case "$PROFILE"`에 cuda와 같은 패턴으로 분기 추가
- `bench-suite`의 WORKLOADS auto-add 로직에 LAB_INCLUDE_ROCM 추가

## 9. 논문 작성으로 연결

캠페인 산출물 중 두 파일이 페이퍼 작성을 단축시킵니다:

- **`method.md`** — 그대로 LaTeX로 옮기거나 약간 다듬어서 §Methodology 또는 §Experimental Setup으로 사용 가능. CPU/GPU 모델, GCC, 커널, governor 상태, repeat 수, 셔플 시드, 통계 방법(Mann-Whitney/Cliff's δ/bootstrap CI 인용 포함)이 자동 채워짐.
- **`reproducibility.md`** — ACM 아티팩트 평가 자기-체크리스트 17개 항목 자동 마킹. 페이퍼 부록 또는 ACM AE 제출용 README의 시작점.

`suite-compare --md`의 출력은 그대로 페이퍼 표로 가능 (Mann-Whitney p, Cliff's δ, magnitude, verdict 다 들어 있음).

**페이퍼 청중을 잘못 잡지 말 것:**
- 적합: ACM REP, SC Reproducibility Appendix, BenchCouncil, LCTES, RTAS-인접, 워크숍 트랙
- 부적합: MLSys/SC/ISCA/MICRO/HPCA 메인트랙 가속기 비교 — 8GB Blackwell로는 desk reject 사유

## 10. 범위 밖 (이 프레임워크가 안 다루는 것)

- 다중 GPU, >8GB 모델, H100/A100 baseline 비교
- 분산/HPC 측정
- AMD ROCm, Intel oneAPI/SYCL (스텁 정도만)
- Windows / macOS

이런 워크로드는 `lab-handoff`로 패키지해서 클라우드(RunPod / Lambda / Vast.ai) 또는 공공 자원(KISTI 누리온, NIPA AI 바우처, 학교 GPU 클러스터)에 보내 실행. 로컬은 개발/방법론 계층으로 유지.

## 11. 막혔을 때 점검 순서

1. `lab-doctor` — 어디가 빨간지 한눈에
2. `lab-profile detect` — 자동 감지 결과 (override가 잘못 잡혀있는지)
3. `cat ~/.config/lab/profile` — 영구 override 내용
4. `lab-pin-system status` — pinning 상태 (governor / no_turbo / ASLR / RAPL 권한)
5. `tail ~/lab/<experiment>/runs/<latest>/stderr.log` — 직전 실행 에러
6. `cat ~/lab/<experiment>/runs/<latest>/result.json` — 종료 코드, throttle 발생 여부, pinned 여부
7. `head ~/lab/<experiment>/suites/<latest>/summary.csv` — 스키마 변형 여부 (확장 작업 후)
8. `env | grep LAB_` — 환경변수 override 여부

## 12. 저장소 / 이슈 트래커

- 코드: https://github.com/oyeong011/lab-tools
- 이슈 / 개선 제안: 위 리포의 GitHub issues

---

**가장 중요한 두 가지만 다시:**

1. 부팅 후 한 번: `sudo lab-pin-system enable-rapl`
2. 캠페인 직전: `sudo lab-pin-system pin` / 직후: `sudo lab-pin-system restore`

이 두 줄 있고 없고가 페이퍼 통계의 정직성 차이를 만듭니다.
