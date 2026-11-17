# 중단 기록 — 규칙 기반(Ontology-K) 경로 중단, 설계 변경 (2026-11-16 16:42)

사용자 요청으로 진행 중이던 시험 세트 수집 배치를 중단했다. 이후 연구는 순환형 R-GAT 기반 간선 선택·항법 알고리즘 선택(`docs/RGAT_DESIGN_KO.md`)으로 바뀐다.
이 문서는 중단 시점까지의 결과만 남긴다.

## 1. 중단한 작업과 방법

| 항목 | 내용 |
|---|---|
| 실행 중이던 것 | `python3 tools/run_batch.py --set test` (2026-11-16 10:41 시작, 다른 Claude 세션의 백그라운드 작업) |
| 중단 시점 | 72/330번째 실행 `S05` 시드 107 (warehouse), 시작 약 2분 경과 |
| 종료 순서 | `run_batch.py` SIGTERM → `run_stack.sh` SIGTERM(trap으로 XRCE agent·OpenVINS·GLIM·context monitor 그룹 종료) → Isaac(kit python)·PX4 SITL·recorder SIGINT/TERM/KILL |
| 확인 | 관련 프로세스 0개, GPU 연산 프로세스 0개 |
| 미완료 실행 | `$IJCAS_DATA/runs/test/S05_107`: DONE/FAILED 표식 없음, `ABORTED` 표식 추가. `run_batch.py`를 다시 실행하면 이 디렉터리를 지우고 처음부터 다시 수행한다. |

## 2. 데이터 수집 현황 (`$IJCAS_DATA` = `~/ijcas_data`)

| 세트 | 완료 | 실패 | 비고 |
|---|---|---|---|
| validation (시드 1–10, S00–S08·S10) | 100/100 | 0 | 1.6 GB |
| test (시드 101–130, S00–S10) | 71/330 | 0 | 1.2 GB; 시드 101–106은 S00–S10 전부, 시드 107은 S00–S04 |

- 기록 내용(`record.npz`)은 선택기 출력 없이 후보 항법기 출력(PX4 EKF2, OpenVINS, GLIM)과 진단을 모두 독립적으로 담는다(`odom_selected` 0행). 선택 결과가 항법기 내부 상태를 바꾸지 않는 상태로 기록됐으므로 새 설계의 학습·평가에 그대로 쓸 수 있다.
- 실행당 실시간 약 4.6–5.4분. 남은 259회를 모두 수집하려면 약 21시간이 걸린다.
- 진행 로그: `$IJCAS_DATA/runs/batch_test.log`, `batch_validation.log`.

## 3. 중단 시점까지 완료된 단계

| 단계 | 상태 | 근거 |
|---|---|---|
| G0 환경 구축 | 진행 중(GLIM 공식 예제만 네트워크 문제로 미검증) | `results/g0/` |
| G1 시뮬레이션 리그 | 완료 | `results/g1/` |
| G2 후보 엔진 단독 성능 | 완료 | `results/g2/`, `docs/G0_G2_KO.md` |
| G3 진단 지표·품질 모델 보정 | 완료, 검증 세트로 고정 | `config/selector/thresholds_validated.yaml`, `results/g3/FROZEN.sha256` |
| G4 지식 표현·추론 | 완료 | `results/g4/`, `ijcas/knowledge/ijcas_nav.ttl` |
| G5 정렬·전환 | 진행 중(실시간 단일 TF 확인은 G7 이후로 미룸) | `docs/G3_G5_KO.md` |
| G6 시험 세트 집계 | **미수행** — 시험 배치가 끝나지 않음 | — |
| G7 실시간 자원 측정, G8 폐루프 | **미수행** | `scripts/run_stack_rt.sh`, `isaac/ijcas_world_closed_loop.py`(미시험) |

검증 세트에서 보정한 Ontology-K 결과(`results/g9/validation_calibrated/report.md`): 평균 가용성 0.979, 오정상 출력 1.72 s/실행. Flat-K는 모든 실행·지표에서 Ontology-K와 같았다.

## 4. 시험 세트 사용 상태

- 시험 세트 71개 실행의 선택 성능은 **한 번도 계산하거나 열람하지 않았다**(G6 미수행).
- 따라서 새 설계에서도 시험 세트를 보류 세트로 쓸 수 있다. 구조·임계값은 검증 세트에서만 정하고, 고정한 뒤에만 시험 세트를 평가한다.

## 5. 무효가 된 계획

- '시험 배치 완료 → `tools/g6_aggregate.py` → G7 → G8' 순서는 이번 설계 변경으로 무효다.
- 규칙 기반 Ontology-K와 고정된 임계값은 지우지 않는다. 새 설계의 규칙 기반 비교 선택기로 쓴다.
