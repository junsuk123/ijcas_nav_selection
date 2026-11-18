# 중단 기록 (2026-11-18 17:07 KST)

사용자 지시: "지금 돌고 있는 모든 프로세스를 멈추고, 융합 구조 프롬프트에 따라 재구성"
이 문서는 중단 시점의 상태와 재개 방법을 기록한다.

## 멈춘 작업

| 작업 | 중단 시점 상태 | 남은 산출물 | 재개 방법 |
|---|---|---|---|
| 시험2 수집 (`tools/run_batch.py --set test2`, 시드 201–230 × 11 = 330회) | 15회 완료(S00–S03 시드 201, S00–S03 시드 202 등), 실패 0. 진행 중이던 S04_202는 DONE 없이 중단(미완성) | `$IJCAS_DATA/runs/test2/` (DONE이 있는 15회만 유효) | 같은 명령으로 재실행하면 DONE이 없는 실행부터 이어서 수집한다(미완성 디렉터리는 도구가 지우고 다시 만든다). 이 시드들은 아직 어떤 평가에도 쓰지 않았다. |
| v2 학습 (`results/rgat_v2/train_all.sh`) | `rgat_dyn` 시드 0 학습 중(약 40/60 에폭). 완성된 모델 없음 | `results/rgat_v2/runs/rgat_dyn/seed0/train.log`(부분 기록) | 재개하지 않는다(아래 이유) |
| 후속 자동 작업 (`results/rgat_v2/after_wait.sh`) | 학습 종료 대기 중 | 없음 | 재개하지 않는다 |

종료 방법: PID를 지정해 SIGTERM을 보냈고, 응답하지 않은 Isaac Sim과 PX4 프로세스는 SIGKILL로 종료했다. 패턴으로 일괄 종료하는 방식은 쓰지 않았다.

## v2(IMU 품질 적격성) 실험을 재개하지 않는 이유

중단 직전에 사후 분석을 했다. 대상은 이전 시험 세트(시드 101–130)이고, 사전 등록에 없던 분석이다. v2 정책만 적용했을 때 결과는 다음과 같다.

| | v1 정책 | v2 정책 |
|---|---|---|
| `rgat_dyn`(v1 모델) 전체 RMSE | 1.127 m | 1.769 m |
| Oracle 전체 RMSE | 0.833 m | 1.321 m |

- 원인: 포화 구간에서 GNSS/INS까지 오염된 실행(13–24 m)이 있다. 이런 실행에서 v2 정책이 LIO·INS를 빼면 남는 후보가 오염된 GNSS/INS뿐이 된다.
- v1 `rgat_dyn`은 결함 뒤 발산한 LIO를 이미 대부분 피하고 있었다. 그래서 정책이 얻는 이득은 작았다.
- 이에 따라 IMU 포화에 의한 후보 배제는 **기본값에서 끈다**(`config` 키 `eligibility.imu_saturation.enabled`, 기본 false). 코드와 시험은 남겨 둔다. IMU 수신 검사는 유지한다(기존 기록에서 평가 구간 영향 없음).

## 보존한 것 (변경하지 않음)

- v1 동결 모델·결과: `results/rgat/` (FROZEN.json 포함). 동결 v1 코드는 커밋 `878eac1`.
- v1 데이터셋: `$IJCAS_DATA/rgat/dataset/{validation,test}`.
- v2 데이터셋(IMU 포화 정책판): `$IJCAS_DATA/rgat/dataset_v2/`. 이전 판은 `$IJCAS_DATA/rgat/dataset_v2_superseded_20261118a/`.
- 수집 기록: `$IJCAS_DATA/runs/{validation,test,test2}`.
