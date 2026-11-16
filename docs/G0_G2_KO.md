# G0–G2: 환경 고정 · 공통 센서 · 정상 항법

작성일: 2026-11-15 · 근거 파일은 모두 `results/` 아래에 있다.

## G0 버전 고정 · 공식 예제 재현 — 진행 중 (GLIM 공식 예제만 남음)

| 구성 | 버전 / commit |
|---|---|
| OS·HW | Ubuntu 22.04.5, i7-13650HX 20코어, 31 GB, RTX 4060 Laptop 8 GB, 드라이버 570.172.08, CUDA 12.8 |
| Isaac Sim | 5.1.0 (소스 빌드, 47d886f2) |
| Pegasus | v5.1.0 (644da37e) |
| PX4 | v1.14.3 (1dacb4cd), SITL `gazebo-classic_iris`, lockstep |
| DDS | Micro XRCE-DDS Agent v2.4.2, px4_msgs release/1.14, ROS 2 Humble, Fast DDS, 도메인 42 |
| OpenVINS | v2.7 (93adc241), colcon Release |
| GLIM | 1.2.2 CPU, 공식 PPA deb를 sudo 없이 사용자 경로에 풂(SHA256 기록) |

`results/g0/env_manifest.json`이 전체 목록이다.

- **OpenVINS:** 공식 시뮬레이터 예제(`run_simulation`, rpng_sim, 295.75 m)를 재현했다. 위치 RMSE 0.045 m, 자세 0.182°이고, 독립 재계산(`tools/g0_check_openvins.py`)도 같은 값이다. EuRoC는 이 네트워크에서 받을 수 없었다(ETH Research Collection 429, Drive 404).
- **PX4 uXRCE-DDS + px4_msgs:** `/fmu/out/*`를 도메인 42에서 수신·디코딩했다(`results/g0/px4_probe/dds.json`).
- **GLIM 공식 예제:** 데이터(Zenodo 403, AIST 연결 불가)가 막혀 미완료다. 데이터를 받으면 `~/ijcas_data/g0/glim/`에 둔다.

## G1 공통 센서 · 시간·좌표계 — 완료

구조는 다음과 같다.
- **강제 궤적 기체(1단계):** PhysX를 끈 Pegasus 기체가 해석적 기준 궤적을 그대로 따른다. 궤적은 호 길이 기준 주기 5차 스플라인이고, 자세는 미분 평탄성(추력축 = a + g)으로 정한다.
- **공통 측정:** 같은 측정이 PX4(HIL)와 ROS 2로 나간다.
- **카메라:** stereo 640×480, 20 Hz, baseline 0.12 m, 15° 하향.
- **LiDAR:** Ouster OS1 32ch, 10 Hz, RTX.
- **나머지 센서:** IMU 200 Hz, GNSS 5 Hz, 기압계·자력계 50 Hz.

| 기준 | 결과 (`results/g1/checks_*.json`) |
|---|---|
| 공통 IMU·GNSS | ROS fix는 PX4 GPS_RAW_INT와 1e-7° 안에서 같다. IMU는 PX4 HIGHRES_IMU와 잡음 수준 차이다(가속도 σ 0.015–0.025, 자이로 σ 0.0011–0.0017). |
| stereo 보정 | 스테이지에서 측정했다. K = [320, 320, 320, 240], 장착 오차 6e-6°. 에피폴라 p90 ≤ 1.44 px, 양의 시차 81–100 % |
| LiDAR 시각 | 스캔 주기 0.100 s, 점별 t ∈ {0, 0.05}(실제 렌더 시각). 이동 중 바닥 높이 평균 0.6–4.6 cm, σ 4–10 cm |
| GT 누설 없음 | IMU에 자세 없음(공분산 −1). PX4 EV/OF/RNG 끔. 엔진 실제 구독: OpenVINS `/cam0` `/cam1` `/ijcas/imu` `/clock`, GLIM `/ijcas/lidar/points` `/ijcas/imu` `/clock` |

구현 중 고친 문제는 다음과 같다. 재발을 막기 위해 남긴다.
1. **강제 궤적의 자세 갱신:** dynamic control로 순간이동시키면 Pegasus가 읽는 USD 자세가 91 % 스텝에서 낡아 있었다. 그래서 물리를 끈 기체를 쓴다.
2. **센서 주기 판정:** float32 물리 dt(0.004999999888) 때문에 Pegasus `update_at_rate`가 200 Hz IMU를 100 Hz로 만들었다. 그 결과 PX4 lockstep 시계가 0.5배로 흘렀다. 허용 오차를 둔 `RateGate`와 IMU 표본 시각을 쓰는 PX4 백엔드로 고쳤다.
3. **PX4 DDS 타임스탬프:** v1.14 DDS 클라이언트는 송신 타임스탬프에 벽시계 동기를 항상 적용한다(끄는 파라미터 없음). 그래서 lockstep 기록은 MAVLink `time_boot_ms`로 받는다.
4. **비동기 렌더 지연:** 비동기 렌더링에서 영상·점군이 최대 0.2 s 늦게 나왔다. 동기 렌더링으로 바꾸고 `rendering_time`으로 찍는다.
5. **카메라 갱신률:** Isaac 카메라 `frequency=20`도 같은 float 문제로 10 Hz가 됐다. 주기를 주지 않는다.
6. **rclpy 대입 속도:** `bytes` 대입이 56 ms 걸렸다(`array.array`는 0.014 ms). 이를 고쳐 RTF가 0.21에서 0.92가 됐다.
7. **고아 엔진 프로세스:** `ros2 run` 래퍼를 종료해도 엔진 자식이 살아남아 중복 발행했다. 프로세스 그룹으로 실행하고 정리한다.
8. **점군 QoS:** best-effort QoS로 640 kB 점군이 조각 손실돼 프레임이 빠졌다. GLIM 구독을 reliable로 바꿨다.

## G2 정상 항법 · 기준선 — 완료

S00(정상, 180 s), 세 장면, 시드 1. 근거는 `results/g2/S00_*_1.json`이고 도구는 `tools/g2_report.py`다.

| 장면 | OpenVINS 첫 출력 / 초기정렬 RMSE / 10 m 드리프트 | GLIM 첫 출력 / 지연 / RMSE | PX4 RMSE / 지연 | GNSS 단독 | INS-DR 10·30·60 s |
|---|---|---|---|---|---|
| 창고 | 11.8 s / 0.64 m / 0.18 m | 3.3 s / 120 ms / 0.02 m | 0.94 m / 40 ms | 0.91 m | 4.4 / 31.7 / 139 m |
| 개활지 | 12.3 s / 1.91 m / 0.16 m | 3.3 s / 95 ms / 0.03 m | 1.01 m / 41 ms | 0.91 m | 4.5 / 35.7 / 105 m |
| 복도 | 20.8 s / 0.83 m / 0.06 m | 3.3 s / 115 ms / 0.09 m | 1.02 m / 36 ms | 0.91 m | 4.3 / 37.6 / 121 m |

### 출력 상태 정의

| 후보 | 프레임 | 출력 |
|---|---|---|
| PX4 EKF2 | PX4 국소 NED(원점 = GPS_GLOBAL_ORIGIN) → ENU | 위치·속도·자세 50 Hz, 상태 플래그·혁신 비율 10 Hz. 원래 공분산 외에 정확도 값 제공 |
| OpenVINS | 자체 중력 정렬 odom 프레임 | IMU 전파 오도메트리 200 Hz, 추적 특징점 수 |
| GLIM | 자체 odom 프레임 | 오도메트리 10 Hz(스캔 시작 시각, 처리 지연 약 0.1 s). 공분산 없음 |
| GNSS 단독 | 원점 접평면 | 위치·속도 5 Hz. 결측이면 출력 없음 |
| INS-DR | 마지막 선택 상태에서 출발 | 적분만 함. 바이어스 추정 없는 MEMS라 10 s 후 약 4 m |

### 초기화 설정

OpenVINS 정적 초기화는 정지 상태의 급변(jerk)을 요구한다. 그런데 이 리그는 정지 비행에서 부드럽게 출발하므로, 공식 동적 초기화(`init_dyn_use`)를 켜고 jerk 임계를 0.5로 낮췄다. 이 설정은 시험 전에 고정했다.

### 해석

- 시뮬레이션 LiDAR는 깨끗해서, 구조가 있는 한 GLIM이 cm 수준이다.
- VIO는 0.6–1.6 %/10 m 드리프트, PX4는 GNSS 잡음 수준(약 1 m)이다.
- 정상 조건의 선택 문제는 쉽다. 연구의 핵심은 열화 시나리오(S01–S10)다.
