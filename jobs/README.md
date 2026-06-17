# Pipeline Jobs

실행 가능한 job은 파이프라인 단계별로 묶어둔다. 파일 트리만 봐도 아키텍처 흐름이 보이도록 구성했다.

```text
snapshot/       Source LeRobot snapshot 다운로드, contract 검증, 구조 검사
lakehouse/      Raw parquet와 metadata를 Iceberg table로 적재
sync/           Sensor sync 및 observation/action window sample 생성
annotations/    Instruction annotation metadata 생성 및 검사
manifest/       Logical dataset manifest 생성, resolve, 검증
export/         LeRobot 스타일 training dataset export 및 검증
pipeline/       Robotis MVP end-to-end pipeline 실행
```

모든 job은 repository root에서 Docker Compose를 통해 실행한다.

```bash
docker compose run --rm app -lc "python jobs/lakehouse/ingest_raw_to_iceberg.py"
```

Robotis MVP 전체 파이프라인은 아래 명령으로 실행한다.

```bash
docker compose run --rm app -lc "python jobs/pipeline/run_robotis_mvp.py"
```
