#!/usr/bin/env bash
set -u

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
cd "$SCRIPT_DIR"

if docker container inspect hive >/dev/null 2>&1; then
  docker exec hive bash -lc '
    pid=$(pgrep -f "[o]rg.apache.hive.service.server.HiveServer2" | head -n 1)
    if [ -n "$pid" ]; then
      kill -TERM "$pid"
    fi
  ' 2>/dev/null || true

  for attempt in $(seq 1 30); do
    if ! docker exec hive bash -lc \
      'pgrep -f "[o]rg.apache.hive.service.server.HiveServer2" >/dev/null'; then
      break
    fi
    sleep 1
  done
fi

docker compose -f compose-hive.yaml stop
echo "Hive 与 MySQL 容器已停止，元数据库命名卷和 HDFS 仓库数据均保留。"
