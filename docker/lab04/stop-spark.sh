#!/usr/bin/env bash
set -u

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
cd "$SCRIPT_DIR"

if docker ps --format '{{.Names}}' | grep -qx master; then
  docker exec master /usr/local/spark/sbin/stop-all.sh || true
  docker exec master /usr/local/hadoop/sbin/stop-yarn.sh || true
  docker exec master /usr/local/hadoop/sbin/stop-dfs.sh || true
fi

docker compose -f compose.yaml stop
echo "Spark、HDFS、YARN 和三个容器已停止；HDFS 命名卷仍然保留。"
