#!/usr/bin/env bash
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
cd "$SCRIPT_DIR"

IMAGE=bigdata/hbase:2.6.1-hadoop3.3.6
NETWORK=lab02-student_hadoopnet

if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
  echo "错误：未找到镜像 $IMAGE。" >&2
  exit 1
fi

if ! docker network inspect "$NETWORK" >/dev/null 2>&1; then
  echo "错误：未找到实验二网络 $NETWORK。请先恢复实验二环境。" >&2
  exit 1
fi

for volume_name in \
  lab02-student_hadoop-master-data \
  lab02-student_hadoop-worker1-data \
  lab02-student_hadoop-worker2-data; do
  if ! docker volume inspect "$volume_name" >/dev/null 2>&1; then
    echo "错误：未找到实验二数据卷 $volume_name。停止启动以避免创建空集群。" >&2
    exit 1
  fi
done

docker compose -f compose.yaml up -d

ssh_ready=0
for attempt in $(seq 1 30); do
  if docker exec master ssh master true >/dev/null 2>&1 \
    && docker exec master ssh worker1 true >/dev/null 2>&1 \
    && docker exec master ssh worker2 true >/dev/null 2>&1; then
    ssh_ready=1
    break
  fi
  sleep 1
done

if [ "$ssh_ready" -ne 1 ]; then
  echo "错误：三个节点没有在 30 秒内完成 SSH 就绪。" >&2
  docker compose -f compose.yaml ps
  exit 1
fi

if ! docker exec master test -f /data/hadoop/dfs/name/current/VERSION; then
  echo "错误：未找到实验二的 NameNode 元数据；为保护数据，本脚本不会自动格式化。" >&2
  exit 1
fi

docker exec master /usr/local/hadoop/sbin/start-dfs.sh
docker exec master /usr/local/hadoop/sbin/start-yarn.sh

hadoop_ready=0
for attempt in $(seq 1 60); do
  if docker exec master hdfs dfsadmin -report 2>/dev/null | grep -q 'Live datanodes (2)' \
    && docker exec master yarn node -list 2>/dev/null | grep -q 'Total Nodes:2'; then
    hadoop_ready=1
    break
  fi
  sleep 2
done

if [ "$hadoop_ready" -ne 1 ]; then
  echo "错误：HDFS/YARN 没有在预期时间内恢复健康。" >&2
  docker exec master jps || true
  docker exec worker1 jps || true
  docker exec worker2 jps || true
  exit 1
fi

echo
echo "===== Hadoop 状态 ====="
docker exec master hdfs dfsadmin -report | grep 'Live datanodes'
docker exec master yarn node -list
echo
echo "HDFS 与 YARN 已启动，HBase 未启动。"

