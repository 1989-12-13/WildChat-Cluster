#!/usr/bin/env bash
# 启动 WildChat 三节点 Hadoop + Spark 集群
#
# 由课程脚本拼接而成：
#   lab02-student/start-hadoop.sh  （HDFS + YARN）
#   lab04/start-spark.sh           （Spark）
# 实验服务器 上两者使用同一镜像 bigdata/spark:3.5.5-hadoop3.3.6，
# 因此合并为一个脚本；启动顺序、等待与判据保持与课程一致。
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
cd "$SCRIPT_DIR"

IMAGE=bigdata/spark:3.5.5-hadoop3.3.6

if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
  echo "错误：未找到镜像 $IMAGE。请先执行 ./build-image.sh。" >&2
  exit 1
fi

mkdir -p shared
docker compose up -d

# ---------- 以下对应 lab02-student/start-hadoop.sh ----------
ready=0
for attempt in $(seq 1 30); do
  if docker exec master ssh master true >/dev/null 2>&1 \
    && docker exec master ssh worker1 true >/dev/null 2>&1 \
    && docker exec master ssh worker2 true >/dev/null 2>&1; then
    ready=1
    break
  fi
  sleep 1
done

if [ "$ready" -ne 1 ]; then
  echo "错误：三个节点没有在 30 秒内完成 SSH 就绪，请执行 docker compose ps 检查容器。" >&2
  exit 1
fi

if ! docker exec master test -f /data/hadoop/dfs/name/current/VERSION; then
  echo "首次启动：正在格式化 NameNode。"
  docker exec master /usr/local/hadoop/bin/hdfs namenode -format -force -nonInteractive
fi

docker exec master /usr/local/hadoop/sbin/start-dfs.sh
docker exec master /usr/local/hadoop/sbin/start-yarn.sh

# ---------- 以下对应 lab04/start-spark.sh ----------
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
  exit 1
fi

docker exec master /usr/local/spark/sbin/start-all.sh

spark_ready=0
for attempt in $(seq 1 30); do
  if docker exec master jps | grep -qxE '[0-9]+ Master' \
    && docker exec master jps | grep -qxE '[0-9]+ Worker' \
    && docker exec worker1 jps | grep -qxE '[0-9]+ Worker' \
    && docker exec worker2 jps | grep -qxE '[0-9]+ Worker'; then
    spark_ready=1
    break
  fi
  sleep 1
done

if [ "$spark_ready" -ne 1 ]; then
  echo "错误：Spark Master/Worker 没有在 30 秒内全部就绪。" >&2
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
echo "===== Spark 进程 ====="
echo "--- master ---"
docker exec master jps
echo "--- worker1 ---"
docker exec worker1 jps
echo "--- worker2 ---"
docker exec worker2 jps
echo
echo "集群已启动。"
echo "NameNode: http://127.0.0.1:19870"
echo "YARN:     http://127.0.0.1:19088"
echo "Spark Master: spark://master:7077"
echo "Spark Web UI: http://127.0.0.1:19080"
