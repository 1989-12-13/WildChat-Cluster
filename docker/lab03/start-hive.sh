#!/usr/bin/env bash
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
cd "$SCRIPT_DIR"

COMPOSE_FILE=compose-hive.yaml
HADOOP_NETWORK=lab02-student_hadoopnet

for image_name in mysql:8.0.43 bigdata/hive:4.0.1-hadoop3.3.6; do
  if ! docker image inspect "$image_name" >/dev/null 2>&1; then
    echo "错误：未找到镜像 $image_name。" >&2
    exit 1
  fi
done

if ! docker network inspect "$HADOOP_NETWORK" >/dev/null 2>&1; then
  echo "错误：未找到 Hadoop 网络 $HADOOP_NETWORK。" >&2
  exit 1
fi

if ! docker exec master hdfs dfsadmin -report 2>/dev/null | grep -q 'Live datanodes (2)'; then
  echo "错误：HDFS 当前不是 2 台 Live DataNode，停止启动 Hive。" >&2
  exit 1
fi

docker compose -f "$COMPOSE_FILE" up -d

mysql_ready=0
for attempt in $(seq 1 60); do
  mysql_status=$(docker inspect hive-mysql --format '{{.State.Health.Status}}' 2>/dev/null || true)
  if [ "$mysql_status" = healthy ]; then
    mysql_ready=1
    break
  fi
  sleep 2
done

if [ "$mysql_ready" -ne 1 ]; then
  echo "错误：MySQL 没有在预期时间内进入 healthy 状态。" >&2
  docker logs --tail 80 hive-mysql >&2 || true
  exit 1
fi

docker exec master hdfs dfs -mkdir -p /tmp /user/hive/warehouse
docker exec master hdfs dfs -chmod 1777 /tmp /user/hive/warehouse

if docker exec hive /usr/local/hive/bin/schematool -dbType mysql -info >/dev/null 2>&1; then
  echo "Hive metastore schema 已存在。"
else
  echo "正在初始化 Hive metastore schema..."
  docker exec hive /usr/local/hive/bin/schematool -dbType mysql -initSchema --verbose
fi

docker exec hive bash -lc '
  if ! pgrep -f "[o]rg.apache.hive.service.server.HiveServer2" >/dev/null; then
    nohup /usr/local/hive/bin/hiveserver2 \
      > /usr/local/hive/logs/hiveserver2.log 2>&1 &
  fi
'

hiveserver_ready=0
for attempt in $(seq 1 90); do
  if docker exec hive bash -lc 'timeout 1 bash -c "</dev/tcp/127.0.0.1/10000"' >/dev/null 2>&1; then
    hiveserver_ready=1
    break
  fi
  sleep 2
done

if [ "$hiveserver_ready" -ne 1 ]; then
  echo "错误：HiveServer2 没有在预期时间内监听 10000 端口。" >&2
  docker exec hive tail -n 120 /usr/local/hive/logs/hiveserver2.log >&2 || true
  exit 1
fi

echo "Hive 服务已启动。"
echo "容器内 JDBC: jdbc:hive2://localhost:10000/default"
echo "虚拟机对外 JDBC: jdbc:hive2://<虚拟机IP>:10001/default"
