#!/usr/bin/env bash
# 修复 Docker DNS 反向解析不一致导致的 RegionServer 启动失败：
#   Master passed us a different hostname to use; was=worker1, but now=worker1.lab02-student_hadoopnet
# 做法：为三个容器补齐彼此的短名映射（172.18.0.x worker1），使正/反向解析一致。
# 说明：容器每次启动都会重新生成 /etc/hosts，所以必须在容器启动后执行本脚本。
set -u

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
cd "$SCRIPT_DIR"

echo
echo "===== 统一三节点主机名解析 ====="
for node in master worker1 worker2; do
  for peer in master worker1 worker2; do
    peer_ip=$(docker exec "$node" getent hosts "$peer" 2>/dev/null | awk '{print $1}' | head -1)
    if [ -z "$peer_ip" ]; then
      echo "警告：$node 无法解析 $peer，跳过。" >&2
      continue
    fi
    docker exec "$node" bash -c "grep -qE '[[:space:]]$peer\$' /etc/hosts || echo '$peer_ip $peer' >> /etc/hosts"
  done
done

for node in master worker1 worker2; do
  echo "--- $node ---"
  docker exec "$node" getent hosts master worker1 worker2 || true
done
