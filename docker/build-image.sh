#!/usr/bin/env bash
# 构建 bigdata/spark:3.5.5-hadoop3.3.6（复刻课程镜像）
#
# 前置：build/ 目录下已有
#   hadoop-3.3.6.tar.gz
#   spark-3.5.5-bin-hadoop3.tgz
# 以及 conf/ 下的 XML 配置。
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
BUILD_DIR="$SCRIPT_DIR/build"
cd "$BUILD_DIR"

IMAGE=bigdata/spark:3.5.5-hadoop3.3.6

# ---- 检查 tar 包 ----
for f in hadoop-3.3.6.tar.gz spark-3.5.5-bin-hadoop3.tgz; do
  if [ ! -f "$f" ]; then
    echo "错误：缺少 $f" >&2
    echo "请先下载：" >&2
    echo "  curl -L -o $f https://mirrors.tuna.tsinghua.edu.cn/apache/..." >&2
    exit 1
  fi
  # tar 完整性快检（gzip 魔数 + 可列出末尾）
  if ! tar -tzf "$f" >/dev/null 2>&1; then
    echo "错误：$f 不是完整的 tar.gz（可能下载未完成）" >&2
    exit 1
  fi
  echo "OK  $f  $(du -h "$f" | cut -f1)"
done

echo
echo "===== 清理旧镜像（可选，避免层堆积） ====="
if docker image inspect "$IMAGE" >/dev/null 2>&1; then
  echo "已存在同名镜像，将重建"
  docker rmi -f "$IMAGE" >/dev/null 2>&1 || true
fi

echo
echo "===== 构建镜像 $IMAGE ====="
DOCKER_BUILDKIT=1 docker build \
  --build-arg BUILDKIT_INLINE_CACHE=1 \
  -t "$IMAGE" \
  -f "$BUILD_DIR/Dockerfile" \
  "$BUILD_DIR"

echo
echo "===== 验证 ====="
docker run --rm --entrypoint bash "$IMAGE" -c '
  echo -n "java:    "; java -version 2>&1 | head -1
  echo -n "hadoop:  "; hadoop version 2>&1 | head -1
  echo -n "spark:   "; ls /usr/local/spark/jars 2>/dev/null | wc -l | xargs echo "jars ="
  echo -n "workers: "; cat $HADOOP_CONF_DIR/workers | tr "\n" " "; echo
  echo -n "fs.defaultFS: "; grep -A2 fs.defaultFS $HADOOP_CONF_DIR/core-site.xml | grep value | sed "s/.*<value>\(.*\)<\/value>.*/\1/"
'

echo
echo "镜像构建完成：$IMAGE"
echo "下一步：./start-cluster.sh"
