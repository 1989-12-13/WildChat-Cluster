#!/usr/bin/env bash
# 下载 Spark 3.5.9 -> 检查完整性 -> 构建镜像
# 全程写日志，可重复执行（幂等）。
set -u
cd "$(dirname "$0")" || exit 1
BUILD=~/wildchat_bench/docker/build
LOG=$BUILD/spark_download.log
TGZ=$BUILD/spark-3.5.9-bin-hadoop3.tgz
EXPECTED=401379923

say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

say "=== 目标: Spark 3.5.9 ($(( EXPECTED/1024/1024 )) MB) ==="

# --- 已完整？ ---
if [ -f "$TGZ" ]; then
    sz=$(stat -c%s "$TGZ")
    if [ "$sz" -eq "$EXPECTED" ]; then
        if tar -tzf "$TGZ" >/dev/null 2>&1; then
            say "已存在且完整 ($(( sz/1024/1024 )) MB)，跳过下载"
        else
            say "文件大小对但 tar 损坏，重下"
            rm -f "$TGZ"
        fi
    else
        say "存在但不完整 ($(( sz/1024/1024 )) MB)，续传"
    fi
fi

# --- 下载（多源轮询，续传） ---
SOURCES=(
  "https://mirrors.aliyun.com/apache/spark/spark-3.5.9/spark-3.5.9-bin-hadoop3.tgz"
  "https://mirrors.huaweicloud.com/apache/spark/spark-3.5.9/spark-3.5.9-bin-hadoop3.tgz"
  "https://mirrors.tuna.tsinghua.edu.cn/apache/spark/spark-3.5.9/spark-3.5.9-bin-hadoop3.tgz"
)

attempt=0
while :; do
    sz=$(stat -c%s "$TGZ" 2>/dev/null || echo 0)
    if [ "$sz" -ge "$EXPECTED" ]; then break; fi
    if [ "$attempt" -ge 12 ]; then say "重试上限，退出"; break; fi
    src=${SOURCES[$(( attempt % ${#SOURCES[@]} ))]}
    attempt=$((attempt+1))
    say "尝试 $attempt: $(echo "$src" | cut -d/ -f3)  (已有 $(( sz/1024/1024 )) MB)"
    curl -sS -L --noproxy '*' --max-time 900 --speed-limit 51200 --speed-time 60 \
         -C - -o "$TGZ" "$src" >>"$LOG" 2>&1
    rc=$?
    sz=$(stat -c%s "$TGZ" 2>/dev/null || echo 0)
    say "  curl rc=$rc, 现在 $(( sz/1024/1024 )) MB"
    if [ "$rc" -ne 0 ] && [ $(( sz/1024/1024 )) -lt $(( EXPECTED/1024/1024 )) ]; then
        # 速度过低或失败，换源继续
        sleep 3
    fi
done

sz=$(stat -c%s "$TGZ" 2>/dev/null || echo 0)
say "最终大小: $(( sz/1024/1024 )) MB (期望 $(( EXPECTED/1024/1024 )) MB)"
if ! tar -tzf "$TGZ" >/dev/null 2>&1; then
    say "FATAL: tar 不完整，放弃"
    exit 1
fi
say "tar 完整性 OK"

# --- 构建镜像 ---
say "=== 开始构建镜像 ==="
cd "$BUILD" || exit 1
# Dockerfile 里引用的 tar 名字需要匹配
if grep -q "spark-3.5.5-bin-hadoop3.tgz" Dockerfile 2>/dev/null; then
    say "更新 Dockerfile 引用 3.5.9"
    sed -i 's/spark-3\.5\.5-bin-hadoop3\.tgz/spark-3.5.9-bin-hadoop3.tgz/g; s|spark-3\.5\.5-bin-hadoop3|spark-3.5.9-bin-hadoop3|g' Dockerfile
fi

DOCKER_BUILDKIT=1 docker build -t bigdata/spark:3.5.5-hadoop3.3.6 \
    -f Dockerfile . >>"$LOG" 2>&1
rc=$?
say "docker build rc=$rc"

if [ $rc -eq 0 ]; then
    say "=== 验证镜像 ==="
    docker run --rm --entrypoint bash bigdata/spark:3.5.5-hadoop3.3.6 -c '
      echo -n "java:   "; java -version 2>&1 | head -1
      echo -n "hadoop: "; hadoop version 2>&1 | head -1
      echo -n "spark:  "; ls /usr/local/spark/jars | wc -l
      echo -n "workers:"; cat $HADOOP_CONF_DIR/workers | tr "\n" " "; echo
    ' 2>&1 | tee -a "$LOG"
    say "=== 全部完成 ==="
else
    say "=== 构建失败，见日志 ==="
    tail -30 "$LOG"
fi
