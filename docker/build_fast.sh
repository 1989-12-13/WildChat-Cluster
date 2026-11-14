#!/usr/bin/env bash
# 用国内 apt 镜像 + 预下载的 tar 构建镜像
# 目标：避免 archive.ubuntu.com 的慢速下载
set -u
BUILD=~/wildchat_bench/docker/build
LOG=$BUILD/build2.log
cd "$BUILD" || exit 1

say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
say "===== 开始构建（国内 apt 源） ====="

# 先测国内 apt 源可用性
say "--- 探测 apt 镜像 ---"
MIRROR=""
for m in \
  "https://mirrors.aliyun.com/ubuntu" \
  "https://mirrors.tuna.tsinghua.edu.cn/ubuntu" \
  "https://repo.huaweicloud.com/ubuntu" ; do
    code=$(curl -s -m 10 --noproxy '*' -o /dev/null -w "%{http_code}" "$m/dists/noble/Release" 2>/dev/null)
    spd=$(curl -s -m 12 --noproxy '*' -o /dev/null -w "%{speed_download}" -r 0-3145727 "$m/dists/noble/main/binary-amd64/Packages.gz" 2>/dev/null)
    say "  $(echo $m|cut -d/ -f3) http=$code speed=${spd}B/s"
    if [ "$code" = "200" ] && [ "${spd:-0}" -gt 200000 ] && [ -z "$MIRROR" ]; then
        MIRROR=$m
    fi
done
if [ -z "$MIRROR" ]; then
    say "未找到足够快的镜像，回退到 archive.ubuntu.com"
    MIRROR="http://archive.ubuntu.com/ubuntu"
fi
say "选用镜像: $MIRROR"

# 生成使用该镜像的 Dockerfile
sed "s|http://archive.ubuntu.com/ubuntu|$MIRROR|g" Dockerfile > Dockerfile.fast
# 在 apt 安装前插入 sed 替换源
python3 - "$MIRROR" <<'PYEOF'
import sys, re
mirror = sys.argv[1]
src = open('Dockerfile.fast').read()
# 把 apt-get update 之前插入改写 sources 的步骤
inject = f"""# 使用国内镜像替换 apt 源
RUN sed -i 's|http://archive.ubuntu.com/ubuntu|{mirror}|g; s|http://security.ubuntu.com/ubuntu|{mirror}|g' \\
        /etc/apt/sources.list.d/ubuntu.sources 2>/dev/null || \\
    sed -i 's|http://archive.ubuntu.com/ubuntu|{mirror}|g; s|http://security.ubuntu.com/ubuntu|{mirror}|g' \\
        /etc/apt/sources.list 2>/dev/null || true

"""
src = src.replace("RUN apt-get update -qq &&", inject + "RUN apt-get update -qq &&", 1)
open('Dockerfile.fast','w').write(src)
print("Dockerfile.fast 已生成")
PYEOF

say "--- 开始 docker build ---"
DOCKER_BUILDKIT=1 docker build -t bigdata/spark:3.5.5-hadoop3.3.6 \
    -f Dockerfile.fast . >>"$LOG" 2>&1
rc=$?
say "docker build rc=$rc"

if [ $rc -eq 0 ]; then
    say "===== 验证镜像 ====="
    docker run --rm --entrypoint bash bigdata/spark:3.5.5-hadoop3.3.6 -c '
      echo -n "java:    "; java -version 2>&1 | head -1
      echo -n "hadoop:  "; hadoop version 2>&1 | head -1
      echo -n "spark jars: "; ls /usr/local/spark/jars | wc -l
      echo -n "workers: "; cat $HADOOP_CONF_DIR/workers | tr "\n" " "; echo
      echo -n "defaultFS: "; grep -A1 fs.defaultFS $HADOOP_CONF_DIR/core-site.xml | tail -1 | sed "s/.*<value>\(.*\)<\/value>.*/\1/"
    ' 2>&1 | tee -a "$LOG"
    say "===== 构建成功 ====="
else
    say "===== 构建失败 ====="
    tail -40 "$LOG"
fi
