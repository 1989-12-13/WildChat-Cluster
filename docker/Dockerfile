# 复刻课程镜像 bigdata/spark:3.5.5-hadoop3.3.6
#
# 依据：从 WSL 课程镜像提取的 Config 与 /usr/local/hadoop/etc/hadoop 配置
#   base   : ubuntu:24.04
#   java   : openjdk-11（课程用 JDK 11.0.32；Hadoop 3.3.6 不支持 17/21）
#   hadoop : 3.3.6     -> /usr/local/hadoop
#   spark  : 3.5.9     -> /usr/local/spark   （课程为 3.5.5，见 README）
#   CMD    : /usr/sbin/sshd -D -e
FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive

# 先替换 apt 源为国内镜像（Ubuntu 24.04 用 deb822 格式的 ubuntu.sources）
# 注意：这里必须显式使用 aliyun 镜像，否则 archive.ubuntu.com 会极慢
RUN set -eux; \
    MIRROR=http://mirrors.aliyun.com/ubuntu; \
    sed -i "s|http://archive.ubuntu.com/ubuntu|${MIRROR}|g" /etc/apt/sources.list.d/ubuntu.sources; \
    sed -i "s|http://security.ubuntu.com/ubuntu|${MIRROR}|g" /etc/apt/sources.list.d/ubuntu.sources; \
    grep -E "^URIs:" /etc/apt/sources.list.d/ubuntu.sources

# 单层安装全部依赖（含 JDK 11）
# python3-setuptools 是必需的：Python 3.12 移除了 distutils，
# 而 PySpark 3.5 的 pyspark.ml 仍会 import distutils，缺失会直接报
# ModuleNotFoundError: No module named 'distutils'
RUN set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends \
        openjdk-11-jdk-headless \
        openssh-server openssh-client \
        curl ca-certificates procps net-tools vim-tiny less \
        python3 python3-numpy python3-pandas python3-pip \
        python3-setuptools tini; \
    rm -rf /var/lib/apt/lists/*

ENV JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64

# 文本处理用的 Python 包（全部为成熟公开实现，非自写）：
#   jieba        中文分词。Spark 的 RegexTokenizer 无词边界概念，必须另配分词器
#   stopwordsiso 多语言停用词表（58 种，含中文 794 词）。Spark 官方词表只有
#                14 种语言且不含中文，用它补缺口
#   pycountry    语言名 -> ISO 639-1 码，供查 stopwordsiso，避免手写语言映射表
# 必须写进镜像：这些包装在容器的可写层时，一旦 docker compose down 或重建
# 容器就会丢失，任务要到运行时才报 ModuleNotFoundError。
# 用 aliyun pip 镜像：容器内直连 PyPI 实测极慢（单个 jieba 超过 15 分钟）。
RUN set -eux; \
    pip3 install --break-system-packages --no-cache-dir \
        -i https://mirrors.aliyun.com/pypi/simple/ \
        jieba==0.42.1 stopwordsiso==0.7.1 pycountry==26.2.16

# Hadoop + Spark（tar 包由宿主机预下载后 COPY，镜像内不联网）
COPY hadoop-3.3.6.tar.gz /tmp/hadoop.tgz
COPY spark-3.5.9-bin-hadoop3.tgz /tmp/spark.tgz
RUN set -eux; \
    tar -xzf /tmp/hadoop.tgz -C /opt; \
    mv /opt/hadoop-3.3.6 /usr/local/hadoop; \
    tar -xzf /tmp/spark.tgz -C /opt; \
    mv /opt/spark-3.5.9-bin-hadoop3 /usr/local/spark; \
    rm -f /tmp/hadoop.tgz /tmp/spark.tgz

ENV HADOOP_HOME=/usr/local/hadoop \
    HADOOP_CONF_DIR=/usr/local/hadoop/etc/hadoop \
    HADOOP_COMMON_HOME=/usr/local/hadoop \
    HADOOP_HDFS_HOME=/usr/local/hadoop \
    HADOOP_MAPRED_HOME=/usr/local/hadoop \
    HADOOP_YARN_HOME=/usr/local/hadoop \
    SPARK_HOME=/usr/local/spark \
    PATH=/usr/local/spark/bin:/usr/local/spark/sbin:/usr/local/hadoop/bin:/usr/local/hadoop/sbin:/usr/lib/jvm/java-11-openjdk-amd64/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

# 与课程镜像一致的配置
COPY conf/core-site.xml   $HADOOP_CONF_DIR/core-site.xml
COPY conf/hdfs-site.xml   $HADOOP_CONF_DIR/hdfs-site.xml
COPY conf/yarn-site.xml   $HADOOP_CONF_DIR/yarn-site.xml
COPY conf/mapred-site.xml $HADOOP_CONF_DIR/mapred-site.xml
COPY conf/workers         $HADOOP_CONF_DIR/workers

# hadoop-env.sh / spark-env.sh 必须在镜像内写死 JAVA_HOME 与运行用户：
# sbin/start-*.sh 经 ssh 拉起远程守护进程时不继承容器 ENV。
COPY conf/hadoop-env.sh $HADOOP_CONF_DIR/hadoop-env.sh
COPY conf/spark-env.sh  $SPARK_HOME/conf/spark-env.sh

# Spark 的 worker 列表：start-all.sh 读取 $SPARK_HOME/conf/workers 决定
# 在哪些主机上启动 Worker。课程镜像含 master/worker1/worker2 三行。
COPY conf/spark-workers $SPARK_HOME/conf/workers
COPY conf/spark-defaults.conf $SPARK_HOME/conf/spark-defaults.conf

# Hadoop 3.x 以 root 运行 sbin/start-dfs.sh 时必须显式声明用户，
# 否则报 "there is no HDFS_NAMENODE_USER defined"。
ENV HDFS_NAMENODE_USER=root \
    HDFS_DATANODE_USER=root \
    HDFS_SECONDARYNAMENODE_USER=root \
    YARN_RESOURCEMANAGER_USER=root \
    YARN_NODEMANAGER_USER=root \
    MAPRED_HISTORYSERVER_USER=root

# root 免密 ssh（节点间互信）
RUN set -eux; \
    mkdir -p /root/.ssh /var/run/sshd /data/hadoop/tmp /data/hadoop/dfs/name /data/hadoop/dfs/data; \
    chmod 700 /root/.ssh; \
    ssh-keygen -t ed25519 -N "" -f /root/.ssh/id_ed25519 -q; \
    cp /root/.ssh/id_ed25519.pub /root/.ssh/authorized_keys; \
    chmod 600 /root/.ssh/authorized_keys; \
    printf 'Host *\n    StrictHostKeyChecking no\n    UserKnownHostsFile /dev/null\n    LogLevel ERROR\n' > /root/.ssh/config; \
    chmod 600 /root/.ssh/config; \
    printf '\nPermitRootLogin yes\nPubkeyAuthentication yes\nPasswordAuthentication no\n' >> /etc/ssh/sshd_config

EXPOSE 22 9000 9870 8088 7077 8080 8081 4040 10000 16010

CMD ["/usr/sbin/sshd", "-D", "-e"]
