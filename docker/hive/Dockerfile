# 复刻课程镜像 bigdata/hive:4.0.1-hadoop3.3.6
#
# 依据：从 WSL 课程镜像提取的 Config 与 /usr/local/hive/conf/hive-site.xml
#   base   : ubuntu:24.04
#   java   : openjdk-11
#   hadoop : 3.3.6（仅需客户端与配置，用于读写 HDFS）
#   hive   : 4.0.1 -> /usr/local/hive
#   mysql  : 8.0 驱动（metastore）
#
# 注意：Hive 依赖 Hadoop 的配置目录（HADOOP_CONF_DIR）来访问 HDFS，
#       所以这里也安装了完整的 Hadoop 3.3.6。
FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive

RUN set -eux; \
    MIRROR=http://mirrors.aliyun.com/ubuntu; \
    sed -i "s|http://archive.ubuntu.com/ubuntu|${MIRROR}|g" /etc/apt/sources.list.d/ubuntu.sources; \
    sed -i "s|http://security.ubuntu.com/ubuntu|${MIRROR}|g" /etc/apt/sources.list.d/ubuntu.sources

RUN set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends \
        openjdk-11-jdk-headless \
        openssh-server openssh-client \
        curl ca-certificates procps net-tools vim-tiny less \
        python3 python3-numpy python3-pandas python3-pip \
        python3-setuptools tini mysql-client; \
    rm -rf /var/lib/apt/lists/*

ENV JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64

# Hadoop 客户端（复用宿主已下载的 tar）
COPY hadoop-3.3.6.tar.gz /tmp/hadoop.tgz
RUN set -eux; \
    tar -xzf /tmp/hadoop.tgz -C /opt; \
    mv /opt/hadoop-3.3.6 /usr/local/hadoop; \
    rm -f /tmp/hadoop.tgz

# Hive 4.0.1（tar 由宿主机预下载）
COPY apache-hive-4.0.1-bin.tar.gz /tmp/hive.tgz
RUN set -eux; \
    tar -xzf /tmp/hive.tgz -C /opt; \
    mv /opt/apache-hive-4.0.1-bin /usr/local/hive; \
    rm -f /tmp/hive.tgz; \
    mkdir -p /usr/local/hive/logs

# MySQL JDBC 驱动（Hive metastore 必需）
COPY mysql-connector-j-8.4.0.jar /usr/local/hive/lib/

ENV HADOOP_HOME=/usr/local/hadoop \
    HADOOP_CONF_DIR=/usr/local/hadoop/etc/hadoop \
    HADOOP_COMMON_HOME=/usr/local/hadoop \
    HADOOP_HDFS_HOME=/usr/local/hadoop \
    HADOOP_MAPRED_HOME=/usr/local/hadoop \
    HADOOP_YARN_HOME=/usr/local/hadoop \
    HIVE_HOME=/usr/local/hive \
    HIVE_CONF_DIR=/usr/local/hive/conf \
    PATH=/usr/local/hive/bin:/usr/local/hadoop/bin:/usr/local/hadoop/sbin:/usr/lib/jvm/java-11-openjdk-amd64/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

# 与课程镜像一致的 Hadoop 配置（访问 HDFS）
COPY conf/core-site.xml /usr/local/hadoop/etc/hadoop/core-site.xml
COPY conf/hdfs-site.xml /usr/local/hadoop/etc/hadoop/hdfs-site.xml
COPY conf/hadoop-env.sh /usr/local/hadoop/etc/hadoop/hadoop-env.sh
# mapred-site.xml 与 yarn-site.xml 必需：否则 mapreduce.framework.name 取默认值
# local，Hive 的聚合查询会跑成 job_local* 而非提交到 YARN 集群
COPY conf/mapred-site.xml /usr/local/hadoop/etc/hadoop/mapred-site.xml
COPY conf/yarn-site.xml   /usr/local/hadoop/etc/hadoop/yarn-site.xml

# Hive 配置（照课程镜像 hive-site.xml）
COPY conf/hive-site.xml /usr/local/hive/conf/hive-site.xml
COPY conf/hive-env.sh  /usr/local/hive/conf/hive-env.sh

# Hive 容器不需要 sshd（HDFS 通过 hadoop 客户端访问），
# 保持容器常驻即可；HiveServer2 由 start-hive.sh 在容器内拉起。
CMD ["sleep", "infinity"]
