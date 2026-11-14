# 照课程镜像 /usr/local/spark/conf/spark-env.sh 改写
# 仅调整 Spark 可用资源（课程为 1 核 / 512m，跑不动 19GB 数据）
export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64
export HADOOP_CONF_DIR=/usr/local/hadoop/etc/hadoop
export SPARK_MASTER_HOST=master
export SPARK_MASTER_PORT=7077
export SPARK_MASTER_WEBUI_PORT=8080
export SPARK_WORKER_CORES=10
export SPARK_WORKER_MEMORY=10g
export SPARK_DAEMON_MEMORY=1g
