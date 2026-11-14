# Hive 环境变量
export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64
export HADOOP_HOME=/usr/local/hadoop
export HIVE_HOME=/usr/local/hive

# Hive 4.x 需要显式指定 Hadoop 版本兼容
export HIVE_AUX_JARS_PATH=

# 以 root 运行时避免权限问题
export HADOOP_USER_NAME=root
