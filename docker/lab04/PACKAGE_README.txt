实验四：Spark 分布式计算控制文件

文件说明：
- compose.yaml：复用实验二网络和 HDFS 数据卷，创建 Spark 三节点容器。
- start-spark.sh：启动 HDFS、YARN、Spark Master 和三个 Spark Worker。
- stop-spark.sh：停止实验四服务和容器，不删除 HDFS 数据卷。
- verify-wordcount.sh：使用 Spark RDD 读取 HDFS 文件并验证 WordCount 结果。

请按照实验手册完成离线镜像导入、脚本权限设置和实验操作。
