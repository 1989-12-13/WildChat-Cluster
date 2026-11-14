# WildChat 聚类项目的 Hadoop + Spark 集群

复刻课程实验二/实验四（`bigdata/hadoop:3.3.6` + `bigdata/spark:3.5.5-hadoop3.3.6`）
的三节点集群，部署在实验服务器上。

## 为什么自建镜像而不直接导入课程镜像

课程镜像只存在于本机 WSL（`~/lab02-student`、`~/lab03`、`~/lab04` 的 tar 包，
以及已导入的 docker 镜像）。经实测，本机到实验服务器的传输只有约 0.3 MB/s，
8 GB 镜像需数小时，不可行。

因此从课程镜像中**提取了全部配置**（见下），在实验服务器上用 `ubuntu:24.04`
基础镜像重建，保证行为一致。

## 与课程镜像的一致性

| 项 | 课程镜像 | 本仓库 | 是否一致 |
| --- | --- | --- | --- |
| 基础系统 | Ubuntu 24.04 | ubuntu:24.04 | ✅ |
| JDK | openjdk 11.0.32 | openjdk 11.0.33 | ✅ 同大版本 |
| Hadoop | 3.3.6 → `/usr/local/hadoop` | 3.3.6 | ✅ |
| Spark | 3.5.5-bin-hadoop3 → `/usr/local/spark` | 3.5.5-bin-hadoop3 | ✅ |
| `core-site.xml` | `hdfs://master:9000`, tmp `/data/hadoop/tmp` | 同 | ✅ 逐字复制 |
| `hdfs-site.xml` | 副本 2，name/data 在 `/data/hadoop/dfs/` | 同 | ✅ 逐字复制 |
| `yarn-site.xml` | RM=master, `mapreduce_shuffle` | 同 | ✅ 逐字复制 |
| `mapred-site.xml` | framework=yarn | 同 | ✅ 逐字复制 |
| `workers` | worker1, worker2 | 同 | ✅ |
| root 免密 SSH | ed25519，`StrictHostKeyChecking no` | 同 | ✅ |
| 启动方式 | `/usr/sbin/sshd -D -e` | 同 | ✅ |
| 节点拓扑 | master + worker1 + worker2 | 同 | ✅ |

## 与课程镜像的差异（必要调整）

| 项 | 课程默认 | 本项目 | 原因 |
| --- | --- | --- | --- |
| 每节点内存 | 512 MB（VMware 环境） | master 8G / worker 各 12G | 课程配置跑不动 19 GB 向量 |
| 每节点 CPU | 1 核 | master 8 / worker 各 12 | 同上 |
| 端口绑定 | `0.0.0.0` | `127.0.0.1` | 实验服务器是共享机，避免外露端口冲突 |
| 网络/卷命名 | `lab02-student_hadoopnet` 等 external | `wildchat_hadoopnet` 独立创建 | 不依赖课程遗留资源 |
| 数据交换 | 无 | master 挂载 `./shared` | 用于把宿主机上的向量 put 进 HDFS |

## 目录

```
docker/
├── build/                    # 构建上下文
│   ├── Dockerfile
│   ├── hadoop-3.3.6.tar.gz   # 宿主机预下载（镜像内不再联网下载）
│   ├── spark-3.5.5-bin-hadoop3.tgz
│   └── conf/                 # 从课程镜像提取的配置
│       ├── core-site.xml
│       ├── hdfs-site.xml
│       ├── yarn-site.xml
│       ├── mapred-site.xml
│       └── workers
├── compose.yaml              # 三节点编排
├── build-image.sh            # 构建镜像
└── start-cluster.sh          # 启动并等待集群就绪
```

## 使用

```bash
# 1. 构建镜像（首次，约 5-10 分钟）
cd ~/wildchat_bench/docker
./build-image.sh

# 2. 启动集群
./start-cluster.sh

# 3. 验证
docker exec wc-master hdfs dfs -ls /
docker exec wc-master hdfs dfsadmin -report | grep 'Live datanodes'
docker exec wc-master yarn node -list
```

启动后：

| 服务 | 地址 |
| --- | --- |
| HDFS NameNode UI | http://127.0.0.1:9870 |
| YARN ResourceManager UI | http://127.0.0.1:8088 |
| Spark Master UI | http://127.0.0.1:8080 |
| Spark Master | `spark://master:7077` |

## 资源占用

| 容器 | 内存上限 | CPU 上限 |
| --- | --- | --- |
| wc-master | 8 GB | 8 |
| wc-worker1 | 12 GB | 12 |
| wc-worker2 | 12 GB | 12 |
| **合计** | **32 GB** | **32 核** |

实验服务器总内存 251 GB（可用约 147 GB），CPU 192 核（load 约 3.6）。
本集群占可用内存约 22%、CPU 约 17%，不影响其他用户的既有服务。

## 数据入 HDFS

19 GB 向量在宿主机 `~/wildchat_bench/out/vectors.npy`。流程：

```bash
# 宿主机：npy 不能直接被 Spark 读，先切块转 parquet
python3 to_parquet.py    # 在 shared/ 下生成分块 parquet

# 容器内：put 进 HDFS
docker exec wc-master hdfs dfs -mkdir -p /wildchat/ods
docker exec wc-master hdfs dfs -put /shared/vectors_*.parquet /wildchat/ods/
```

`./shared` 在宿主机与 master 容器之间共享（compose 里已挂载）。

## 停止

```bash
docker compose -f compose.yaml stop     # 保留数据卷
docker compose -f compose.yaml down     # 删除容器，保留卷
docker compose -f compose.yaml down -v  # 连数据卷一起删（清空 HDFS）
```

## 本目录的文件来源

| 子目录 | 来源 |
| --- | --- |
| 根目录的 `Dockerfile`、`compose.yaml`、`build-image.sh`、`build_fast.sh`、`fetch_and_build.sh`、`start-cluster.sh`、`conf/`、`hive/` | 本项目在实验服务器上部署集群所用的文件 |
| `course-image/` | 从课程镜像 `bigdata/spark:3.5.5-hadoop3.3.6` 中提取的 Hadoop 与 Spark 原始配置，用于核对一致性 |
| `lab03/`、`lab04/` | 本机 WSL 中课程实验三与实验四的编排与启动脚本，参考用 |
| `inspect/` | 课程镜像里五个容器的 inspect 输出、镜像清单与数据卷清单 |

本机 WSL 的容器（`master`、`worker1`、`worker2`、`hive`、`hive-mysql`）来自课程实验四的
`compose.yaml`，属于课程实验环境，与本项目的集群分开。
