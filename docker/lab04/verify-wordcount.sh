#!/usr/bin/env bash
set -eu

INPUT=/user/root/input/words.txt

if ! docker exec master hdfs dfs -test -e "$INPUT"; then
  echo "未找到 $INPUT，正在创建实验输入。"
  docker exec -i master bash -c 'cat > /tmp/lab04-words.txt' <<'EOF'
hadoop hdfs yarn
hadoop mapreduce
hdfs hadoop
spark hive hadoop
EOF
  docker exec master hdfs dfs -mkdir -p /user/root/input
  docker exec master hdfs dfs -put -f /tmp/lab04-words.txt "$INPUT"
fi

output=$(
  docker exec -i master /usr/local/spark/bin/spark-shell \
    --master spark://master:7077 \
    --conf spark.ui.enabled=false \
    2>&1 <<'EOF'
val text = sc.textFile("hdfs://master:9000/user/root/input/words.txt")
val counts = text.flatMap(_.split("\\s+")).filter(_.nonEmpty).map(word => (word, 1)).reduceByKey(_ + _)
counts.sortByKey().collect().foreach(println)
:quit
EOF
)

printf '%s\n' "$output"

for expected in \
  '(hadoop,4)' \
  '(hdfs,2)' \
  '(hive,1)' \
  '(mapreduce,1)' \
  '(spark,1)' \
  '(yarn,1)'; do
  if ! printf '%s\n' "$output" | grep -Fq "$expected"; then
    echo "错误：Spark WordCount 缺少预期结果 $expected" >&2
    exit 1
  fi
done

echo
echo "Spark WordCount verified"
