"""
基于BERT做情感分析
1）数据集来自：weibo_senti_100k（微博公开数据集），这里只是演示，使用2400条数据
2）模型权重使用：bert-base-chinese
"""

# step 1 引入数据库
import evaluate
import pandas as pd
from datasets import load_dataset
from transformers import BertTokenizer, BertForSequenceClassification, TrainingArguments, Trainer, \
    DataCollatorWithPadding, pipeline

model_path = "./model/tiansz/bert-base-chinese"
data_path = "./data/weibo_senti_100k.csv"

# step 2 数据集处理
df = pd.read_csv(data_path)
dataset = load_dataset("csv", data_files=data_path, split='train')
dataset = dataset.filter(lambda x: x["review"] is not None)
datasets = dataset.train_test_split(test_size=0.1)  # 划分数据集
print("train data size:", len(datasets["train"]["label"]))
tokenizer = BertTokenizer.from_pretrained(model_path)


def process_function(datas):
    tokenized_datas = tokenizer(datas["review"], max_length=256, truncation=True)
    tokenized_datas["labels"] = datas["label"]
    return tokenized_datas


new_datasets = datasets.map(process_function, batched=True, remove_columns=datasets["train"].column_names)

# step 3 加载模型
model = BertForSequenceClassification.from_pretrained(model_path, num_labels=2)

# step 4 评估函数：此处的评估函数可以从https://github.com/huggingface/evaluate下载到本地
acc_metric = evaluate.load("./evaluate/metric_accuracy.py")


def evaluate_function(prepredictions):
    predictions, labels = prepredictions
    predictions = predictions.argmax(axis=1)
    acc = acc_metric.compute(predictions=predictions, references=labels)
    return acc


# step 5 创建TrainingArguments
# 2400条数据，其中train和test比例9:1，因此train数据为2160条，batch_size=32，每个epoch的step=68，epoch=3，因此总共step=204，
train_args = TrainingArguments(output_dir="./checkpoints",      # 输出文件夹
                               per_device_train_batch_size=32,  # 训练时的batch_size
                               # gradient_accumulation_steps=2,   # *** 梯度累加 ***
                               gradient_checkpointing=True,     # *** 梯度检查点 ***
                               per_device_eval_batch_size=32,    # 验证时的batch_size
                               num_train_epochs=3,              # 训练轮数
                               logging_steps=20,                # log 打印的频率
                               evaluation_strategy="epoch",     # 评估策略
                               save_strategy="epoch",           # 保存策略
                               save_total_limit=3,              # 最大保存数
                               learning_rate=2e-5,              # 学习率
                               weight_decay=0.01,               # weight_decay
                               metric_for_best_model="accuracy",      # 设定评估指标
                               load_best_model_at_end=True)     # 训练完成后加载最优模型

# step 6 创建Trainer
trainer = Trainer(model=model,
                  args=train_args,
                  train_dataset=new_datasets["train"],
                  eval_dataset=new_datasets["test"],
                  data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
                  compute_metrics=evaluate_function,
                  )

# step 7 训练
trainer.train()

# step 8 模型评估
evaluate_result = trainer.evaluate(new_datasets["test"])
print(evaluate_result)

# step 9：模型预测
id2_label = {0: "消极", 1: "积极"}
model.config.id2label = id2_label
pipe = pipeline("text-classification", model=model, tokenizer=tokenizer)
sen = "不止啦，金沙 碧水 翠苇 飞鸟 游鱼 远山 彩荷七大要素哒"
print(pipe(sen))
