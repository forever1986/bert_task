"""
基于BERT做命名实体识别
1）数据集来自：weibo_ner
2）模型权重使用：bert-base-chinese
"""
# step 1 引入数据库
import numpy
import evaluate
from datasets import DatasetDict
from transformers import BertForTokenClassification, TrainingArguments, Trainer, DataCollatorForTokenClassification, \
     pipeline, BertTokenizerFast


model_path = "./model/tiansz/bert-base-chinese"
data_path = "data/weibo_ner"


# step 2 数据集处理
datasets = DatasetDict.load_from_disk(data_path)
# 保存label真实描述，用于显示正确结果和传入模型初始化告诉模型分类数量
label_list = ['B-GPE.NAM', 'B-GPE.NOM', 'B-LOC.NAM', 'B-LOC.NOM', 'B-ORG.NAM', 'B-ORG.NOM', 'B-PER.NAM', 'B-PER.NOM',
              'I-GPE.NAM', 'I-GPE.NOM', 'I-LOC.NAM', 'I-LOC.NOM', 'I-ORG.NAM', 'I-ORG.NOM', 'I-PER.NAM', 'I-PER.NOM',
              'O']
tokenizer = BertTokenizerFast.from_pretrained(model_path)


# 借助word_ids 实现标签映射
def process_function(datas):
    tokenized_datas = tokenizer(datas["tokens"], max_length=256, truncation=True, is_split_into_words=True)
    labels = []
    for i, label in enumerate(datas["ner_tags"]):
        word_ids = tokenized_datas.word_ids(batch_index=i)
        label_ids = []
        for word_id in word_ids:
            if word_id is None:
                label_ids.append(-100)
            else:
                label_ids.append(label[word_id])
        labels.append(label_ids)
    tokenized_datas["labels"] = labels
    return tokenized_datas


new_datasets = datasets.map(process_function, batched=True)

# step 3 加载模型
model = BertForTokenClassification.from_pretrained(model_path, num_labels=len(label_list))

# step 4 评估函数：此处的评估函数可以从https://github.com/huggingface/evaluate下载到本地
seqeval = evaluate.load("./evaluate/seqeval_metric.py")


def evaluate_function(prepredictions):
    predictions, labels = prepredictions
    predictions = numpy.argmax(predictions, axis=-1)
    # 将id转换为原始的字符串类型的标签
    true_predictions = [
        [label_list[p] for p, l in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    true_labels = [
        [label_list[l] for p, l in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    result = seqeval.compute(predictions=true_predictions, references=true_labels, mode="strict", scheme="IOB2")
    return {
        "f1": result["overall_f1"]
    }


# step 5 创建TrainingArguments
# train是1350条数据，batch_size=32，因此每个epoch的step=43，总step=129
train_args = TrainingArguments(output_dir="./checkpoints",      # 输出文件夹
                               per_device_train_batch_size=32,  # 训练时的batch_size
                               # gradient_checkpointing=True,     # *** 梯度检查点 ***
                               per_device_eval_batch_size=32,    # 验证时的batch_size
                               num_train_epochs=3,              # 训练轮数
                               logging_steps=20,                # log 打印的频率
                               evaluation_strategy="epoch",     # 评估策略
                               save_strategy="epoch",           # 保存策略
                               save_total_limit=3,              # 最大保存数
                               metric_for_best_model="f1",      # 设定评估指标
                               load_best_model_at_end=True      # 训练完成后加载最优模型
                               )

# step 6 创建Trainer
trainer = Trainer(model=model,
                  args=train_args,
                  train_dataset=new_datasets["train"],
                  eval_dataset=new_datasets["validation"],
                  data_collator=DataCollatorForTokenClassification(tokenizer=tokenizer),
                  compute_metrics=evaluate_function,
                  )

# step 7 训练
trainer.train()

# step 8 模型评估
evaluate_result = trainer.evaluate(new_datasets["test"])
print(evaluate_result)

# step 9：模型预测
ner_pipe = pipeline("token-classification", model=model, tokenizer=tokenizer, device=0, aggregation_strategy="simple")
res = ner_pipe("对，输给一个女人，的成绩。失望")
print(res)
