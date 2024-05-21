"""
基于BERT做单选题
1）数据集来自：c3
2）模型权重使用：bert-base-chinese
"""
# step 1 引入数据库
import numpy
import torch
import evaluate
from typing import Any
from datasets import DatasetDict
from transformers import BertForMultipleChoice, TrainingArguments, Trainer, BertTokenizerFast

model_path = "./model/tiansz/bert-base-chinese"
data_path = "data/c3"


# step 2 数据集处理
datasets = DatasetDict.load_from_disk(data_path)
# test数据集没有答案answer，因此去除，也不做模型评估
datasets.pop("test")
tokenizer = BertTokenizerFast.from_pretrained(model_path)


def process_function(datas):
    context = []
    question_choice = []
    labels = []
    for idx in range(len(datas["context"])):
        ctx = "\n".join(datas["context"][idx])
        question = datas["question"][idx]
        choices = datas["choice"][idx]
        for choice in choices:
            context.append(ctx)
            question_choice.append(question + " " + choice)
        if len(choices) < 4:
            for _ in range(4 - len(choices)):
                context.append(ctx)
                question_choice.append(question + " " + "不知道")
        labels.append(choices.index(datas["answer"][idx]))
    tokenized_datas = tokenizer(context, question_choice, truncation="only_first", max_length=256, padding="max_length")
    # 将原先的4个选项都是自身组成一条句子，token之后是一个2维的向量：4倍的datas数量*max_length，这里需要将其变成3维：datas数量*4*max_length
    tokenized_datas = {key: [data[i: i + 4] for i in range(0, len(data), 4)] for key, data in tokenized_datas.items()}
    tokenized_datas["labels"] = labels
    return tokenized_datas


new_datasets = datasets.map(process_function, batched=True)

# step 3 加载模型
model = BertForMultipleChoice.from_pretrained(model_path)

# step 4 评估函数：此处的评估函数可以从https://github.com/huggingface/evaluate下载到本地
accuracy = evaluate.load("./evaluate/metric_accuracy.py")


def evaluate_function(prepredictions):
    predictions, labels = prepredictions
    predictions = numpy.argmax(predictions, axis=-1)
    return accuracy.compute(predictions=predictions, references=labels)


# step 5 创建TrainingArguments
# train是11869条数据，batch_size=16，因此每个epoch的step=742，总step=2226
train_args = TrainingArguments(output_dir="./checkpoints",      # 输出文件夹
                               per_device_train_batch_size=16,  # 训练时的batch_size
                               per_device_eval_batch_size=16,    # 验证时的batch_size
                               num_train_epochs=3,              # 训练轮数
                               logging_steps=100,                # log 打印的频率
                               evaluation_strategy="epoch",     # 评估策略
                               save_strategy="epoch",           # 保存策略
                               save_total_limit=3,              # 最大保存数
                               load_best_model_at_end=True      # 训练完成后加载最优模型
                               )

# step 6 创建Trainer
trainer = Trainer(model=model,
                  args=train_args,
                  train_dataset=new_datasets["train"],
                  eval_dataset=new_datasets["validation"],
                  compute_metrics=evaluate_function,
                  )


# step 7 训练
trainer.train()


# step 8 模型预测
class MultipleChoicePipeline:

    def __init__(self, model, tokenizer) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.device = model.device

    def preprocess(self, context, quesiton, choices):
        cs, qcs = [], []
        for choice in choices:
            cs.append(context)
            qcs.append(quesiton + " " + choice)
        return tokenizer(cs, qcs, truncation="only_first", max_length=256, return_tensors="pt")

    def predict(self, inputs):
        inputs = {k: v.unsqueeze(0).to(self.device) for k, v in inputs.items()}
        return self.model(**inputs).logits

    def postprocess(self, logits, choices):
        predition = torch.argmax(logits, dim=-1).cpu().item()
        return choices[predition]

    def __call__(self, context, question, choices) -> Any:
        inputs = self.preprocess(context, question, choices)
        logits = self.predict(inputs)
        result = self.postprocess(logits, choices)
        return result


pipe = MultipleChoicePipeline(model, tokenizer)
res = pipe("男：还是古典音乐好听，我就受不了摇滚乐，实在太吵了。女：那是因为你没听过现场，流行乐和摇滚乐才有感觉呢。",
           "男的喜欢什么样的音乐？", ["古典音乐", "摇滚音乐", "流行音乐", "乡村音乐"])
print(res)