"""
基于BERT做文本相似度
1）数据集来自：Chinese_Text_Similarity
2）模型权重使用：bert-base-chinese
"""

# step 1 引入数据库
import torch
from torch.nn import CosineSimilarity, CosineEmbeddingLoss

import evaluate
import pandas as pd
from typing import Optional
from datasets import Dataset
from transformers import TrainingArguments, Trainer, BertTokenizerFast, BertPreTrainedModel, PretrainedConfig, BertModel

model_path = "./model/tiansz/bert-base-chinese"
data_path = "./data/Chinese_Text_Similarity.txt"

# step 2 数据集处理
df = pd.read_csv(data_path, sep='\s+')
df = df.sample(n=5000)  # 取其中5000条
datasets = Dataset.from_pandas(df)
# 划分训练集和测试集
datasets = datasets.train_test_split(test_size=0.1, shuffle=True, seed=42)
# 划分训练集和验证集
train_datasets = datasets["train"].train_test_split(test_size=0.05, shuffle=True, seed=42)
datasets["train"] = train_datasets["train"]
datasets["validation"] = train_datasets["test"]
tokenizer = BertTokenizerFast.from_pretrained(model_path)


def process_function(datas):
    sentences = []
    labels = []
    for sentence1, sentence2, label in zip(datas["句子1"], datas["句子2"], datas["相似度"]):
        sentences.append(sentence1)
        sentences.append(sentence2)
        labels.append(1 if int(label) == 1 else -1)
    tokenized_datas = tokenizer(sentences, max_length=256, truncation=True, padding="max_length")
    # 这里将2条数据合并为一组，也就是reshape，从（2倍datas数量 * max_length），变成（datas数量 * 2 * max_length）
    tokenized_datas = {k: [v[i: i + 2] for i in range(0, len(v), 2)] for k, v in tokenized_datas.items()}
    tokenized_datas["labels"] = labels
    return tokenized_datas


new_datasets = datasets.map(process_function, batched=True)


# step 3 加载模型
class SimilarityModel(BertPreTrainedModel):

    def __init__(self, config: PretrainedConfig, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.bert = BertModel(config)
        self.post_init()

    def forward(
            self,
            input_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            token_type_ids: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.Tensor] = None,
            head_mask: Optional[torch.Tensor] = None,
            inputs_embeds: Optional[torch.Tensor] = None,
            labels: Optional[torch.Tensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Step1 分别获取sentenceA 和 sentenceB的输入
        senA_input_ids, senB_input_ids = input_ids[:, 0], input_ids[:, 1]
        senA_attention_mask, senB_attention_mask = attention_mask[:, 0], attention_mask[:, 1]
        senA_token_type_ids, senB_token_type_ids = token_type_ids[:, 0], token_type_ids[:, 1]

        # Step2 分别获取sentenceA 和 sentenceB的向量表示
        senA_outputs = self.bert(
            senA_input_ids,
            attention_mask=senA_attention_mask,
            token_type_ids=senA_token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        senA_pooled_output = senA_outputs[1]  # [batch, hidden]

        senB_outputs = self.bert(
            senB_input_ids,
            attention_mask=senB_attention_mask,
            token_type_ids=senB_token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        senB_pooled_output = senB_outputs[1]  # [batch, hidden]

        # step3 计算相似度

        cos = CosineSimilarity()(senA_pooled_output, senB_pooled_output)  # [batch, ]

        # step4 计算loss

        loss = None
        if labels is not None:
            loss_fct = CosineEmbeddingLoss(0.3)
            loss = loss_fct(senA_pooled_output, senB_pooled_output, labels)

        output = (cos,)
        return ((loss,) + output) if loss is not None else output


model = SimilarityModel.from_pretrained(model_path)

# step 4 评估函数：此处的评估函数可以从https://github.com/huggingface/evaluate下载到本地
acc_metric = evaluate.load("./evaluate/metric_accuracy.py")
f1_metric = evaluate.load("./evaluate/metric_f1.py")


def evaluate_function(eval_predict):
    predictions, labels = eval_predict
    predictions = [int(p > 0.7) for p in predictions]
    labels = [int(l > 0) for l in labels]
    acc = acc_metric.compute(predictions=predictions, references=labels)
    f1 = f1_metric.compute(predictions=predictions, references=labels)
    acc.update(f1)
    return acc


# step 5 创建TrainingArguments
# train是4275条数据，batch_size=32，因此每个epoch的step=134，总step=402
train_args = TrainingArguments(output_dir="./checkpoints",      # 输出文件夹
                               per_device_train_batch_size=32,  # 训练时的batch_size
                               per_device_eval_batch_size=32,    # 验证时的batch_size
                               num_train_epochs=3,              # 训练轮数
                               logging_steps=50,                # log 打印的频率
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

# step 8 模型评估
evaluate_result = trainer.evaluate(new_datasets["test"])
print(evaluate_result)


# step 9 模型预测
class SentenceSimilarityPipeline:

    def __init__(self, model, tokenizer) -> None:
        self.model = model.bert
        self.tokenizer = tokenizer
        self.device = model.device

    def preprocess(self, senA, senB):
        return self.tokenizer([senA, senB], max_length=128, truncation=True, return_tensors="pt", padding=True)

    def predict(self, inputs):
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        return self.model(**inputs)[1]  # [2, 768]

    def postprocess(self, logits):
        cos = CosineSimilarity()(logits[None, 0, :], logits[None,1, :]).squeeze().cpu().item()
        return cos

    def __call__(self, senA, senB):
        inputs = self.preprocess(senA, senB)
        logits = self.predict(inputs)
        result = self.postprocess(logits)
        if result >= 0.7:
            return "相似"
        return "不相似"


pipe = SentenceSimilarityPipeline(model, tokenizer)
print(pipe("广东哪里最好玩啊？", "广东最好玩的地方在哪？"))
