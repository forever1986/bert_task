"""
基于BERT做阅读理解（问答任务）
1）数据集来自：cmrc2018
2）模型权重使用：bert-base-chinese
"""
# step 1 引入数据库
import nltk
import numpy
import collections
from datasets import DatasetDict
from evaluate.cmrc_eval import evaluate_cmrc
from transformers import BertForQuestionAnswering, TrainingArguments, Trainer, DefaultDataCollator, \
     pipeline, BertTokenizerFast

nltk.download("punkt")  # 评估函数中使用的库
model_path = "./model/tiansz/bert-base-chinese"
data_path = "data/cmrc2018"

# step 2 数据集处理
datasets = DatasetDict.load_from_disk(data_path)
tokenizer = BertTokenizerFast.from_pretrained(model_path)


def process_function(datas):
    tokenized_datas = tokenizer(text=datas["question"],
                                text_pair=datas["context"],
                                return_offsets_mapping=True,  # 返回token与input_ids的位置映射
                                return_overflowing_tokens=True,  # 设置将句子切分为多个句子（如果句子超过max_length的话）
                                stride=128,  # 切分的重叠token
                                max_length=512, truncation="only_second", padding="max_length")
    sample_mappings = tokenized_datas.pop("overflow_to_sample_mapping")
    start_positions = []  # 答案在输入的context的起始位置
    end_positions = []  # 答案在输入的context的结束位置
    data_ids = []
    for idx, _ in enumerate(sample_mappings):
        answer = datas["answers"][sample_mappings[idx]]
        answer_start = answer["answer_start"][0]
        answer_end = answer_start + len(answer["text"][0])
        context_start = tokenized_datas.sequence_ids(idx).index(1)
        context_end = tokenized_datas.sequence_ids(idx).index(None, context_start) - 1
        offset = tokenized_datas.get("offset_mapping")[idx]
        # 如果答案没有在context中
        if offset[context_end][1] < answer_start or offset[context_start][0] > answer_end:
            start_pos = 0
            end_pos = 0
        else:
            # 如果答案在context中
            token_index = context_start
            while token_index <= context_end and offset[token_index][0] < answer_start:
                token_index += 1
            start_pos = token_index
            token_index = context_end
            while token_index >= context_start and offset[token_index][1] > answer_end:
                token_index -= 1
            end_pos = token_index
        start_positions.append(start_pos)
        end_positions.append(end_pos)
        data_ids.append(datas["id"][sample_mappings[idx]])
        # 将question部分的offset_mapping设置为None，为了方便在评估时查找context时，快速过滤掉question部分
        tokenized_datas["offset_mapping"][idx] = [
            (o if tokenized_datas.sequence_ids(idx)[k] == 1 else None)
            for k, o in enumerate(tokenized_datas["offset_mapping"][idx])
        ]

    tokenized_datas["data_ids"] = data_ids
    tokenized_datas["start_positions"] = start_positions
    tokenized_datas["end_positions"] = end_positions
    return tokenized_datas


new_datasets = datasets.map(process_function, batched=True, remove_columns=datasets["train"].column_names)


# step 3 加载模型
model = BertForQuestionAnswering.from_pretrained(model_path)


# step 4 评估函数
def evaluate_function(prepredictions):
    start_logits, end_logits = prepredictions[0]
    if start_logits.shape[0] == len(new_datasets["validation"]):
        p, r = get_result(start_logits, end_logits, datasets["validation"], new_datasets["validation"])
    else:
        p, r = get_result(start_logits, end_logits, datasets["test"], new_datasets["test"])
    return evaluate_cmrc(p, r)


def get_result(start_logits, end_logits, datas, features):
    predictions = {}
    references = {}
    # datas 和 feature的映射
    example_to_feature = collections.defaultdict(list)
    for idx, example_id in enumerate(features["data_ids"]):
        example_to_feature[example_id].append(idx)

    # 最优答案候选
    n_best = 20
    # 最大答案长度
    max_answer_length = 30

    for example in datas:
        example_id = example["id"]
        context = example["context"]
        answers = []
        for feature_idx in example_to_feature[example_id]:
            start_logit = start_logits[feature_idx]  # 预测结果开始位置
            end_logit = end_logits[feature_idx]     # 预测结果结束位置
            offset = features[feature_idx]["offset_mapping"]  # 每个词与token的映射
            start_indexes = numpy.argsort(start_logit)[::-1][:n_best].tolist()
            end_indexes = numpy.argsort(end_logit)[::-1][:n_best].tolist()
            for start_index in start_indexes:
                for end_index in end_indexes:
                    if offset[start_index] is None or offset[end_index] is None:
                        continue
                    if end_index < start_index or end_index - start_index + 1 > max_answer_length:
                        continue
                    answers.append({
                        "text": context[offset[start_index][0]: offset[end_index][1]],
                        "score": start_logit[start_index] + end_logit[end_index]
                    })
        if len(answers) > 0:
            best_answer = max(answers, key=lambda x: x["score"])
            predictions[example_id] = best_answer["text"]
        else:
            predictions[example_id] = ""
        references[example_id] = example["answers"]["text"]

    return predictions, references


# step 5 创建TrainingArguments
# 原先train是1002条数据，但是拆分后的数据量是1439，batch_size=32，因此每个epoch的step=45，总step=135
train_args = TrainingArguments(output_dir="./checkpoints",      # 输出文件夹
                               per_device_train_batch_size=32,  # 训练时的batch_size
                               per_device_eval_batch_size=32,    # 验证时的batch_size
                               num_train_epochs=3,              # 训练轮数
                               logging_steps=20,                # log 打印的频率
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
                  data_collator=DefaultDataCollator(),
                  compute_metrics=evaluate_function,
                  )


# Step 7 模型训练
trainer.train()

# step 8 模型评估
evaluate_result = trainer.evaluate(new_datasets["test"])
print(evaluate_result)

# Step 9 模型预测
pipe = pipeline("question-answering", model=model, tokenizer=tokenizer)
result = pipe(question="乍都节公园位于什么地方？", context="乍都节公园位于泰国的首都曼谷的乍都节县，是拍凤裕庭路、威拍哇丽兰室路、甘烹碧路之间的一处公众游园地。")
print(result)
