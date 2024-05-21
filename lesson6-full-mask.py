"""
基于BERT做完形填空
1）数据集来自：ChnSentiCorp
2）模型权重使用：bert-base-chinese
"""
# step 1 引入数据库
from datasets import DatasetDict
from transformers import TrainingArguments, Trainer, BertTokenizerFast, BertForMaskedLM, DataCollatorForLanguageModeling, pipeline

model_path = "./model/tiansz/bert-base-chinese"
data_path = "./data/ChnSentiCorp"

# step 2 数据集处理
datasets = DatasetDict.load_from_disk(data_path)
tokenizer = BertTokenizerFast.from_pretrained(model_path)


def process_function(datas):
    tokenized_datas = tokenizer(datas["text"], max_length=256, truncation=True)
    return tokenized_datas


new_datasets = datasets.map(process_function, batched=True, remove_columns=datasets["train"].column_names)


# step 3 加载模型
model = BertForMaskedLM.from_pretrained(model_path)


# step 4 创建TrainingArguments
# 原先train是9600条数据，batch_size=32，因此每个epoch的step=300
train_args = TrainingArguments(output_dir="./checkpoints",      # 输出文件夹
                               per_device_train_batch_size=32,  # 训练时的batch_size
                               num_train_epochs=1,              # 训练轮数
                               logging_steps=30,                # log 打印的频率
                               )


# step 5 创建Trainer
trainer = Trainer(model=model,
                  args=train_args,
                  train_dataset=new_datasets["train"],
                  # 自动MASK关键所在，通过DataCollatorForLanguageModeling实现自动MASK数据
                  data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=True, mlm_probability=0.15),
                  )


# Step 6 模型训练
trainer.train()

# step 7 模型评估
pipe = pipeline("fill-mask", model=model, tokenizer=tokenizer, device=0)
str = datasets["test"][3]["text"]
str = str.replace("方便","[MASK][MASK]")
results = pipe(str)
# results[0][0]["token_str"]
print(results[0][0]["token_str"]+results[1][0]["token_str"])