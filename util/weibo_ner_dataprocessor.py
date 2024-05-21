"""
用于处理weibo_ner的数据，将其保存arrow格式
"""
from datasets import Dataset, DatasetDict

_DESCRIPTION = """\
Tags: PER(人名), LOC(地点名), GPE(行政区名), ORG(机构名)
Label	Tag	Meaning
PER	PER.NAM	名字（张三）
PER.NOM	代称、类别名（穷人）
LOC	LOC.NAM	特指名称（紫玉山庄）
LOC.NOM	泛称（大峡谷、宾馆）
GPE	GPE.NAM	行政区的名称（北京）
ORG	ORG.NAM	特定机构名称（通惠医院）
ORG.NOM	泛指名称、统称（文艺公司）
"""

def generate_examples(data_path):
    sentence_counter = 0
    with open(data_path, encoding="utf-8") as f:
        sentences = []
        current_words = []
        current_labels = []
        for row in f:
            row = row.rstrip()
            row_split = row.split("\t")
            if len(row_split) == 2:
                token, label = row_split
                current_words.append(token)
                loc = labels_list.index(label)
                current_labels.append(loc)
            else:
                if not current_words:
                    continue
                assert len(current_words) == len(current_labels), "word len doesnt match label length"
                sentence = (
                    sentence_counter,
                    {
                        "id": str(sentence_counter),
                        "tokens": current_words,
                        "ner_tags": current_labels,
                    },
                )
                sentence_counter += 1
                current_words = []
                current_labels = []
                sentences.append(sentence)

        # if something remains:
        if current_words:
            sentence = (
                sentence_counter,
                {
                    "id": str(sentence_counter),
                    "tokens": current_words,
                    "ner_tags": current_labels,
                },
            )
            sentences.append(sentence)

        return sentences


labels_list = ["B-GPE.NAM", "B-GPE.NOM", "B-LOC.NAM", "B-LOC.NOM", "B-ORG.NAM", "B-ORG.NOM", "B-PER.NAM", "B-PER.NOM"
    , "I-GPE.NAM", "I-GPE.NOM", "I-LOC.NAM", "I-LOC.NOM", "I-ORG.NAM", "I-ORG.NOM", "I-PER.NAM", "I-PER.NOM", "O", ]
file_path = "../data/weibo_ner_original/"
train_file_name = "train.txt"
validation_file_name = "dev.txt"
test_file_name = "test.txt"
save_path= "../data/weibo_ner"

train = generate_examples(file_path+train_file_name)
train_list = []
for result in train:
    train_list.append(result[1])
train_dataset = Dataset.from_list(train_list)

validation = generate_examples(file_path+validation_file_name)
validation_list = []
for result in validation:
    validation_list.append(result[1])
validation_dataset = Dataset.from_list(validation_list)

test = generate_examples(file_path+test_file_name)
test_list = []
for result in test:
    test_list.append(result[1])
test_list = Dataset.from_list(test_list)

dict = DatasetDict({"train":train_dataset,"validation":validation_dataset,"test":test_list})
print(dict)
# dict.save_to_disk(save_path)
