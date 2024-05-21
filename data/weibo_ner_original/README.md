---
license: other
tags:
- NER
text:
  token-classification:    
    type:
      - ner
    language:
      - zh

---

# weibo命名实体识别数据集

## 数据集概述
weibo数据集是面向社交媒体的中文命名实体识别数据集。

### 数据集简介
本数据集包括训练集（1350）、验证集（269）、测试集（270），实体类型包括地缘政治实体(GPE.NAM)、地名(LOC.NAM)、机构名(ORG.NAM)、人名(PER.NAM)及其对应的代指(以NOM为结尾）。

### 数据集的格式和结构
数据格式采用conll标准，数据分为两列，第一列是输入句中的词划分，第二列是每个词对应的命名实体类型标签。一个具体case的例子如下：

```
人	O
生	O
如	O
戏	O
，	O
导	B-PER.NOM
演	I-PER.NOM
是	O
自	O
己	O
蜡	O
烛	O
```

## 数据集版权信息

Creative Commons Attribution 4.0 International。


## 引用方式
  ```bib
    @inproceedings{peng-dredze-2015-named,
        title = "Named Entity Recognition for {C}hinese Social Media with Jointly Trained Embeddings",
        author = "Peng, Nanyun  and
        Dredze, Mark",
        booktitle = "Proceedings of the 2015 Conference on Empirical Methods in Natural Language Processing",
        month = sep,
        year = "2015",
        address = "Lisbon, Portugal",
        publisher = "Association for Computational Linguistics",
        url = "https://aclanthology.org/D15-1064",
        doi = "10.18653/v1/D15-1064",
        pages = "548--554",
    }

    @inproceedings{peng-dredze-2016-improving,
        title = "Improving Named Entity Recognition for {C}hinese Social Media with Word Segmentation Representation Learning",
        author = "Peng, Nanyun  and
        Dredze, Mark",
        booktitle = "Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers)",
        month = aug,
        year = "2016",
        address = "Berlin, Germany",
        publisher = "Association for Computational Linguistics",
        url = "https://aclanthology.org/P16-2025",
        doi = "10.18653/v1/P16-2025",
        pages = "149--155",
    }
  ```