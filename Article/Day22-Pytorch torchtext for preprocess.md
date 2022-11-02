# *Reference*

- [Machine-Learn-Collection Torchtext [1]](https://github.com/aladdinpersson/Machine-Learning-Collection/blob/master/ML/Pytorch/more_advanced/torchtext/torchtext_tutorial1.py)
- [Machine-Learn-Collection Torchtext [2]](https://github.com/aladdinpersson/Machine-Learning-Collection/blob/master/ML/Pytorch/more_advanced/torchtext/torchtext_tutorial2.py)
- [Machine-Learn-Collection Torchtext [3]](https://github.com/aladdinpersson/Machine-Learning-Collection/blob/master/ML/Pytorch/more_advanced/torchtext/torchtext_tutorial3.py)
引用這作者的範例，以我個人的角度記錄及說明
-------------

### 前言
本來Day22 是要來弄一下seq2seq的，但是`越看越不對勁，前面的坑洞很深啊` 因此這邊先來補一個要進行 seq2seq 文字處理上要做的事情，包含字典間立即轉換等等的狀況都在這裡先處理，因此這邊會先開始介紹要做文字上的Machine Learning 專案會需要接觸的東西


### 先調降版本號
```Python
!pip install torch==1.8.0+cu111 torchvision==0.9.0+cu111 torchaudio==0.8.0 torchtext==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html

# Reload environment
exit()
```

`然後等他reload一下`

### 下載Spacy  語言包

```Python
!python -m spacy download de_core_news_sm # 德文
!python -m spacy download zh_core_web_sm # 中文
# 英文不用，因為英文是預設下載
```

### Import Packages
```Python
import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.legacy.datasets import Multi30k
from torchtext.legacy.data import Field, TabularDataset, BucketIterator
import numpy as np
import spacy
import random
from torch.utils.tensorboard import SummaryWriter
import pandas as pd
from sklearn.model_selection import train_test_split
```

```Python
# 掛上Google
from google.colab import drive
drive.mount('/content/drive')
```

```Python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```


### Tutorial 1

這是把文字使用Field 的方式，然後

```Python
# python -m spacy download en
spacy_en = spacy.load("en_core_web_sm")


def tokenize(text):
    return [tok.text for tok in spacy_en.tokenizer(text)]


quote = Field(sequential=True, use_vocab=True, tokenize=tokenize, lower=True)
score = Field(sequential=False, use_vocab=False)

# 筆者上面的 quote 是一段句子，score 的部分則是0 跟1 代表正向跟負向
fields = {"quote": ("q", quote), "score": ("s", score)}
```

再把剛剛切的 Fields 放入 TabularDataset 裡面，簡單就可以快速建立一個 Dataset
```Python
train_data, test_data = TabularDataset.splits(
    path="/content/drive/MyDrive/Colab Notebooks/ithome/torchtext_preprocess", train="train.json", test="test.json", format="json", fields=fields
)
```

這裡可以看一下dataset裡面的樣子
```Python
print(train_data[0].__dict__.keys())
print(train_data[0].__dict__.values())
```

建立 `Vocabulary`
```Python
# glove.6B.100d 這個大小有快900M，所以斟酌使用下載
quote.build_vocab(train_data, max_size=10000, min_freq=1, vectors="glove.6B.100d")


# BucketIterator 這就是依據 batch_size做切割
train_iterator, test_iterator = BucketIterator.splits(
    (train_data, test_data), batch_size=2, device=device
)

```

然後這邊可以看一下 `BucketIterator` 切出來的結果如何，就直接 print 出來，然後再丟到 model 去訓練即可
```Python

for batch_idx, batch in enumerate(train_iterator):
        print(batch.q)
```


### Tutorial 2

這邊就是會載入一段文字對話，然後是德文跟英文的，會用德文主要是因為 `Multi30k` 的資料及只有德文跟英文

```Python
# 載入語言包
spacy_eng = spacy.load("en_core_web_sm")
spacy_ger = spacy.load("de_core_news_sm")

# 語言包就是拿來把句子執行 tokenize
def tokenize_eng(text):
    return [tok.text for tok in spacy_eng.tokenizer(text)]


def tokenize_ger(text):
    return [tok.text for tok in spacy_ger.tokenizer(text)]
```

然後一樣建立 Field ，然後把 Multi30k 的資料及套用 fields，所以是Multi30k 的句子套用到 Fields 上面，而 Fields 又有tokenize 把句子切成 token
```Python
# Field 這邊就會使用剛剛的 tokenize 了
english = Field(sequential=True, use_vocab=True, tokenize=tokenize_eng, lower=True)
german = Field(sequential=True, use_vocab=True, tokenize=tokenize_ger, lower=True)

train_data, validation_data, test_data = Multi30k.splits(
    exts=(".de", ".en"), fields=(german, english)
)
```

建立 `Vocabulary` 這邊回顧一下，這個就是建立一個 stoi 與 itos 的用途
```Python

english.build_vocab(train_data, max_size=10000, min_freq=1, vectors="glove.6B.100d")
german.build_vocab(train_data, max_size=10000, min_freq=1, vectors="glove.6B.100d")


# string to integer (stoi)
print(f'Index of the word (the) is: {english.vocab.stoi["the"]}')

# print integer to string (itos)
print(f"Word of the index (1612) is: {english.vocab.itos[1612]}")
print(f"Word of the index (0) is: {english.vocab.itos[0]}")

```

最後這邊一樣利用 `BucketIterator` 依據 batch_size 把 Dataset 切好後就可以用來後續的 Train model
```
```Python
train_iterator, validation_iterator, test_iterator = BucketIterator.splits(
    (train_data, validation_data, test_data), batch_size=64, device=device
)
for batch in train_iterator:
    print(batch)
```

Result
下方的就是結果
但是可以看得出來 .src 與 .trg 因為句子的長度不一樣，所以會有大小不一至的狀況，但是要丟進 RNN 的 model訓練的話`長度都需要一至`，因此這邊通常後續都會補 `<pad>`

```Text

[torchtext.legacy.data.batch.Batch of size 64 from MULTI30K]
    [.src]:[torch.LongTensor of size 31x64]
    [.trg]:[torch.LongTensor of size 28x64]

[torchtext.legacy.data.batch.Batch of size 64 from MULTI30K]
    [.src]:[torch.LongTensor of size 36x64]
    [.trg]:[torch.LongTensor of size 35x64]
Index of the word (the) is: 5
Word of the index (1612) is: binoculars
Word of the index (0) is: <unk>
```

### Tutorial 3
這一個部分跟我們後面的 Seq2seq 的項目有連接性，因此這邊處理完後的資料就是下一章 Seq2seq的 train data ，所以這邊要稍微注意一下

首先載入資料，在下面Reference 有說要去哪裡下載
```Python
tsv_file = open("/content/drive/MyDrive/Colab Notebooks/ithome/torchtext_preprocess/news-commentary-v15.en-zh.tsv", encoding="utf8").read().split("\n")


	# 這邊就是在處理剛剛載入的tsv
my_raw_data = {"English" : [], "Chinese": []}
for line in tsv_file[1:1000]:
    sp_line = line.split('\t')
    my_raw_data['English'].append(sp_line[0])
    my_raw_data['Chinese'].append(sp_line[1])
# my_raw_data

#然後這邊就是建立 Dataframe
df = pd.DataFrame(my_raw_data, columns=["English", "Chinese"])

# create train and test set
train, test = train_test_split(df, test_size=0.1)


# 這邊要注意一下 如果有中文的話to_json 要加入force_ascii=False
# Get train, test data to json and csv format which can be read by torchtext
train.to_json("/content/drive/MyDrive/Colab Notebooks/ithome/torchtext_preprocess/t3_train.json", orient="records", lines=True, force_ascii=False)
test.to_json("/content/drive/MyDrive/Colab Notebooks/ithome/torchtext_preprocess/t3_test.json", orient="records", lines=True, force_ascii=False)

# 這邊是to save csv
# train.to_csv("/content/drive/MyDrive/Colab Notebooks/ithome/torchtext_preprocess/t3_train.csv", index=False)
# test.to_csv("/content/drive/MyDrive/Colab Notebooks/ithome/torchtext_preprocess/t3_test.csv", index=False)
```

```Python
# 然後載入語言包
spacy_eng = spacy.load("en_core_web_sm")
spacy_zh = spacy.load("zh_core_web_sm")

# tokenize
def tokenize_eng(text):
    return [tok.text for tok in spacy_eng.tokenizer(text)]


def tokenize_zh(text):
    return [tok.text for tok in spacy_zh.tokenizer(text)]

# 然後建立 Field
english = Field(sequential=True, use_vocab=True, tokenize=tokenize_eng, lower=True)
chinese = Field(sequential=True, use_vocab=True, tokenize=tokenize_zh, lower=True)

fields = {"English": ("eng", english), "Chinese": ("zh", chinese)}

# 建立 TabularDataset
train_data, test_data = TabularDataset.splits(
    path="/content/drive/MyDrive/Colab Notebooks/ithome/torchtext_preprocess", train="t3_train.json", test="t3_test.json", format="json", fields=fields
)

# BucketIterator splite by batch_size
train_iterator, test_iterator = BucketIterator.splits(
    (train_data, test_data), batch_size=32, device=device
)

for batch in train_iterator:
    print(batch)
```

### 結論
這些都是進行 `Machine  Translate` 會用到的一些前置作業，包含了找文件然後轉換成 `Dataset` 然後建立 `Vocabulary` 還原成文字。`接下來就放到下一章seq2seq`

----------
# *Reference*

- [Machine-Learn-Collection Torchtext [1]](https://github.com/aladdinpersson/Machine-Learning-Collection/blob/master/ML/Pytorch/more_advanced/torchtext/torchtext_tutorial1.py)
- [Machine-Learn-Collection Torchtext [2]](https://github.com/aladdinpersson/Machine-Learning-Collection/blob/master/ML/Pytorch/more_advanced/torchtext/torchtext_tutorial2.py)
- [Machine-Learn-Collection Torchtext [3]](https://github.com/aladdinpersson/Machine-Learning-Collection/blob/master/ML/Pytorch/more_advanced/torchtext/torchtext_tutorial3.py)
- [Spacy load language](https://spacy.io/usage/models)
- [WMT20](https://statmt.org/wmt20/translation-task.html)
- [WMT20 裡面的 news commentary](https://data.statmt.org/news-commentary/v15/)
- 