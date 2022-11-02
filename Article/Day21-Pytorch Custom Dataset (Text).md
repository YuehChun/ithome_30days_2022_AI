# *Reference*

- [Machine-Learn-Collection Custom Dataset (Text)](https://github.com/aladdinpersson/Machine-Learning-Collection/blob/master/ML/Pytorch/Basics/custom_dataset_txt/loader_customtext.py)
引用這作者的範例，以我個人的角度記錄及說明
-------------

### 前言
這章節簡單在講就是會使用 Flickr8k 的資料集，裡面有兩個重要的項目，一個是 Images 用來裝所有的圖片的，另一個是 captions.txt 是來說明圖片裡面的敘述。
如果仔細看 captions.txt 裡面的資料，可以發現一張圖片有很多個comment，因此這邊需要在整理一下，變成一張圖片變成一大串文字。
但是在ML領域裡面，文字沒辦法代表任何意思，因此就需要把它轉成數字或者是 vector (~~轉vector 就是另一個領域了~~)，但是這邊簡單就是直接先轉成數子的方式進行，因此會需要自己建立 Vocabular。
接下來我們邊看程式碼邊講，首先起手式就是 `import package`


### import
```Python
import os  # when loading file paths
import pandas as pd  # for lookup in annotation file
import spacy  # for tokenizer
import torch
from torch.nn.utils.rnn import pad_sequence  # pad batch
from torch.utils.data import DataLoader, Dataset
from PIL import Image  # Load img
import torchvision.transforms as transforms
```

### 宣告 Vocabulary
再來這邊先宣告一個 `Vocabulary` ，然後這個還有計頻次，如果頻次太低就不放在這個辭典裡面

```Python
# 下載英文的字典庫
spacy_eng = spacy.load("en_core_web_sm")

class Vocabulary:
    # 這邊是先建立自己的字典，後面才會繼續增加
    def __init__(self, freq_threshold):
        self.itos = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
        self.stoi = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3}
        self.freq_threshold = freq_threshold

    def __len__(self):
        return len(self.itos)

    # 把一文字先做 tokenizer 切成token ，再把token換成小寫
    @staticmethod
    def tokenizer_eng(text):
        return [tok.text.lower() for tok in spacy_eng.tokenizer(text)]

    
    def build_vocabulary(self, sentence_list):
        frequencies = {}
        idx = 4

        # 每個句子
        for sentence in sentence_list:
            # 把句子透過tokenizer 轉換成words
            for word in self.tokenizer_eng(sentence):
                if word not in frequencies:
                    frequencies[word] = 1

                else:
                    frequencies[word] += 1
                # words 要出現夠多次才會被加入到 vocabulary
                if frequencies[word] == self.freq_threshold:
                    self.stoi[word] = idx
                    self.itos[idx] = word
                    idx += 1

    # 這個就是文字轉數值的地方，簡單來說文字先看 stoi 裡面有沒有，如果沒有的話就回傳<UNK>的數值
    def numericalize(self, text):
        tokenized_text = self.tokenizer_eng(text)

        return [
            self.stoi[token] if token in self.stoi else self.stoi["<UNK>"]
            for token in tokenized_text
        ]
```

### Dataset
這邊就是在建立Dataset的部分，就是把圖跟後面敘述的句子做一個對應，而且對應的同時也會把句子之中的文字轉換成數字，因此可以看成圖片跟一串vector

```Python

# 這邊是建立圖片跟文字之間的關係
class FlickrDataset(Dataset):
    def __init__(self, root_dir, captions_file, transform=None, freq_threshold=5):
        self.root_dir = root_dir
        self.df = pd.read_csv(captions_file)
        self.transform = transform

        # 載入圖片跟敘述
        self.imgs = self.df["image"]
        self.captions = self.df["caption"]

        # 然後這邊就是設定 Vocab 的頻率threshold
        self.vocab = Vocabulary(freq_threshold)
        # 然後這邊就是設定把文字的部分丟進去建立字典
        self.vocab.build_vocabulary(self.captions.tolist())

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        caption = self.captions[index]
        img_id = self.imgs[index]
        img = Image.open(os.path.join(self.root_dir, img_id)).convert("RGB")

        if self.transform is not None:
            img = self.transform(img)
        # SOS => start of sentence
        numericalized_caption = [self.vocab.stoi["<SOS>"]]
        # 這邊就是轉換成向量
        numericalized_caption += self.vocab.numericalize(caption)
        # EOS => end of sentence
        numericalized_caption.append(self.vocab.stoi["<EOS>"])

        # 因此這裡就是一個影像跟 一排vector 的輸出（vector 是文字轉出來的)
        return img, torch.tensor(numericalized_caption)
```

### Pad Sequence

因為在做 Batch process 的時候，input 的向量都是要一樣的，`但是句子的長短不一，因此這邊都要特別考慮到句子對應出來的vector 長度要一樣才有辦法丟到batch裡面去學習`

```Python
class MyCollate:
    def __init__(self, pad_idx):
        self.pad_idx = pad_idx

    def __call__(self, batch):
        imgs = [item[0].unsqueeze(0) for item in batch]
        imgs = torch.cat(imgs, dim=0)
        targets = [item[1] for item in batch]
        # 把他填充到等長
        targets = pad_sequence(targets, batch_first=False, padding_value=self.pad_idx)

        return imgs, targets
```

### DataLoader
DataLoader 的重要性我就不多説了，不過在補齊欄位的地方因為要做batch ，所以是在這裡才把shape補成一樣的喔～

```Python
def get_loader(
    root_folder,
    annotation_file,
    transform,
    batch_size=32,
    num_workers=8,
    shuffle=True,
    pin_memory=True,
):
    dataset = FlickrDataset(root_folder, annotation_file, transform=transform)
	# 比較短的句子後面就補上<PAD>
    pad_idx = dataset.dataset.stoi["<PAD>"]

    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        pin_memory=pin_memory,
        collate_fn=MyCollate(pad_idx=pad_idx),
    )

    return loader, dataset

```

### Google Driver
這就不用多說了
```Python
%%time
from google.colab import drive
drive.mount('/content/drive')
```

### 最後
```Python

if __name__ == "__main__":
    transform = transforms.Compose(
        [transforms.Resize((224, 224)), transforms.ToTensor(),]
    )

    loader, dataset = get_loader(
        "/content/drive/MyDrive/Colab Notebooks/ithome/Flickr8k/Images/", 
        "/content/drive/MyDrive/Colab Notebooks/ithome/Flickr8k/captions.txt", 
        transform=transform
    )
    print(len(dataset.vocab))

    for idx, (imgs, captions) in enumerate(loader.head(0)):
        print(imgs.shape)
        print(captions.shape)
```

### 結論
其實作者做得很詳細的說明，尤其在ML領域碰到文字的時候都需要另外的花心思處理，因為可能就是資料沒問題，然後Model 也正確，但是就是 Train不起來，原因就是 batch input shape 不一致的關係所導致的狀況，如果沒有經驗還真的很容易走到這個陷阱



----------
# *Reference*

- [Machine-Learn-Collection Custom Dataset (Text)](https://github.com/aladdinpersson/Machine-Learning-Collection/blob/master/ML/Pytorch/Basics/custom_dataset_txt/loader_customtext.py)
- [Flickr8k dataset Kaggle](https://www.kaggle.com/datasets/adityajn105/flickr8k)
