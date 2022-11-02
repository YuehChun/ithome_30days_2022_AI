# *Reference*

- [Machine-Learn-Collection Image Captioning](https://github.com/aladdinpersson/Machine-Learning-Collection/tree/master/ML/Pytorch/more_advanced/image_captioning)
引用這作者的範例，以我個人的角度記錄及說明
-------------

### 前言
這篇結合之前的內容，如果不懂的話，可以參考前面的【Day21】的部分，而資料集是使用之前提到的Flickr8k，所以這邊前面的部分還是會處理資料的問題，沒辦法直接透過api 把資料直接載進來，因此這篇的主要的分成三大塊
1. 處理資料 (Dataset and Dataloader)
2. 宣告 Model (Encoder and Decoder and combine all)
3. Train and checkpoint
當然還有其他的小的 function 這邊就不多說

那以Flickr8k的來說明，資料集的內容主要的是一張影片會對應到一段說明，而Image Captioning 主要是利用一張image圖片，來產生一段字幕，所以輸入的話是影像(image)，輸出的話是文字(Captioning)，然後在 Dataset 的時候都會把兩邊都轉換成tensor，並在後面的 Dataloader 丟進去model去訓練。

Model 的部分主要分成 Encoder 跟 Decoder 兩塊，在 Encoder 的部分主要是使用 CNN 的架構來取出圖片的資訊，然後會再帶入 Decoder 利用 RNN 的架構來產生文字，而在 Model 的部分會說明怎麼去接的。
另外 CNN 的部分我們這邊直接使用 Inception v3 的架構，然後做 Fine-tune 。就是移除最後面的 fully connect ，然後加入新的沒有權重的 fully connect 去訓練調整權重層即可，而後面的 RNN 的 Model 是這篇自己訓練的。
以上大概是這篇的內容，如果有興趣我們就繼續看下去吧



### import package and google driver
首先就是要 import 重要的 package，然後還有要使用的 pretrained inception
```Python 
import re,io,os,sys
import pandas as pd  # for lookup in annotation file
import spacy  # for tokenizer
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence  # pad batch
from torch.utils.data import DataLoader, Dataset
from PIL import Image  # Load img
import torchvision.transforms as transforms
from torchtext.data.utils import get_tokenizer

from torchtext.vocab import vocab
from tqdm import tqdm
from collections import Counter
import statistics
# import torchvision.models as models 還有 weight
from torchvision.models import inception_v3, Inception_V3_Weights
```

連接Google Drive
```Python
from google.colab import drive
drive.mount('/content/drive')
```

下載語料庫
```Python
!python -m spacy download en_core_web_sm
```

### 處理文字的部分

需要把文字轉換成數值，所以要使用一個 Vocabulary 來做 mapping 把 token 轉成數字或把數字轉成對應的 token 。
另外我們需要設定及建立 Dataset 跟 Dataloader ，而 Dataset 的部分就是在做 captioning 跟 image 之間的對應關係，另外 Dataloader 的部分則是處理每批次載入的資料問題

然後記住處理文字時batch 需要一致的長度，因此對於較短的seqence 會補 `<PAD>` 跟最長的 seqence 對齊，而這些步驟都是在建立 Datatset 的時候會多做一個 tansform 的功夫

```Python
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
    pad_idx = dataset.vocab.stoi["<PAD>"]

    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        pin_memory=pin_memory,
        collate_fn=MyCollate(pad_idx=pad_idx),
    )

    return loader, dataset

def testDataLoader():
    transform = transforms.Compose(
        [transforms.Resize((224, 224)), transforms.ToTensor(),]
    )

    loader, dataset = get_loader(
        "/content/drive/MyDrive/Colab Notebooks/ithome/Flickr8k/Images/", 
        "/content/drive/MyDrive/Colab Notebooks/ithome/Flickr8k/captions.txt", 
        transform=transform
    )

    for idx, (imgs, captions) in enumerate(loader):
        print(type(imgs))
        print(type(captions))
        # print(imgs.shape)
        # print(captions.shape)
```

### Check point

如果跟我一樣沒有硬體的GPU 可以來訓練，但是使用的是 colab 的 GPU 的話，相信常常會遇到使用到一半就說使用時間到期，因此這邊就需要設定checkpoint 把訓練到一半的資料就先存起來，然後利用空閒資源就跑一下，能跑多少算多少這樣
所以這邊會有一個 `print_example` 的 function 就是在看每一次 epoch 訓練後的model 然後處理的成效/結果如何？
另外就是 `save_checkpoint` 跟 `load_checkpoint` 的部分我就不用在多講了。

```Python
checkPointPath = "/content/drive/MyDrive/Colab Notebooks/ithome/checkpoints/image_captioing.pth.tar"
# print_examples 是訓練途中來檢查訓練的結果好不好的
def print_examples(model, device, dataset):
    test_path = "/content/drive/MyDrive/Colab Notebooks/ithome/Flickr8k/test_examples"
    transform = transforms.Compose(
        [
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    model.eval()
    test_img1 = transform(Image.open(f"{test_path}/dog.jpg").convert("RGB")).unsqueeze(0)
    print("Example 1 CORRECT: Dog on a beach by the ocean")
    print(
        "Example 1 OUTPUT: "
        + " ".join(model.caption_image(test_img1.to(device), dataset.vocab))
    )
    test_img2 = transform(
        Image.open(f"{test_path}/child.jpg").convert("RGB")
    ).unsqueeze(0)
    print("Example 2 CORRECT: Child holding red frisbee outdoors")
    print(
        "Example 2 OUTPUT: "
        + " ".join(model.caption_image(test_img2.to(device), dataset.vocab))
    )
    test_img3 = transform(Image.open(f"{test_path}/bus.png").convert("RGB")).unsqueeze(
        0
    )
    print("Example 3 CORRECT: Bus driving by parked cars")
    print(
        "Example 3 OUTPUT: "
        + " ".join(model.caption_image(test_img3.to(device), dataset.vocab))
    )
    test_img4 = transform(
        Image.open(f"{test_path}/boat.png").convert("RGB")
    ).unsqueeze(0)
    print("Example 4 CORRECT: A small boat in the ocean")
    print(
        "Example 4 OUTPUT: "
        + " ".join(model.caption_image(test_img4.to(device), dataset.vocab))
    )
    test_img5 = transform(
        Image.open(f"{test_path}/horse.png").convert("RGB")
    ).unsqueeze(0)
    print("Example 5 CORRECT: A cowboy riding a horse in the desert")
    print(
        "Example 5 OUTPUT: "
        + " ".join(model.caption_image(test_img5.to(device), dataset.vocab))
    )
    model.train()



def save_checkpoint(state, filename="/content/drive/MyDrive/Colab Notebooks/ithome/checkpoints/image_captioing.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)


def load_checkpoint(checkpoint, model, optimizer):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    step = checkpoint["step"]
    return step

```

### Encoder (CNN - Pretrained Inception V3)

這邊就是使用Inception V3 訓練好的架構跟權重，(不要想說自己來訓練權重，沒那麼多資源可以燒)，然後使用別人訓練好的model的時候是為了要做 Transfer Learning (遷移學習)，最重要的後面的 output 那層要依據自己資料的特性去做訓練調整權重，因此這邊要把他換掉

```Python
class EncoderCNN(nn.Module):
    def __init__(self, embed_size, train_CNN=False):
        super(EncoderCNN, self).__init__()
        self.train_CNN = train_CNN
        # 載入 inception v3架構及權重
        self.inception = inception_v3(aux_logits=False,init_weights=Inception_V3_Weights.IMAGENET1K_V1)
        # 把最後一層fc的部分換掉成我們的output 為 embed_size
        self.inception.fc = nn.Linear(self.inception.fc.in_features, embed_size)
        self.relu = nn.ReLU()
        self.times = []
        self.dropout = nn.Dropout(0.5)

    def forward(self, images):
        features = self.inception(images)
        return self.dropout(self.relu(features))
```

### Decoder 
Decoder 就沒有什麼特別好講的，裡面就是一般的seqence 的decoder 可以參考【Day21】的部分

```Python
class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers):
        super(DecoderRNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(0.5)

    def forward(self, features, captions):
        embeddings = self.dropout(self.embed(captions))
	    # 加入 encoder output 的feature 進來
        embeddings = torch.cat((features.unsqueeze(0), embeddings), dim=0)
	    # 然後丟出
        hiddens, _ = self.lstm(embeddings)
        outputs = self.linear(hiddens)
        return outputs
```

再來就是用一個 Encoder 跟 Decoder 串接起來的Model，這邊沒有什麼好講的，就是先把Image 丟進去 Encoder CNN ，並把 image 的 feature 丟到 Decoder 裏面然後跑出來的結果是一個機率分佈，然後挑機率最大的就是 output 的 token 數值，然後要再轉換成 token 之後就回傳整個 caption。

不過在做完整的 caption 的時候，因為使用 RNN 做 Decoder 的時候要把 output 的結果當作下次的input 所以裡面會有一個for loop 是在做 inference 的時候使用的

```Python
class CNNtoRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers):
        super(CNNtoRNN, self).__init__()
        self.encoderCNN = EncoderCNN(embed_size)
        self.decoderRNN = DecoderRNN(embed_size, hidden_size, vocab_size, num_layers)

    def forward(self, images, captions):
        features = self.encoderCNN(images)
        outputs = self.decoderRNN(features, captions)
        return outputs

    def caption_image(self, image, vocabulary, max_length=50):
        result_caption = []

        with torch.no_grad():
          
            x = self.encoderCNN(image).unsqueeze(0)
            states = None

            for _ in range(max_length):
	            # RNN每次的 output 都會把 state 覆蓋
                hiddens, states = self.decoderRNN.lstm(x, states)
                # 然後再丟到 linear 跑及幾率
                output = self.decoderRNN.linear(hiddens.squeeze(0))
                # 找最大的
                predicted = output.argmax(1)
	            # 把結果放置 result_caption
                result_caption.append(predicted.item())
                x = self.decoderRNN.embed(predicted).unsqueeze(0)
				#一直重複上面的過程一直跑到結束token (<EOS>) 出現在停止
                if vocabulary.itos[predicted.item()] == "<EOS>":
                    break

        return [vocabulary.itos[idx] for idx in result_caption]
```


### Train model 

再來就是把前面的 Dataset & DataLoader 當作input 丟進  Model 的串起來，然後還有 optimizer 的部份

```Python
def train():
	# 因為 Inpection V3 的 model input 的影像是299 x 299
    transform = transforms.Compose(
        [
            transforms.Resize((356, 356)),
            transforms.RandomCrop((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    train_loader, dataset = get_loader(
        root_folder="/content/drive/MyDrive/Colab Notebooks/ithome/Flickr8k/Images",
        annotation_file="/content/drive/MyDrive/Colab Notebooks/ithome/Flickr8k/captions.txt",
        transform=transform,
        num_workers=2,
    )
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    save_model = True
    train_CNN = False # 不需要重新Train CNN

    # Hyperparameters
    embed_size = 256
    hidden_size = 256
    vocab_size = len(dataset.vocab)
    num_layers = 1
    learning_rate = 3e-4
    num_epochs = 100

    # for tensorboard
    # writer = SummaryWriter("runs/flickr")
    step = 0

    # initialize model, loss etc
    model = CNNtoRNN(embed_size, hidden_size, vocab_size, num_layers).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=dataset.vocab.stoi["<PAD>"])
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 僅針對CNN的fc層做 finetune
    for name, param in model.encoderCNN.inception.named_parameters():
        if "fc.weight" in name or "fc.bias" in name:
            param.requires_grad = True
        else:
            param.requires_grad = train_CNN

        if torch.cuda.is_available():
          step = load_checkpoint(torch.load(checkPointPath), model, optimizer)
        else:
          step = load_checkpoint(torch.load(checkPointPath, map_location=torch.device('cpu')), model, optimizer)
          
    model.train()

    for epoch in range(num_epochs):
        # Uncomment the line below to see a couple of test cases
        print_examples(model, device, dataset)
        if save_model:
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "step": step,
            }
            save_checkpoint(checkpoint)
        for idx, (imgs, captions) in tqdm(
            enumerate(train_loader), total=len(train_loader), leave=False
        ):
            imgs = imgs.to(device)
            captions = captions.to(device)

            outputs = model(imgs, captions[:-1])
            loss = criterion(
                outputs.reshape(-1, outputs.shape[2]), captions.reshape(-1)
            )
          
			# writer.add_scalar("Training loss", loss.item(), global_step=step)
			step += 1

            optimizer.zero_grad()
            loss.backward(loss)
            optimizer.step()

```



### 結論
這概念上其實蠻簡單的，但是實作上會比較複雜一點，需要懂點CNN跟RNN的變形，因此放在後面的篇幅再來講解。 

因為colab的資源被我跑光了(colab pro+)，所以目前還沒有執行的成果，如果有興趣的話，請follow 我的文章，後續還會繼續train model

----------
# *Reference*
- [Machine-Learn-Collection Image Captioning](https://github.com/aladdinpersson/Machine-Learning-Collection/tree/master/ML/Pytorch/more_advanced/image_captioning)
