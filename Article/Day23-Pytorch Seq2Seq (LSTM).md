# *Reference*

- [Machine-Learn-Collection Sequence to Sequence (LSTM)](https://github.com/aladdinpersson/Machine-Learning-Collection/blob/master/ML/Pytorch/more_advanced/Seq2Seq/seq2seq.py)
引用這作者的範例，以我個人的角度記錄及說明
-------------

### 前言
今天要來寫 Seq2seq 的文章了，而這個 Seq2seq 的文章是一系列的文章，下一章節會在增加使用 self-aatention 的機制，然後還會使用 Transformer ，不過這邊不會講太多細節，如果想要知道細節的網路上的資源很多，這邊就來做程式碼的觀賞


### 先處理環境
因為作者他是兩年前就在寫了，所以版本上很多都需要調整，因此這邊要做降版
```Python
!pip install torch==1.8.0+cu111 torchvision==0.9.0+cu111 torchaudio==0.8.0 torchtext==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html

!pip install spacy

exit() # restart kernal
```

再來 import 的部分，因為後面版本整個 `torchtext.legacy.data` 都被拿掉了，所以前面才會做一個降版的步驟

```Python

import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.legacy.data import Field, TabularDataset, BucketIterator
import numpy as np
import spacy
import random
from torch.utils.tensorboard import SummaryWriter  # to print to tensorboard
# from utils import translate_sentence, bleu, save_checkpoint, load_checkpoint
from torchtext.data.metrics import bleu_score
import sys
import os
```

檢查版本一下
```Python 
import torch, torchtext
print(torch.__version__) # 1.8.0 + cu111
print(torchtext.__version__) # 0.9.0
```

使用Google Driver
```Python
from google.colab import drive
drive.mount('/content/drive')
```

這邊要下載這兩個spacy的vocabulary
```Python
!python -m spacy download zh_core_web_sm
!python -m spacy download en_core_web_sm
```

然後這邊就是先處理斷詞，同時把資料切成 `train_data` 和 `test_data`

```Python
spacy_zh = spacy.load("zh_core_web_sm")
spacy_eng = spacy.load("en_core_web_sm")

def tokenize_eng(text):
    return [tok.text for tok in spacy_eng.tokenizer(text)]


def tokenize_zh(text):
    return [tok.text for tok in spacy_zh.tokenizer(text)]


english = Field(sequential=True, use_vocab=True, tokenize=tokenize_eng, lower=True)
chinese = Field(sequential=True, use_vocab=True, tokenize=tokenize_zh, lower=True)

fields = {"English": ("eng", english), "Chinese": ("zh", chinese)}

train_data, test_data = TabularDataset.splits(
    path="/content/drive/MyDrive/Colab Notebooks/ithome/torchtext_preprocess", train="t3_train.json", test="t3_test.json", format="json", fields=fields
)

```

然後建立自己的 `vocabulary`  ，轉換用的
```Python
english.build_vocab(train_data, max_size=50000, min_freq=20, vectors="glove.6B.100d")
chinese.build_vocab(train_data, max_size=50000, min_freq=50, vectors="glove.6B.100d")
print(len(english.vocab))
print(len(chinese.vocab))
```

### 一些訓練Machine Translation 用的小功能
把句子直接做翻譯的 Function ，我就不另外寫了
```Python
def translate_sentence(model, sentence, chinese, english, device, max_length=25):
    # 先載入 tokensizer
    spacy_zh = spacy.load("zh_core_web_sm")
    # Create tokens using spacy and everything in lower case (which is what our vocab is)
    if type(sentence) == str:
        tokens = [token.text.lower() for token in spacy_zh(sentence)]
    else:
        tokens = [token.lower() for token in sentence]

    # 加入開始符號跟結束符號，代表一個句子
    tokens.insert(0, chinese.init_token)
    tokens.append(chinese.eos_token)

    # 然後把文字轉自 vactor
    text_to_indices = [chinese.vocab.stoi[token] for token in tokens]
    # 再把 vactor list 轉換成 Tensor
    sentence_tensor = torch.LongTensor(text_to_indices).unsqueeze(1).to(device)
    # 取消梯度修正
    with torch.no_grad():
        hidden, cell = model.encoder(sentence_tensor)

    # 先宣告 outputs ，然後裡面放一個開符號
    outputs = [english.vocab.stoi["<sos>"]]

    for _ in range(max_length):
        previous_word = torch.LongTensor([outputs[-1]]).to(device)

        # seq2seq 的decoder 會把上一個 hidden_state 跟 cell 當作input 這是觀念
        # 然後output機率最大的
        with torch.no_grad():
            output, hidden, cell = model.decoder(previous_word, hidden, cell)
            best_guess = output.argmax(1).item()

        # 機率最大的數值再把它放進output
        outputs.append(best_guess)

        # 如果是結束字元 eos 的話就中斷不然會一直預測下去
        if output.argmax(1).item() == english.vocab.stoi["<eos>"]:
            break

    # 再把 vactor 轉成文字， itos = integer to string
    translated_sentence = [english.vocab.itos[idx] for idx in outputs]

    # remove start token
    return translated_sentence[1:]
```

然後再來是 bleu socre，這是常常被使用在 machine translation 上的計算方式，是用來計算翻譯品質的方法
```Python

def bleu(data, model, chinese, english, device):
    targets = []
    outputs = []

    for example in data:
        src = vars(example)["zh"]
        trg = vars(example)["eng"]

        prediction = translate_sentence(model, src, chinese, english, device)
        prediction = prediction[:-1]  # remove <eos> token
		# 
        targets.append([trg])
        outputs.append(prediction)
	# bleu score 計算方法
    return bleu_score(outputs, targets)

```

### 建置 Checkpoint 

因為在訓練的過程中啊，像我這種比較客家的工程師就不會直接買Colab pro+ ，因為真的工作薪水就快養不活自己了，還要花錢享受 Colab 的服務，因此為了解決這個方法，我就會另外寫 checkpoint ，而我們前面就有提到怎麼寫，我就不多說了
還有其用途很簡單，因為我們這次訓練的是  machine translation ，所以會訓練很久，大約一小時才一個epoch，因此都要把訓練好的結果存起來，利用時間換取金錢ＸＤ

```Python

def save_checkpoint(state, filename="/content/drive/MyDrive/Colab Notebooks/ithome/seq2seq_cp.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)


def load_checkpoint(checkpoint, model, optimizer):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
```

### Encoder 
Encoder 的部分主要是要訓練 embedding 那一層
```Python
class Encoder(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, num_layers, p):
        super(Encoder, self).__init__()
        self.dropout = nn.Dropout(p)
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.embedding = nn.Embedding(input_size, embedding_size)
        self.rnn = nn.LSTM(embedding_size, hidden_size, num_layers, dropout=p)

    def forward(self, x):
        # x shape: (seq_length, N) where N is batch size

        embedding = self.dropout(self.embedding(x))
        # embedding shape: (seq_length, N, embedding_size)

        outputs, (hidden, cell) = self.rnn(embedding)
        # outputs shape: (seq_length, N, hidden_size)

        return hidden, cell
```

### Decoder
從encoder 的output 接起來後，每一次的input 是上個decoder 的output(hidden , cell)，這邊要注意的是shape的大小，也是在訓練embedding
```Python
class Decoder(nn.Module):
    def __init__(
        self, input_size, embedding_size, hidden_size, output_size, num_layers, p
    ):
        super(Decoder, self).__init__()
        self.dropout = nn.Dropout(p)
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.embedding = nn.Embedding(input_size, embedding_size)
        self.rnn = nn.LSTM(embedding_size, hidden_size, num_layers, dropout=p)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden, cell):
        # x shape: (N) where N is for batch size, we want it to be (1, N), seq_length
        # is 1 here because we are sending in a single word and not a sentence
        x = x.unsqueeze(0)

        embedding = self.dropout(self.embedding(x))
        # embedding shape: (1, N, embedding_size)

        outputs, (hidden, cell) = self.rnn(embedding, (hidden, cell))
        # outputs shape: (1, N, hidden_size)

        predictions = self.fc(outputs)

        # predictions shape: (1, N, length_target_vocabulary) to send it to
        # loss function we want it to be (N, length_target_vocabulary) so we're
        # just gonna remove the first dim
        predictions = predictions.squeeze(0)

        return predictions, hidden, cell
```

### seq2seq

注意一下 encoder 出來的只有 hidden ,cell ，然後decoder 第一個是 sos 開始解
另外 `teacher_force_ratio` 這比較特別我還不太懂他的意思，看意思有點像是用50%的機率用decoder 預測的數值，或者直接用正解的數值，因為怕一直錯下去，大概是這樣吧ＸＤ
主要還是為了讓訓練加速
```Python
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, source, target, teacher_force_ratio=0.5):
        batch_size = source.shape[1]
        # output 英文就是英文句子的長度（會加pad)
        target_len = target.shape[0]
        target_vocab_size = len(english.vocab)

        outputs = torch.zeros(target_len, batch_size, target_vocab_size).to(device)

        hidden, cell = self.encoder(source)

        # Grab the first input to the Decoder which will be <SOS> token
        x = target[0]


        for t in range(1, target_len):
            # Use previous hidden, cell as context from encoder at start
            output, hidden, cell = self.decoder(x, hidden, cell)

            # Store next output prediction
            outputs[t] = output

            # Get the best word the Decoder predicted (index in the vocabulary)
            best_guess = output.argmax(1)

            # With probability of teacher_force_ratio we take the actual next word
            # otherwise we take the word that the Decoder predicted it to be.
            # Teacher Forcing is used so that the model gets used to seeing
            # similar inputs at training and testing time, if teacher forcing is 1
            # then inputs at test time might be completely different than what the
            # network is used to. This was a long comment.
            x = target[t] if random.random() < teacher_force_ratio else best_guess

        return outputs
```


### Parameter
```Python
checkPointPath = "/content/drive/MyDrive/Colab Notebooks/ithome/seq2seq_cp.pth.tar"


num_epochs = 100
learning_rate = 0.001
batch_size = 64

# Model hyperparameters
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_size_encoder = len(chinese.vocab)
input_size_decoder = len(english.vocab)
output_size = len(english.vocab)
encoder_embedding_size = 300
decoder_embedding_size = 300
hidden_size = 1024 # Needs to be the same for both RNN's
num_layers = 2
enc_dropout = 0.5
dec_dropout = 0.
```



### tensorboard 跟 spliter
```Python

# Tensorboard to get nice loss plot
writer = SummaryWriter(f"/content/drive/MyDrive/Colab Notebooks/ithome/runs/loss_plot")
step = 0

train_iterator, test_iterator = BucketIterator.splits(
    (train_data, test_data),
    batch_size=batch_size,
    sort_within_batch=True,
    sort_key = lambda x: len(x.zh),
    device=device,
)
```

### Encoder & Decoder 
這邊先把 encoder 跟 decoder 分開宣告
然後最後再一起丟進 seq2seq的 model
```Python
encoder_net = Encoder(input_size_encoder, encoder_embedding_size, hidden_size, num_layers, enc_dropout).to(device)

decoder_net = Decoder(
    input_size_decoder,
    decoder_embedding_size,
    hidden_size,
    output_size,
    num_layers,
    dec_dropout,
).to(device)


```

seq2seq 的model 就這樣被宣告，把剛剛宣告的 encoder 跟 decoder 放進來seq2seq裡面
```Python

model = Seq2Seq(encoder_net, decoder_net).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

pad_idx = english.vocab.stoi["<pad>"]

# CrossEntropy來計算誤差
criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

# checkpoint很重要
if os.path.isfile(checkPointPath):
    load_checkpoint(torch.load(checkPointPath ), model, optimizer)
```

### Train
這訓練中每次的epoch 都會看一下翻譯的結果，看有沒有越翻越好的狀況，所以會先丟進去翻譯，然後再來訓練
```Python
sentence = "你想不想要来我家看猫?或者一起看Netflix?"

for epoch in range(num_epochs):
    print(f"[Epoch {epoch} / {num_epochs}]")

    checkpoint = {"state_dict": model.state_dict(), "optimizer": optimizer.state_dict()}
    save_checkpoint(checkpoint)

    model.eval()

    translated_sentence = translate_sentence(
        model, sentence, chinese, english, device, max_length=50
    )

    print(f"Translated example sentence: \n {translated_sentence}")

    model.train()

    for batch_idx, batch in enumerate(train_iterator):
        # Get input and targets and get to cuda
        inp_data = batch.zh.to(device)
        target = batch.eng.to(device)

        # Forward prop
        output = model(inp_data, target)
        # Output is of shape (trg_len, batch_size, output_dim) but Cross Entropy Loss
        # doesn't take input in that form. For example if we have MNIST we want to have
        # output to be: (N, 10) and targets just (N). Here we can view it in a similar
        # way that we have output_words * batch_size that we want to send in into
        # our cost function, so we need to do some reshapin. While we're at it
        # Let's also remove the start token while we're at it
        output = output[1:].reshape(-1, output.shape[2])
        target = target[1:].reshape(-1)

        optimizer.zero_grad()
        loss = criterion(output, target)

        # Back prop
        loss.backward()

        # Clip to avoid exploding gradient issues, makes sure grads are
        # within a healthy range
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)

        # Gradient descent step
        optimizer.step()

        # Plot to tensorboard
        writer.add_scalar("Training loss", loss, global_step=step)
        step += 1


score = bleu(test_data[1:100], model, chinese, english, device)
print(f"Bleu score {score*100:.2f}")
```

### 結論

這個真的需要train很久～～～～～～～～真的他Ｘ的很久～～～～

但是可以看出一點點成果，後面有出現一些bug 我之後再修，不過到最後會用 self attention 取代 RNN

----------
# *Reference*

- [Machine-Learn-Collection Sequence to Sequence (LSTM)](https://github.com/aladdinpersson/Machine-Learning-Collection/blob/master/ML/Pytorch/more_advanced/Seq2Seq/seq2seq.py)
- [Spacy load language](https://spacy.io/usage/models)