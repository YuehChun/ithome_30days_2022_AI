# *Reference*

- [Machine-Learn-Collection Bidirectional Recurrent Neural Network](https://github.com/aladdinpersson/Machine-Learning-Collection/blob/master/ML/Pytorch/Basics/pytorch_bidirectional_lstm.py)
引用這作者的範例，以我個人的角度記錄及說明
-------------

### 前言
雙向LSTM 跟先前提到的LSTM其實很像，唯一比較要注意的是他需要兩倍的 hidden_state 跟 cell_state 來記錄數值(*為什麼需要兩倍的原因可以上網查查看，這邊就不多說明了)

另外，在Pytorch裡面只需要增加`bidirectional=True` 就可以變成雙向的LSTM，但接下來在 `Forward` 的那邊就需要特別注意了，在Class 那邊會再另外多做說明。


### 首先第一段就是 import 

**Dash 來呼叫** `torch_import`

###  Hyperparameter 的部分

```python
input_size = 28
sequence_length = 28
num_layers = 2
hidden_size = 256
num_classes = 10
learning_rate = 0.001
batch_size = 64
num_epochs = 2
```

然後記得 加入 device
**Dash 來呼叫** `torch_device`

### 載入資料集

**Dash 來呼叫** `torch_MNIST`

### Bi-LSTM

再來就是處理 GRU 的架構，而已下範例為 many to one

```python

class BiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(BiLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers, batch_first=True, bidirectional=True
        )
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(device)

        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])

        return out

```

### 訓練BiLSTM

New 一個 BiLSTM
```python 
model = BiLSTM(input_size, hidden_size, num_layers, num_classes).to(device)
```

Train 一下 BiLSTM

```python
for epoch in range(num_epochs):
    for batch_idx, (data, targets) in enumerate(train_loader):
        # Get data to cuda if possible
        data = data.to(device=device).squeeze(1)
        targets = targets.to(device=device)

        # forward
        scores = model(data)
        loss = criterion(scores, targets)

        # backward
        optimizer.zero_grad()
        loss.backward()

        # gradient descent or adam step
        optimizer.step()
```

### 評估BiLSTM

```python
def check_accuracy(loader, model):
    if loader.dataset.train:
        print("Checking accuracy on training data")
    else:
        print("Checking accuracy on test data")

    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device).squeeze(1)
            y = y.to(device=device)

            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

        print(
            f"Got {num_correct} / {num_samples} with accuracy  \
              {float(num_correct)/float(num_samples)*100:.2f}"
        )

    model.train()


check_accuracy(train_loader, model)
check_accuracy(test_loader, model)
```

這邊跟LSTM 的訓練的階段跟驗證階段所使用的程式碼都一樣

----------
# *Reference*
- [Machine-Learn-Collection Bidirectional Recurrent Neural Network](https://github.com/aladdinpersson/Machine-Learning-Collection/blob/master/ML/Pytorch/Basics/pytorch_bidirectional_lstm.py)
