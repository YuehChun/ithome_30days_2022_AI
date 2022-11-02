# *Reference*

- [Machine-Learn-Collection Recurrent Neural Network](https://github.com/aladdinpersson/Machine-Learning-Collection/blob/master/ML/Pytorch/Basics/pytorch_rnn_gru_lstm.py)

-------------

### 前言

因為沒時間寫新的文章，只好把RNN 拆成兩天的文章來記錄，偷懶一下我承認，我會好好跟媽祖懺悔的

### 首先第一段就是 import 

**Dash 來呼叫** `torch_import`


###  Hyperparameter 的部分

```python

input_dim = 28
hidden_dim = 256
num_layers = 2
output_class = 10
sequence_length = 28
learning_rate = 0.005
batch_size = 64
num_epochs = 3
```

然後記得 加入 device
**Dash 來呼叫** `torch_device`

### 載入資料集

**Dash 來呼叫** `torch_MNIST`

### GRU

再來就是處理 GRU 的架構，而已下範例為 many to one

```python

class RNN_GRU(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_class):
        super(RNN_GRU, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim * sequence_length, output_class)

    def forward(self, x):
        # 設定hidden_state 初始的參數，可以使用 zeros / randn
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(device)

        # Forward propagate GRU
        out, _ = self.gru(x, h0)
        out = out.reshape(out.shape[0], -1)

        # 最後一在一個 fully connect 結束
        out = self.fc(out)
        return out

```

### 訓練GRU

New 一個 GRU
```python 
model = RNN_GRU(input_dim, hidden_dim, num_layers, output_classes).to(device)
```

Train 一下 GRU

```python
for epoch in range(num_epochs):
    for batch_idx, (data, targets) in enumerate(tqdm(train_loader)):

        data = data.to(device=device).squeeze(1)
        targets = targets.to(device=device)

        # forward
        scores = model(data)
        loss = criterion(scores, targets)

        # backward
        optimizer.zero_grad()
        loss.backward()

        # gradient descent 
        optimizer.step()
```

### 評估GRU

```python
def check_accuracy(loader, model):

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


    model.train()
    return num_correct / num_samples

print(f"Accuracy on training set: {check_accuracy(train_loader, model)*100:2f}")
print(f"Accuracy on test set: {check_accuracy(test_loader, model)*100:.2f}")
```

這邊跟LSTM 的訓練的階段跟驗證階段所使用的程式碼都一樣
兩者就差在中間的架構的LSTM多一個 c0 (cell state)

### RNN

**這邊另外再補充一下最基本RNN架構**
其實文章中大部分的程式碼都可以一起使用，其中比較有差異的就是 class 的裡面的架構有一點不同

```Python
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size * sequence_length, num_classes)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)

        out, _ = self.rnn(x, h0)
        out = out.reshape(out.shape[0], -1)

        out = self.fc(out)
        return out
```

然後再訓練之前記得把 **model 換掉 RNN **

```python 
model = RNN(input_dim, hidden_dim, num_layers, output_classes).to(device)
```


----------
# *Reference*
- [Machine-Learn-Collection Recurrent Neural Network](https://github.com/aladdinpersson/Machine-Learning-Collection/blob/master/ML/Pytorch/Basics/pytorch_rnn_gru_lstm.py)