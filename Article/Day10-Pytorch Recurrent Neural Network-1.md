# *Reference*

-[Machine-Learn-Collection Recurrent Neural Network](https://github.com/aladdinpersson/Machine-Learning-Collection/blob/master/ML/Pytorch/Basics/pytorch_rnn_gru_lstm.py)

-------------

### 前言

第二次交手RNN 的模型了，上次因為專案需要直接拿 Model 起來改，對模型的架構及原理幾乎是完全不理解，面試的過程中不少面試官會問到LSTM 跟 GRU 的差異及運作原理，我就陣亡了。

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

### 來寫 LSTM 

再來就是處理 LSTM 的架構，而已下範例為 many to one

```python
class RNN_LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_class):
        super(RNN_LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim * sequence_length, output_class)

    def forward(self, x):
        # 設定hidden_state 初始的參數，可以使用 zeros / randn
        # LSTM 需要多一個 cell states 
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(device)

        out, _ = self.lstm(
            x, (h0, c0)
        )  # out: tensor of shape (batch_size, seq_length, hidden_dim)
        out = out.reshape(out.shape[0], -1)
        
		# 所以上面的 nn.Linear input shape才會是 hiedden_dim*sequence_length
        out = self.fc(out)
        return out
```



趁現在還有記憶的時候快點紀錄
hidden_state(h0) : LSTM GRU RNN都會使用到，用來記錄 cell 運算的結果
cell_state(c0): LSTM在儲存memory cell的值，會傳達到下一個cell，但如果forget gate 為0的話，當前的 cell 就不會影響到這個數值

### 訓練LSTM

New 一個 LSTM
```python 
model = RNN_LSTM(input_dim, hidden_dim, num_layers, output_class).to(device)
```

Train 一下 LSTM

```python
for epoch in range(num_epochs):
    for batch_idx, (data, targets) in enumerate(tqdm(train_loader)):

        # 這邊注意原本是 (64,1,28,28)變成(64,28,28)
        data = data.to(device=device).squeeze(1)
        targets = targets.to(device=device)

        # forward
        scores = model(data)
        loss = criterion(scores, targets)

        # backward
        optimizer.zero_grad()
        loss.backward()

        # gradient descent update step/adam step
        optimizer.step()
```

### 評估LSTM

```python
def check_accuracy(loader, model):

    num_correct = 0
    num_samples = 0
    model.eval()
    
    with torch.no_grad():

        for x, y in loader:
            ### 訓練的時候有使用squeeze(1)，這邊也要跟上
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
~~幫自己挖一個坑，之後有時間請好好研究 self-attention 機制~~




----------
# *Reference*
- [Machine-Learn-Collection Recurrent Neural Network](https://github.com/aladdinpersson/Machine-Learning-Collection/blob/master/ML/Pytorch/Basics/pytorch_rnn_gru_lstm.py)
- [Pytorch實作LSTM執行訊號預測](https://zhenglungwu.medium.com/pytorch%E5%AF%A6%E4%BD%9Clstm%E5%9F%B7%E8%A1%8C%E8%A8%8A%E8%99%9F%E9%A0%90%E6%B8%AC-d1d3f17549e7)
- [12屆IT鐵人賽 ## Day 11 / DL x NLP / 語言與 RNN](https://ithelp.ithome.com.tw/articles/10244308)
