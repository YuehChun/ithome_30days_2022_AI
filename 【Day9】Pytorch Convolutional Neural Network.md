# *Reference*

[Machine-Learn-Collection Convolutional Neural Network](https://github.com/aladdinpersson/Machine-Learning-Collection/blob/master/ML/Pytorch/Basics/pytorch_simple_CNN.py)

-------------
今天開始講 Convolutional Neural Network ，雖然大家可能都看到爛了，確實這是很基本的東西，但是小弟我這些東西都是在工作的時候才開始自學接觸的，所以容許小弟我耍廢來紀錄自己的小空間

至於CNN的運作原理這就不多說，網路上文章就很多了

### 首先第一段就是 import 
之前 Dash 那邊有建立這邊就不多說了，如果不知道的話請看 【Day7】 的部分
**Dash 來呼叫** `torch_import`

### 設定一些參數

**Dash 來呼叫** `torch_device`

```python
# 設定超參數
in_channels = 1
num_classes = 10
learning_rate = 0.001
batch_size = 64
num_epochs = 3
```


### 載入資料集
這邊使用的就是經典的 MNIST的資料集，【Day6】已經有說明這部分的細節，所以這邊就不在多說了，然後因為這個範例也是使用 `MNIST` 

**Dash 來呼叫** `torch_MNIST`

###  宣告 CNN Class
這邊就不使用 Dash 來做 `Snippet`  ，因為 `__init__` 這個會被判斷成一個變數，所以會比較麻煩

```python

class CNN(nn.Module):
    def __init__(self, in_channels=1, num_classes=10):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=8,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
        )
        # kernel_size ,stride , padding 這都是 Conv2d 才會出現的東西
		#不知道怎麼來的可以看一下 Conv2d 的運作原理
        
        self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
	    # MaxPool2d 也是會出現 Kernel_size 跟 stride ，所以有這部分的問題的話
        self.conv2 = nn.Conv2d(
            in_channels=8,
            out_channels=16,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
        )
        self.fc1 = nn.Linear(16 * 7 * 7, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc1(x)
        return x
	# Forward  這邊就是把整個神經架構串起來
```


再來就是訓練Model的部分，首先new CNN。
接著在設定  `loss function` 和  `optimizer`，為了要來訓練 Model ，這部分直接用 Dash 呼叫當作練習

```python
# new 一個 CNN
model = CNN(in_channels=in_channels, num_classes=num_classes).to(device)
```

```python
# 訓練model
for epoch in range(num_epochs):
    for batch_idx, (data, targets) in enumerate(tqdm(train_loader)):
        # Get data to cuda if possible
        data = data.to(device=device)
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


### 檢測Model Accuracy
先宣告檢測的方法

```python

# 定義檢測 accuracy function
def check_accuracy(loader, model):
    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device)
            y = y.to(device=device)

            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)


    model.train()
    return num_correct/num_samples

# 檢測
print(f"Accuracy on training set: {check_accuracy(train_loader, model)*100:.2f}")
print(f"Accuracy on test set: {check_accuracy(test_loader, model)*100:.2f}")
```

### 補充一下小知識
- model.train()會啟用 Batch Normalization 和 Dropout。
- 
- model.eval()會停用 Batch Normalization 和 Dropout。

----------
# *Reference*
[Machine-Learn-Collection Feedforward Neural Network](https://github.com/aladdinpersson/Machine-Learning-Collection/blob/master/ML/Pytorch/Basics/pytorch_simple_fullynet.py)