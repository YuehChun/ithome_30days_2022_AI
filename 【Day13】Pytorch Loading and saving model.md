# *Reference*

- [Machine-Learn-Collection Loading and saving model](https://github.com/aladdinpersson/Machine-Learning-Collection/blob/master/ML/Pytorch/Basics/pytorch_loadsave.py)
引用這作者的範例，以我個人的角度記錄及說明
-------------

### 前言
在訓練的 Model 的過程中，如果訓練時間很長，那就可以設立 `check point` ，把Model 訓練的state先存起來放，以防萬一。

如果突然間有意外，例如這兩天的花東地震突然間斷電(**希望大家平安無事**)，這樣就不會把之前花大量時間訓練的結果又要重新再跑了，因此設立 `check point` 買個保險很重要，

### 首先第一段就是 import 

**Dash 來呼叫** `torch_import`

###  Hyperparameter 的部分

```python
in_channels = 1
num_classes = 10
learning_rate = 0.001
batch_size = 64
num_epochs = 3
```

然後記得 加入 device
**Dash 來呼叫** `torch_device`

### 載入資料集
**Dash 來呼叫** `torch_MNIST`


### CNN
這篇使用CNN來做Demo

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
        self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
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
```

### 使用 Colab 
首先需要一個空間存放的`state_dict` 的檔案，因此可以放在自己的`本機端` 或者跟我一樣使用 `Colab` 的話需要跟`Google Drive` 要權限

```python
from google.colab import drive
drive.mount('/content/drive')
```

### 儲存 check point

這邊建立一個函數，負責把訓練的model state 存到特定的路徑 (**可以自行到自己指定的路徑查看**)

```Python
def save_checkpoint(state, filename="/content/drive/MyDrive/Colab Notebooks/ithome/checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)
```



### 第一階段訓練

因為要示範load model 的差別，所以這邊用**兩階段的方式訓練**，理想上第二階段訓練的loss 會接續第一階段的loss 持續下降。

```Python
for epoch in range(num_epochs):
	# losses 是用來記錄每次的 loss 
    losses = []
    for batch_idx, (data, targets) in enumerate(tqdm(train_loader)):
        data = data.to(device=device)
        targets = targets.to(device=device)

		# forward
        scores = model(data)
        loss = criterion(scores, targets)
        # losses 累計lose
        losses.append(loss.item())

        # backward
        optimizer.zero_grad()
        loss.backward()

        # gradient descent or adam step
        optimizer.step()
    
	# 計算平均的loss
    mean_loss = sum(losses)/len(losses)
    print(f'Loss at epoch {epoch} was {mean_loss:.5f}')

	# 最後一個epoch後存起來
    if epoch == (num_epochs-1):
        checkpoint = {'model' : model.state_dict() , 'optimizer' : optimizer.state_dict()}
        save_checkpoint(checkpoint)

```

結果如下，從第一個 `epoch` 的loss 為 `0.318` ，然後第三個 `epoch` 之後就樣到 `0.071` 。然後最重要的是後面在 `checkpoint` 的時候有要把 `model state` 存起來。 *以上為第一次訓練的結果*
```text
100%|██████████| 938/938 [00:13<00:00, 68.11it/s] 
Loss at epoch 0 was 0.31863
100%|██████████| 938/938 [00:07<00:00, 131.39it/s]
Loss at epoch 1 was 0.09301
100%|██████████| 938/938 [00:07<00:00, 129.00it/s]
Loss at epoch 2 was 0.07140
=> Saving checkpoint
```


### 第二階段訓練

第二階段訓練之前要記得先把第一階段的 `model state` 載入進來

```Python

def load_checkpoint(checkpoint, model, optimizer):
    print("=> Loading checkpoint")
    # model.load_state_dict 的方式把model state 載入
    model.load_state_dict(checkpoint["model"])
    # optimizer 也是一樣的做法

optimizer.load_state_dict(checkpoint["optimizer"])
```

然後我們再訓練一次，這邊來跑 6個 `epoch` ，然後要載入 model state 就是使用 `torch.load` 的方式並且指定特定的路徑


```Python

continue_epochs = 6
# 先讀取先前checkpoint的 model state
old_checkpoint = torch.load("/content/drive/MyDrive/Colab Notebooks/ithome/checkpoint.pth.tar")

#這邊把 model state放上去
load_checkpoint(old_checkpoint , model, optimizer)


for epoch in range(continue_epochs):
    losses = []
    for batch_idx, (data, targets) in enumerate(tqdm(train_loader)):
        data = data.to(device=device)
        targets = targets.to(device=device)


        scores = model(data)
        loss = criterion(scores, targets)
        losses.append(loss.item())

        optimizer.zero_grad()
        loss.backward()

        optimizer.step()
    

    mean_loss = sum(losses)/len(losses)
    print(f'Loss at epoch {epoch} was {mean_loss:.5f}')
```

然後剛剛上一個階段訓練的`最終 loss` 是 `0.071`，然後接著跑了6次 `epoch`  的輸出結果如下
```Text
=> Loading checkpoint
100%|██████████| 938/938 [00:07<00:00, 131.93it/s]
Loss at epoch 0 was 0.05976
100%|██████████| 938/938 [00:07<00:00, 133.65it/s]
Loss at epoch 1 was 0.05218
100%|██████████| 938/938 [00:09<00:00, 96.52it/s] 
Loss at epoch 2 was 0.04764
100%|██████████| 938/938 [00:06<00:00, 134.90it/s]
Loss at epoch 3 was 0.04185
100%|██████████| 938/938 [00:06<00:00, 135.86it/s]
Loss at epoch 4 was 0.03852
100%|██████████| 938/938 [00:06<00:00, 134.44it/s]
Loss at epoch 5 was 0.03582
```

可以看得出來第一個 epoch 就是0.059 並且小於 0.071 ，而最後的 loss 為  0.03 (*這邊數值大家執行可能會不一樣，但有比上一階段的數值還小就好*)

###  結論
這個 `save 和 load model state`  的技巧在訓練複雜的Model 或資料量太大的時候就會使用到，可能每次 epoch 就存起來，或者拆分資料的方式存起來，這對於資料科學家來講可以省去重新再執行的時間浪費

----------
# *Reference*

- [Machine-Learn-Collection Loading and saving model](https://github.com/aladdinpersson/Machine-Learning-Collection/blob/master/ML/Pytorch/Basics/pytorch_loadsave.py)