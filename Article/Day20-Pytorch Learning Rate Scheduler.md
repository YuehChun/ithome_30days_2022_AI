# *Reference*
- [Machine-Learn-Collection Learning Rate Scheduler](https://github.com/aladdinpersson/Machine-Learning-Collection/blob/master/ML/Pytorch/Basics/pytorch_lr_ratescheduler.py)
引用這作者的範例，以我個人的角度記錄及說明
-------------
### 前言
這個講完之後，接下來要講點比較進階的項目，所以如果後面更深入探討DNN的應用就可能會更模糊了（或者是說技術坑更多了），有興趣的再繼續往下看吧（笑）

這次來說的是 Learning rate Scheduler ，可能有人會問為什麼會有多一個 Scheduler ，而原本調整 Learning rate 就有一個 optimization 的東西幫忙調整，而這個又是什麼？

詳情可以參考這篇文章：
[Adam和學習率衰減(Learning rate decay)](https://www.cnblogs.com/wuliytTaotao/p/11101652.html)

簡單來講就是加速訓練速度？但是如果一開始設定 Learning rate 就很低的話確實不會加速（但是Learning rate 太低就會造成訓練太久的現象），因此Scheduler 就是一開始設定比較大的 Learning rate，然後加速收斂，而且還不會 overshooting 的小工具。

![[Images/D20-1.png]]
Source: [Adam和學習率衰減(Learning rate decay)](https://www.cnblogs.com/wuliytTaotao/p/11101652.html)

### import 的部分
```Python
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from tqdm import tqdm
```

### CNN model
```Python

class CNN(nn.Module):
    def __init__(self, in_channels=1, num_classes=10):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=in_channels, out_channels=8, kernel_size=3, stride=1, padding=1
        )
        self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.conv2 = nn.Conv2d(
            in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1
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

### 參數的部分
```Python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
in_channels = 1
num_classes = 10
num_epochs = 50

batch_size = 64
learning_rate = 0.1

train_dataset = datasets.MNIST(
    root="dataset/", train=True, transform=transforms.ToTensor(), download=True
)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

model = CNN(in_channels=in_channels, num_classes=num_classes).to(device)


criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
```

### Scheduler
Scheduler 裡面放著就是 `Optimizer` ，然後還有一個 `Patience` 也很重要
```Python
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(

optimizer, factor=0.1, patience=5, verbose=True

)
```


### Train Model
```Python
for epoch in range(1, num_epochs):
    losses = []

    for batch_idx, (data, targets) in enumerate(train_loader):
        data = data.reshape(data.shape[0], -1)
        data = data.to(device=device)
        targets = targets.to(device=device)

        scores = model(data)
        loss = criterion(scores, targets)

        losses.append(loss.item())

        loss.backward()

        # gradient descent or adam step
        # scheduler.step(loss)
        optimizer.step()
        optimizer.zero_grad()

    mean_loss = sum(losses) / len(losses)

    # After each epoch do scheduler.step, note in this scheduler we need to send
    # in loss for that epoch!
    scheduler.step(mean_loss)
    print(f"Cost at epoch {epoch} is {mean_loss}")

```

### 結論
大概的執行方法就是這樣，建議可以把 scheduler 的方式也寫進 Dash 裡面，之後應該很常會使用

----------
# *Reference*
- [Machine-Learn-Collection Learning Rate Scheduler](https://github.com/aladdinpersson/Machine-Learning-Collection/blob/master/ML/Pytorch/Basics/pytorch_lr_ratescheduler.py)
- [Adam和學習率衰減(Learning rate decay)](https://www.cnblogs.com/wuliytTaotao/p/11101652.html)
