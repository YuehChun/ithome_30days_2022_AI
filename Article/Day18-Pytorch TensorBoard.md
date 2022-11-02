# *Reference*

- [Machine-Learn-Collection Pytorch TensorBoard Example](https://github.com/aladdinpersson/Machine-Learning-Collection/blob/master/ML/Pytorch/Basics/pytorch_tensorboard_.py)
引用這作者的範例，以我個人的角度記錄及說明
-------------
### 前言
寫鐵人賽也過半了，這段期間真的是趕稿趕到壓力山大，不過自己學習了很多資訊，希望再看到這篇文章的鐵人賽友們在加油，再撐個幾天就可以玩賽了

### TensorBoard 
是一個用來檢測model 哪裡出問題時可以使用的工具，但是這需要一點 model debug 的知識，所以我這章節可能說明的不是很清楚，如果要糾正的話歡迎到留言謝謝

這次的範例使用的是 CNN 來做來做

### Import & CNN
這邊先放import package  跟之前宣告的CNN


```Python
import torch
import torchvision
import torch.nn as nn functions
import torch.optim as optim 
import torch.nn.functional as F 
import torchvision.datasets as datasets 
import torchvision.transforms as transforms 
from torch.utils.data import (
    DataLoader,
) 

# import SummaryWriter
from torch.utils.tensorboard import SummaryWriter 
```

下面是CNN
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

### Hyperparameter & Setting 
TensorBoard 另外的就是可以視覺化一些訓練Model 的參數及對應的結過，因此這邊特別設定 Hyperparameter 讓 TensorBoard 可以看出一些差異
(**TensorBoard 針對Model tuning 是很有幫助的**)

```Python

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

in_channels = 1
num_classes = 10
num_epochs = 1

# Load Data
train_dataset = datasets.MNIST(
    root="dataset/", train=True, transform=transforms.ToTensor(), download=True
)

# 這邊用兩個 Hyperparameter 分別為 batch_size 和 learning_rate ，為了在 TensorBoard 上面看一些差異

batch_sizes = [16, 64, 128 ,256]
learning_rates = [0.1, 0.01, 0.001, 0.001]
classes = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]

```

### Colab 上面叫 TensorBoard

```Python
# 外部載入 Tensorboard
%load_ext tensorboard
# 打開 Tensorboard 的版面，然後需要執行下面才會有
%tensorboard --logdir=runs
```

此時的因為還沒執行訓練，因此訓練時的資料尚未進來，因此執行這段的時候回出現下面這個現象，但是能代表是的 TensorBoard 可以正常的運作

![[Images/D18-1.png]]

### Train Model 產出Logs
然後接下來就是執行 TrainModel 然後跑出一些資料出來就可以了

```Python
# Model 要包在超參數的項目下面
for batch_size in batch_sizes:
    for learning_rate in learning_rates:
        step = 0
        # Initialize network
        model = CNN(in_channels=in_channels, num_classes=num_classes)
        model.to(device)
        model.train()
        criterion = nn.CrossEntropyLoss()
        train_loader = DataLoader(
            dataset=train_dataset, batch_size=batch_size, shuffle=True
        )
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.0)
        # 然後指定的結果要給一個名稱這樣方便判斷，然後寫道 runs 裡面
        writer = SummaryWriter(
            f"runs/MNIST/MiniBatchSize  {batch_size} LR {learning_rate}"
        )

        # Visualize model in TensorBoard
        images, _ = next(iter(train_loader))
        writer.add_graph(model, images.to(device))
        writer.close()

        for epoch in range(num_epochs):
            losses = []
            accuracies = []

            for batch_idx, (data, targets) in enumerate(train_loader):
                # Get data to cuda if possible
                data = data.to(device=device)
                targets = targets.to(device=device)

                # forward
                scores = model(data)
                loss = criterion(scores, targets)
                losses.append(loss.item())

                # backward
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                
                features = data.reshape(data.shape[0], -1)
                img_grid = torchvision.utils.make_grid(data)
                _, predictions = scores.max(1)
                num_correct = (predictions == targets).sum()
                # 計算訓練的準確度
                running_train_acc = float(num_correct) / float(data.shape[0])
                accuracies.append(running_train_acc)

                # Plot things to tensorboard
                class_labels = [classes[label] for label in predictions]
                writer.add_image("mnist_images", img_grid)
                writer.add_histogram("fc1", model.fc1.weight)
                writer.add_scalar("Training loss", loss, global_step=step)
                writer.add_scalar(
                    "Training Accuracy", running_train_acc, global_step=step
                )

                step += 1

            writer.add_hparams(
                {"lr": learning_rate, "bsize": batch_size},
                {
                    "accuracy": sum(accuracies) / len(accuracies),
                    "loss": sum(losses) / len(losses),
                },
            )
```

### 後續

至於 TensorBoard  的內容這邊就不細講，主要是協助在訓練Model的時候可以檢測哪裡出了問題，或者哪裡有

----------
# *Reference*

- [Machine-Learn-Collection Pytorch TensorBoard Example](https://github.com/aladdinpersson/Machine-Learning-Collection/blob/master/ML/Pytorch/Basics/pytorch_tensorboard_.py)