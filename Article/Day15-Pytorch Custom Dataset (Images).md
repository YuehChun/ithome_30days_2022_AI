# *Reference*

- [Machine-Learn-Collection Custom Dataset (Images)](https://github.com/aladdinpersson/Machine-Learning-Collection/tree/master/ML/Pytorch/Basics/custom_dataset)
引用這作者的範例，以我個人的角度記錄及說明
-------------

### 前言
今天要來講 `Custom Dataset`， 可能會讓人很奇怪為什麼這個東西要特別拉出來講？
主要是因為 `Dataset` 可以直接丟到 `Dataloader`，而 `Dataloader` 再丟給 Model 去執行訓練的時候，那個 batch_size 就會自動幫你切，所以只要 Dataset設定好就好

然後建立自己的 dataset 的話，這邊示範的是使用影像圖片 跟 csv 來組dataset，因此事前要先準備
1. 很多照片的datase(**這邊範例需要reshape**)
2. 一個紀錄照片類別的csv

### 首先第一段就是 import 

**Dash 來呼叫** `torch_import`
**Dash 來呼叫** `torch_device`

### 載入CSV檔案

```CSV
Animal,Label
cat.0.jpg,0
cat.1.jpg,0
cat.2.jpg,0
cat.3.jpg,0
cat.4.jpg,0
cat.5.jpg,0
cat.6.jpg,0
cat.7.jpg,0
dog.0.jpg,1
dog.1.jpg,1
```

### Google Drive
因為使用 **Colab** 的關係，所以任何的 dataset都丟到Google Drive ，所以要**使用的時候都記得要先要存取權**
```Python
from google.colab import drive
drive.mount('/content/drive')
```

### Customer Dataset
`Customer Dataset` 這需要宣告 `__init__` 跟 `__len__` 還有 `__getitem__` ，其中 `__gettime__` 是會影響資料集取出來的項目，因此都在 gettime 上面定義清楚

```Python
class CatsAndDogsDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        img_path = os.path.join(self.root_dir, self.annotations.iloc[index, 0])
        image = io.imread(img_path)
        y_label = torch.tensor(int(self.annotations.iloc[index, 1]))

        if self.transform:
            image = self.transform(image)

        return (image, y_label)
```

### Dataset and DataLoader
這邊call 剛剛宣告的 `dataset` = **CatsAndDogsDataset** ，然後 transforms 的格式設定為 Tensor
```Python

dataset = CatsAndDogsDataset(
    csv_file="/content/drive/MyDrive/Colab Notebooks/ithome/cats_dogs.csv",
    root_dir="/content/drive/MyDrive/Colab Notebooks/ithome/cats_dogs",
    transform=transforms.ToTensor(),
)
```



```Python
# Train 跟 test 的大小
test_size = int(len(dataset)*0.2)
train_size = len(dataset)- int(len(dataset)*0.2)

# 切割資料集
train_dataset, test_dataset = 
torch.utils.data.random_split(dataset, [train_size, test_size])

# 套上dataloader
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)
```

### GoogleNet
這邊直接用先前 trained  `GoogleNet架構` 直接使用

```Python
model = torchvision.models.googlenet(pretrained=True)
model.to(device)


criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

```

### Train
訓練model 其實就跟之前的訓練方式一樣

```Python
for epoch in range(num_epochs):
    losses = []

    for batch_idx, (data, targets) in enumerate(train_loader):
        data = data.to(device=device)
        targets = targets.to(device=device)

        scores = model(data)
        loss = criterion(scores, targets)

        losses.append(loss.item())

        optimizer.zero_grad()
        loss.backward()

        optimizer.step()

    print(f"Cost at epoch {epoch} is {sum(losses)/len(losses)}")
```


### Check Accuracy
然後再檢查 model 的 accuracy

```Python 
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

        print(
            f"Got {num_correct} / {num_samples} with accuracy {float(num_correct)/float(num_samples)*100:.2f}"
        )

    model.train()


print("Checking accuracy on Training Set")
check_accuracy(train_loader, model)

print("Checking accuracy on Test Set")
check_accuracy(test_loader, model)
```



----------
# *Reference*

- [Machine-Learn-Collection Custom Dataset (Images)](https://github.com/aladdinpersson/Machine-Learning-Collection/tree/master/ML/Pytorch/Basics/custom_dataset)