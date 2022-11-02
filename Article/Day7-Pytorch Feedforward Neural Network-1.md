# *Reference*

[Machine-Learn-Collection Feedforward Neural Network](https://github.com/aladdinpersson/Machine-Learning-Collection/blob/master/ML/Pytorch/Basics/pytorch_simple_fullynet.py)

----------


### 先載入需要用的 package

- `torchvision` 是處理影像相關的工具
- `nn.functional` 是例如 activation functions 在用的，例如 : `ReLu` `sigmoid` 等等
- `datasets` 跟 `DataLoader` 這兩個很重要常常會一起使用
- `transforms` 在 **data augmentation** 階段會使用到，因為影像轉向的話可以增加多的角度來協助判斷
- `optim` 就是像 **SGD**  跟 **Adam** 這些優化器
- `nn` **neural network** 的模組
- `tqdm` 是處理進度條的小工具

這邊些把上述的東西先包起來成一個 `Snippet` 叫做 `import_torch` 這樣下次直接輸入 `Snippet` 就可以直接輸入了，如果有問題可以參考 【Day3】的文章

``` python
import torch
import torchvision
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms  
from torch import optim
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
```

建立 **Dash Snippet** -> `torch_import`

### 設定訓練用的變數(hyperparameyers)

這邊先設定一些訓練model用的變數，最後如果要跑超參數的話，就這邊處理就可以了 in

``` python
input_dim = 784
output_dim = 10
learning_rate = 0.001
batch_size = 64
num_epochs = 3

```

再來就是介紹資料集的部分，首先要先載入資料集，這邊使用內建的 `MNIST` ，然後分別載入到 `當前目錄的dataset` ，然後轉換成 Tensor 的形式，為了就是讓 `DataLoader` 可以直接處理
而 `DataLoader` 這部分就是會配合  `batch的大小` 來做切分資料及，然後還有 `shuffle` 的功能，另外補充 `shuffle` 後的資料對 `訓練model` 比較好

```python

train_dataset = datasets.MNIST(root="dataset/", train=True, transform=transforms.ToTensor(), download=True)
test_dataset = datasets.MNIST(root="dataset/", train=False, transform=transforms.ToTensor(), download=True)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)
```

建立 **Dash Snippet** -> `torch_MNIST`

### 建立 model

然後 Import 完成之後，就是先建立一下Model
先宣告兩個 `Linear` ，如果這邊有使用 `Snippet` 的話要特別注意 `__init__` 因為會被判斷成 `變數` 所以要注意
而 init 主要就是宣告我們這邊有 兩個 `Linear function` 
1. input : input_dim , output : 50
2. input : 50 , output : output_dim

然而最重要的是 `Forward function`  ，這是負責把剛剛宣告的 `Linear function` 串起來，這邊還使用了 **activation function** `ReLu` 處理第一個  **Linear function** `fc1` 的結果，然後在傳到 Linear function `fc2` 上計算後的結果就是 `output_dim`

```python

class DNN(nn.Module):
    def __init__(self, input_dim, output_dim): #Notice!!!!!!
        super(__ModelName__, self).__init__()
        self.fc1 = nn.Linear(input_dim, 50)
        self.fc2 = nn.Linear(50, output_dim)

    def forward(self, x):
        
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
```


### 指定 Device 類別

再來就是指定Device，如果有GPU可以用的話就是用cuda，反之就是使用CPU來執行，下一天就會一直使用這部分

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```


建立 **Dash Snippet** -> `torch_device`

----------
# *Reference*
[Machine-Learn-Collection Feedforward Neural Network](https://github.com/aladdinpersson/Machine-Learning-Collection/blob/master/ML/Pytorch/Basics/pytorch_simple_fullynet.py)