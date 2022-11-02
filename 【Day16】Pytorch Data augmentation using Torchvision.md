# *Reference*

- [Machine-Learn-Collection Data augmentation using Torchvision](https://github.com/aladdinpersson/Machine-Learning-Collection/blob/master/ML/Pytorch/Basics/pytorch_transforms.py)
引用這作者的範例，以我個人的角度記錄及說明
-------------

### 前言
今天來講講訓練影像資料的時候會常常遇到問的問題，例如不是每個張影像中的物品都是直向的（*由上到下*），有可能該物品會是橫向的（*由左到右*）。
如果我們一直訓練機器去學習判斷直向的物品，那圖然有一個影像的同樣物品是左到右，這樣機器就會無法判斷出來。
因此為了解決影像有角度的問題，因此做了` Data augmentation` ，為了就是製造不同角度的照片，因為一張圖片，你正著看跟反著看，物品就是物品，但是機器如果沒看過反著的，就會誤判

### 首先第一段就是 import 

**Dash 來呼叫** `torch_import`



### 建立Customer Dataset
這邊使用前一天所宣告的 `CatsAndDogsDataset` 作為這次示範的 Dataset 

```Python
dataset = CatsAndDogsDataset(

csv_file="/content/drive/MyDrive/Colab Notebooks/ithome/cats_dogs.csv",

root_dir="/content/drive/MyDrive/Colab Notebooks/ithome/cats_dogs",

transform=transforms.ToTensor(),

)
```

然後我們使用的Dataset 圖片的資料只有10張，縮圖大概如下：

![[D16-1.png]]

###  Transforms 的方法

如果要做 `Data Augmentation` 就要建立一個 `transforms.Compose` 然後把想要做的不同的轉換方法依序寫入，如下方的程式碼
而我們做Data Augmentation 的項目依序有
1. 先轉成 PILImage
2. Resize 256x256
3. 隨機取224x224的大小出來
4. 更改明亮度
5. 然後再隨機的機率轉成灰階影像
6. 輸出為Tensor

然後並且在 **Dataset 那邊的時候 transform 就可以指定這個方法**

```Python

my_transforms = transforms.Compose(
    [   
        transforms.ToPILImage(),
        transforms.Resize((256, 256)),  # 先全部Resize為256,256
        transforms.RandomCrop((224, 224)),  # 然後全部在隨機取大小為224ｘ224的烏來
        transforms.ColorJitter(brightness=0.5),  # 更改明亮度
        transforms.RandomGrayscale(p=0.2),  # 設定轉變成灰階的機率為 0.2
        transforms.RandomRotation(
            degrees=45
        ),  # 隨機轉向的方向-45 到 45度
        transforms.ToTensor(),  # 最後再轉乘To
    ]
)
```

### 建立 Dataset
這邊就是比較關鍵的地方，建立Dataset的同時就要指定 **Data Augmentation 的方法**，而我們剛剛建立的Data Augmentation 的名稱為 `my_transforms`
```Python
aug_dataset = CatsAndDogsDataset(
    csv_file="/content/drive/MyDrive/Colab Notebooks/ithome/cats_dogs.csv",
    root_dir="/content/drive/MyDrive/Colab Notebooks/ithome/cats_dogs",
    transform=my_transforms,
)
```

### 執行

執行 `Data augmentation` ，這邊是一直不段重複的去取dataset的欄位，但是經過dataset的定義的時候，在輸出之前都需要執行 `my_transforms` 的方法，所以不斷地取 dataset 10次，每次的影像都做一些變化，因此執行完後影像就會產生很多
```Python

img_num = 0
for _ in range(10):
	# 每次都存取一次 aug_dataset
    for  img , label in aug_dataset:
        save_image(img, '/content/drive/MyDrive/Colab Notebooks/ithome/aug_cats_dogs/aug_'+str(img_num)+'.png')
        img_num+=1
```



最後的結果
![[D16-2.png]]

### 結論

產生很多不同角度與明亮度差異的圖片後，再丟給機器去學習，機器看的影像資料更多了對於後續應用識別度的時候判斷才會更加的準確（*因為考慮各種角度及各種明亮度*)

----------
# *Reference*

- [Machine-Learn-Collection Data augmentation using Torchvision](https://github.com/aladdinpersson/Machine-Learning-Collection/blob/master/ML/Pytorch/Basics/pytorch_transforms.py)