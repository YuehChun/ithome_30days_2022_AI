
# *Reference*

[Machine-Learn-Collection basics](https://github.com/aladdinpersson/Machine-Learning-Collection/blob/master/ML/Pytorch/Basics/pytorch_tensorbasics.py)

-------------


**這邊就是直接使用網路上人家整理好的文件上做整理後使用，*接下來的程式碼建議都是先直接放在`Dash` 上面會比較適當*，雖然很偷懶但是他是真的整理的不錯，記得去幫忙按 `star` **


### 觀察自己的tensor 的狀態方法
使用時機簡單來說就是看一下 `torch` 是在哪個`device` 比較常使用，把這些比較容易忘記的語法放置在 `Dash` 上面，有時候需要就可以自己查詢 `Docs`

```
import torch

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

# 產製一個 2x3 (2 rows, 3 columns)
__tensor__ = torch.tensor(
    [[1, 2, 3], [4, 5, 6]], dtype=torch.float32, device=device, requires_grad=True
)

# tensor attributes
print( f"Information: {__tensor__}" )
print( f"Type: {__tensor__.dtype}" )  # (torch.float32)
print( f"Device: {__tensor__.device}" )  # cpu/cuda
print( f"Shape: {__tensor__.shape}")
print( f"Requires gradient: {__tensor__.requires_grad}")

```


### 建立torch的方法

建立一個 `torch` 的方法，常用的 `zeros` 跟 `ones` 都是常用的，然後整個貼上去 `Dash` 上面 `Print`  出來就知道自己需要用什麼樣的 `torch`

```
import torch
zeros__x__ = torch.zeros((3, 3))  # 3x3 with values of 0
print(zeros__x__)

rand__x__ = torch.rand((3, 3)) 
print(rand__x__)

ones__x__ = torch.ones((3, 3))  
print(ones__x__) # 3x3 with values of 1

eye__x__ = torch.eye(5, 5)
print(eye__x__)  # 斜線的5x5

arange__x__ = torch.arange( start=0, end=5, step=1)
print(arange__x__)  # 也可以 torch.arange(5)

linspace__x__ = torch.linspace(start=0.1, end=1, steps=10)  
print(linspace__x__)

normal__x__ = torch.empty(size=(1, 5)).normal_( mean=0, std=1 ) 
print(normal__x__) # 常態分佈 with mean=0, std=1

diag__x__ = torch.diag( torch.ones(3) )  
print(diag__x__) # Diagonal matrix of shape 3x3
```


### torch 轉換型態的方法
時常在處理資料的時候會需要轉換資料型態，因此這邊整理一寫長用的轉換型態的方法，例如： `long()` 和 `float()` 這兩個常常使用到

```
__tensor__ = torch.arange(4)
print(f"Converted int16 {__tensor__.short()}")  # To int16
print(f"Converted int64 {__tensor__.long()}") # To int64
print(f"Converted float16 {__tensor__.half()}")  # To float16
print(f"Converted float32 {__tensor__.float()}")  # To float32
print(f"Converted float64 {__tensor__.double()}")  # To float64

```

----------
# *Reference*
[Machine-Learn-Collection basics](https://github.com/aladdinpersson/Machine-Learning-Collection/blob/master/ML/Pytorch/Basics/pytorch_tensorbasics.py)


