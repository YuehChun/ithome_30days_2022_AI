# *Reference*

[Machine-Learn-Collection basics](https://github.com/aladdinpersson/Machine-Learning-Collection/blob/master/ML/Pytorch/Basics/pytorch_tensorbasics.py)

-------------
### Torch 的index方法
index 在資料處理的過程中一定會被拿來使用，而比較常用就是簡單的`indices的方法`，或者最下方利用`二維` 的方式進行索引，例如下放 `x<2 | x>8` 這種x的filter其實在應用上很常用，另外函數的 `where`  也在資料處理的時候常常使用的到

```

__x__ = torch.arange(10)
print(x.shape)

indices = [2, 5, 8]
print(__x__[indices]) # x[indices] = [2, 5, 8]

x = torch.arange(10)
print(__x__[(__x__ < 2) | (__x__ > 8)])
print(__x__[__x__.remainder(2) == 0]) #[0, 2, 4, 6, 8]

print(torch.where(__x__ > 5, __x__, __x__ * 2))
# where 後面有三個參數，如果x>5 的話就return x, 否則回傳x*2

print(__x__.ndimension()) # x的維度

print(torch.tensor([0, 0, 1, 2, 2, 3, 4]).unique())
# unique 是用來找出不重複的數值

  

__y__ = torch.rand((3, 5))
print(__y__)
rows = torch.tensor([1, 0])
cols = torch.tensor([4, 0])
print(__y__[rows, cols]) # print(__y__[1,4] and __y__[0,0])

  

print(__y__.numel())# __y__裡面有多少數值
```


再來討論的是 torch 的 `shape轉換` ，在資料型態的部分常常會減少或者 `改變shape` 

```

__x__ = torch.arange(9)
result = __x__.reshape(3, 3)
print(result) # [9] 轉換成 [3,3]

result = result.t()
print(result) # 轉置

# If we instead have an additional dimension and we wish to keep those as is we can do:
batch = 64
__y__ = torch.rand((batch, 2, 5))
result = __y__.reshape(batch, -1) 
print(result.shape) # [64,10]

result = __y__.permute(0, 2, 1)
print(result.shape) # 原本 (64,2,5) => (64,5,2)

print(torch.cat((__y__, __y__), dim=0).shape)  # Shape: [128,2,5] 
print(torch.cat((__y__, __y__), dim=1).shape)  # Shape: [64,4,5]

# 增加或者減少維度 (不是降維)
print(__y__.unsqueeze(0).shape)  # [64,2,5]增加維度 ->[1, 64, 2, 5]
print(__y__.unsqueeze(1).shape)  # [64,2,5]增加維度 ->[64, 1, 2, 5]

# __x__ 為[9]
result = __x__.unsqueeze(0).unsqueeze(1)
print(result.shape) # [1,1,9]

result = result.squeeze(2)  
print(result.shape) # [1,1]

```
 
----------
# *Reference*
[Machine-Learn-Collection basics](https://github.com/aladdinpersson/Machine-Learning-Collection/blob/master/ML/Pytorch/Basics/pytorch_tensorbasics.py)