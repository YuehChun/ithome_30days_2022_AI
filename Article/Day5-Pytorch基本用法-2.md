# *Reference*

[Machine-Learn-Collection basics](https://github.com/aladdinpersson/Machine-Learning-Collection/blob/master/ML/Pytorch/Basics/pytorch_tensorbasics.py)

-------------

在處理Torch 的時候會需要很多數學的計算，例如加減乘除等計算元，因此整理一下記錄在下面

**值得注意的在的時候要特別注意是 `矩陣相乘` 還是 `value相乘` ，另外還有一個重點就是在相乘時候的 `dtype` 是否一樣**

```

__x__ = torch.tensor([1, 2, 3])
__y__ = torch.tensor([9, 8, 7])

# -- 加 --
result = __x__ + __y__
result = torch.add(x, __y__)

# -- 減 --
result = __x__ - __y__


# -- 乘 --
print(__x__ * __y__) # value 相乘，不是Matrix相乘


# -- 除 --
result = torch.true_divide(__x__ , __y__)  # 這要同樣的shape , [ 1/9 , 2/8 , 3/7]
print(result)

# -- 直接改變tensor變數 --
result = torch.zeros(3)
print(result)
result.add_(__x__)  # t = t + __x__
print(result)
result += __x__  # 同上
print(result)



# -- 次方 --
result = __x__.pow(2)  # __x__的二次方, [1,4,9]
result = __x__ ** 2  # 同上
print(result)

# -- 比大小 --
result = __x__ > 1  # [False, True, True]


# -- 矩陣相乘 --
__z__ = torch.rand((3, 1))
result = __z__.mm(__x__.reshape(1,3).float()) 
# 輸出的大小為(2,5)*(5,3) = (2,3)
# 注意 dtype 跟 shape, 
# 跟 torch.mm(__x__, __z__)一樣結果
print(result)

# -- 矩陣次方 --
print(result.matrix_power(3))  # 等同於matrix_exp(mm)

# -- Dot product --
result = torch.dot(__x__ , __y__)  # Dot product, result = 1*9 + 2*8 + 3*7
print(result)



```


另外還有另外一些 torch 的函數式可以使用，例如：
`x.max(dim=0)` 跟 `x.min(dim=0)` 在做資料探索的時候也常常被使用

另外可以稍微記一下的就是 `torch.clamp` ，因為過濾 `離群值 Outlier` 其實也是很好用的東西


```

__x__ = torch.rand((5, 5))
print(__x__)

values, indices = torch.max( __x__ , dim=0) 
# values, indices = 等同 __x__.max(dim=0)
# torch.argmax(__x__, dim=0) 也一樣找 indices
print(f"Max : values = {values} , indices ={indices} ", ) 
values, indices = torch.min(__x__, dim=0)
# values, indices = 等同 __x__.min(dim=0)
# torch.argmin(__x__, dim=0) 也一樣找 indices
print(f"Min : values = {values} , indices ={indices} ", ) 

print("Sum of dim-0 : " ,torch.sum( __x__, dim=0 ))

print("Abs :"  ,torch.abs(__x__))

print("means : ", torch.mean(__x__.float(), dim=0))  # __x__需要為float型態

result, indices = torch.sort(__x__, dim=0, descending=False)
print(result)

print("="* 5)

print(torch.clamp(result, min=0.3, max=0.6))
# 如果<min then min else 不改變
# 如果>max then max else 不改變
# torch.clamp(result, min=0) -> ReLu Function

__x___bool = torch.tensor([1, 0, 1], dtype=torch.bool)
print("bool tensor any : ", torch.any(__x___bool))
print("bool tensor all : ", torch.all(__x___bool))

```

今天就補充到這邊，這部分都是從 **Reference**  上面整理出來的資料，然後一樣要丟在 `Dash` 上面，如果要直接執行的話也是可以，但還是希望自己能夠建立自己的 `Docset`
 
----------
# *Reference*
[Machine-Learn-Collection basics](https://github.com/aladdinpersson/Machine-Learning-Collection/blob/master/ML/Pytorch/Basics/pytorch_tensorbasics.py)