# *Reference*

[Machine-Learn-Collection Feedforward Neural Network](https://github.com/aladdinpersson/Machine-Learning-Collection/blob/master/ML/Pytorch/Basics/pytorch_simple_fullynet.py)

----------
前一天 Day 7 講完了一些前置處理之後，接下來就是要看怎麼訓練NN跟評估NN的好壞

### 先 New 一個 NN

先new model，然後同時指定 `input_dim` 跟 `output_dim`
**這邊很重要，因為會dim的數量不一樣，model 會無法處理**


```python
model = __ModelName__(input_dim=input_dim, output_dim=output_dim).to(device)
```


### Loss function and Optimizer 

然後再來 **指定訓練NN 用的 Loss function 跟 Optimizer** ，分別為 `CrossEntropyLoss` 與 `Adam`

```python
# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
```

這邊會建立 **Snippet** `torch_loss` 跟 `torch_adam` ，後面呼叫比較快


### Train Neural Network
上面這些都宣告要如何訓練NN之後，接下來就是真的要跑訓練的程式了

```Python
for epoch in range(num_epochs):
    for batch_idx, (data, targets) in enumerate(tqdm(train_loader)):
        data = data.to(device=device)
        targets = targets.to(device=device)

        data = data.reshape(data.shape[0], -1)

        # forward
        scores = model(data)
        loss = criterion(scores, targets)

        # backward
        optimizer.zero_grad()
        loss.backward()

        # gradient descent or adam step
        optimizer.step()

```

這邊會建立 **Snippet** `torch_train`

### 檢查train and test dataset accuracy

再來先建立評估NN的 accuracy

```
def check_accuracy(loader, model):
    num_correct = 0
    num_samples = 0
    model.eval() #有補充在Day9

	# 檢測的時候，不要執行梯度下降
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device)
            y = y.to(device=device)
            x = x.reshape(x.shape[0], -1)

            scores = model(x)
            _, predictions = scores.max(1)
            # 計算預測結果是否跟y 相同
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

    model.train() #有補充在Day9
    return num_correct/num_samples
```

```Python
print(f"Accuracy on training set: {check_accuracy(train_loader, model)*100:.2f}")
print(f"Accuracy on test set: {check_accuracy(test_loader, model)*100:.2f}")
```

因為如果執行訓練後時使用的 dataset 都是使用 `train dataset` ，但是我們一開始還有另一個 `test dataset`，而 test dataset這邊的結果會互相比較，來檢測所訓練過後的 `NN model` 是否有 **over fitting** 的狀況

這邊會建立 **Snippet** `torch_eval`


**補充說明一下：如果有 overfitting 的現象代表說 train 跟 test 的accuracy 差異過大，這 Model generalization 不好，因此不建議使用 **


`

----------
# *Reference*
[Machine-Learn-Collection Feedforward Neural Network](https://github.com/aladdinpersson/Machine-Learning-Collection/blob/master/ML/Pytorch/Basics/pytorch_simple_fullynet.py)