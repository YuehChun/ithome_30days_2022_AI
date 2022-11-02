### Dash

這是第二篇，如果要看Dash的基本用法的話請參考上篇。

接著來講 **建立自己的Docs** 然後再說明如何 **建立自己片段文字keyword** ，兩個整合起來就是利用`keyword` 來呼叫自己建立的Snippets。

為什麼這部分會特別提出來講，主要是在 *資料分析* 或在 *資料前處理* 的時候，常常會忘記一些細節，例如: 要執行 `Scaler` 但是只記得Scaler，忘記要用`fit_transform`，或執行 `Data Augmentation` 需要執行 `SMOTE` 時要先 Import 什麼 Package ，然後要怎麼寫，這都需要花點時間從Google找到資料，甚至一個一個result 慢慢試順便回憶之前遇到相似的狀況是怎麼處理的？

因此每次做過或者整理過後的 `Functions` ，我都會建議稍微再精簡整理一下，然後拆成小小的 `Snippets` 再放到Dash Docs裡面，之後只需要輸入 `Keyword` 就可以順利把整理完後的 `Snippets` 叫出來使用，下方就是Step by step 的教怎麼建立 Dash Docs ，然後 Snippet 的強大功能。

### 建立自己的`Docs`

首先隨手捻來先一段簡單的 plot 正副座標的圖表
```
import numpy as np
import matplotlib.pyplot as plt

a=np.linspace(0,5,100)

y1 = np.sin(2 * np.pi * a)
y2 = np.cos(2 * np.pi * a)

fig, ax1 = plt.subplots()
color1 , color2 = "#696D7D" ,  "#8FC0A9"

ax1.set_xlabel('time (s)')
ax1.set_ylabel('sin', color=color1)
ax1.plot(a, y1, color=color1)
ax1.tick_params(axis='y', labelcolor=color1)

ax2 = ax1.twinx()  
ax2.set_ylabel('cos', color=color2)  
ax2.plot(a, y2, color=color2)
ax2.tick_params(axis='y', labelcolor=color2)

fig.tight_layout()
plt.show()
```


### 建立自己`片段文字Keyword`

先`Control` + `C` 先複製上面那段程式碼，然後切換到Dash 之後在 `Control` + `N` 再貼上(`Control` + `V`) 或直接使用 `Control` + `Shift` + `N` 就可以直接把剪貼簿中的內容直接帶入，結果就會像這樣(如果沒顏色的話`Snippet Abberviation` 右邊可以選擇語言)

![[Pasted image 20220908164016.png]]

並在上方的 `Snippet Abbreviation` 打上自己喜歡的關鍵字，這邊用 `!test_2y_plot` 這段來示範使用(前面有一個驚嘆號！，不過我私底下都喜歡用一小點) 

所以完成就像這樣
![[Pasted image 20220908164114.png]]
*如果不使用要記得刪除*

##### 搜尋Keyword
如果像我一樣常常忘記裡面自己下的 `Snippet Abbreviation` 也可以直接在 Dash 上面 `command` + `L` 就可以在上方搜尋 `keyword` ，例如可以搜尋 `show` 就會找到我們這個程式碼

##### `Snippet Abbreviation` 的使用

現在要講到重點了就是 `Snippet` 的使用方式，先打開 `Jupyter Notebook` 然後 **就先新增一個筆記本然後到可以打code的地方**。

![[Pasted image 20220908164947.png]]

然後在Dash 打開的狀況下(背景執行或者怎樣都可以就是不能 `command` +`Q` 關掉)，直接輸入 剛剛我們設定的  `Snippet Abbreviation` 也就是範例 : `!test_2y_plot` 

**PS:可以修改打錯，但是如果跳出或者切換地方就要重打，所以麻煩一次打對，也就是不要設太長**


![[Pasted image 20220908165357.png]]

輸入完後就會直接跳出剛設定好的 `Snippet` 這就是最簡單的使用方式。然後執行就完成了。

![[Pasted image 20220908165417.png]]

這個 `Snippet` 功能可以用在很多的地方，尤其是對於一些 Pytorch 或者 TensorFlow 的架構都可以先用這種方式整理一下存起來，以便未來要使用的時候，可以直接尻出來改就好了，相當的省時

### `Snippet 設定 變數`

除此之外，另外有更神奇的東西是他還有變數的功能，一樣的 `Snippet` 然後在剛剛顏色設定那邊把中間的變數動一下手腳 *(前後都要加2個底線)*
- `color1` 改成 `__color1__` 
- `color2` 改成 `__color2__`  

![[Pasted image 20220908171432.png]]

這邊改好之後，接下來在一樣在 `Jupyter notebook` 上試著打 `Snippet Abbreviation` 也就是範例 : `!test_2y_plot` 

此時畫面就會出現這個就是叫你打文字，所以可以接著打 `"red"` 就會把`__color1__` 全都變成 `"red"` 然後 `Enter` 就可以接下來改 `__color2__`

![[Pasted image 20220908172059.png]]


這變數後續應用就直接換成自己在處理的資料的片段程式碼然後再套上變數就可以不斷地到處去使用，感覺很強，說穿了也只是記 `keyword`

這邊再分享一小段我自己在處理 `SMOTE` 的 `Snippet` ，只要套用 `X` 跟 `Y` 就可以直接把 `SMOTE` 的功能寫出來，對腦霧的人來說，**能偷懶就給推！**


```
from imblearn.over_sampling import SMOTE

pre_X = __X__
pre_Y = __Y__

print("__X__.shape = [%d,%d]" % (pre_X.shape))
print("__Y__.shape = [%d,%d]" % (pre_X.shape))

smote_make = SMOTE(k_neighbors=5,random_state=69)
X_smote, Y_smote = smote_make.fit_resample(pre_X, pre_Y)


SMOTE_X = pd.concat([pre_X,X_smote], axis=0)
SMOTE_Y = pd.concat([pre_Y,Y_smote], axis=0)


print("SMOTE_X.shape = [%d,%d]" % (SMOTE_X.shape))
print("SMOTE_Y.shape = [%d,%d]" % (SMOTE_Y.shape))
```

以上說明就是 Dash 強大的 `Snippet` 的功能，如果有更好的軟體的話歡迎底下留言介紹，謝謝


### Reference 
[Dash](https://kapeli.com/dash)