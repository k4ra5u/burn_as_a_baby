# burn_as_a_baby
> 一个磕盐之余自娱自乐的项目
## 游戏简介
- 前两天朋友给我转了微信上一个叫做开局托儿所的小游戏，简单来讲是一个棋盘的1-9的数字，我们需要不断框选一个矩形做消消乐，这个矩形里的数字和为10即可消除
- 真是相当刺激的小游戏，所以准备直接交给程序来解决，结果写着写着发现我还是想简单了
## 用法
- `dfs_loop.py`是主程序，自动进行每局游戏，不断搜素局部最优路径，并在游戏临近结束时根据最优解完成游戏。*冲分！！*
- 交互：执行本身不需要任何参数`python dfs_loop.py`，代码中需要根据实际情况修改部分参数（当前参数针对默认的限时游戏）
```python
rows = 16 # 当前棋盘的行数
cols = 10 # 当前棋盘的列数
use_api = True # 是否使用ocr api
accepted_target = 140 # 当超过这个最高分时立即结束游戏
next_solve_min_times = 4 # dfs的玄学优化：当dfs深度达到当前最大值4次之后，执行下一步的迭代求解
next_solve_max_times = 15 # dfs的玄学优化：当dfs深度达到当前最大值15次之后.放弃当前分支（将能量向其他分支调度）
```
## 算法简介
**识别&交互**。本机是Mac，因此控制鼠标点击和移动的函数并未在Windows和Linux上进行测试，如有需要请自行测试和替换相应系统的接口。OCR使用的是阿里云的图片识别接口，需要自行注册阿里云账户并开通OCR业务，每月200次免费调用，并存储于np.array中。

**解法**。简单利用拓扑排序的思想，首先考虑下面这种情况：
```
4 5 5 6
```
这种情况下`1*4`的矩形A和为20，中间`1*2`的矩阵B和为10，那么我们可以建立一个由B指向A的边，也就是将图中所有和为10的倍数的矩形看成一个点，用全包含关系看作一个有向边，构建一个有向拓扑图。理论上我们不断找入度为0的边就好，但是后续情况更为复杂。比如以下这种情况：
```
4 9 
3 4
```
这种情况下虽然矩阵和为10的倍数，但是不包含任何值为10的矩形，因此尽管入度为0，但这是一个废点。另外拓扑排序的顺序也会对结果产生很大的影响：
```
X 1 X
1 9 1
X 1 X
```
9无论和上下左右哪个消除，另外3个均无法构成10的矩形。更棘手的是下面这种情况：
```
9 4 1 
X 6 X
```
这种情况下 9 4 1 因为和不是10的倍数，因此不会被计入为节点，但是当4 6消除掉后 9 0 1的矩阵又满足了10的倍数，也就是说我们每消除一个矩阵都要考虑对有部分重叠的矩形的值的影响。枚举所有产生影响的矩阵复杂度过高，但是能够达成何为10的倍数的矩阵应该不是很多。感觉会有大佬想出奇奇怪怪的数据结构来优化这种动态更新问题，或是使用深度学习，由AI来进行决策。但是无奈算法实在太烂，只能使用暴力搜索来枚举可能的决策。

**暴力搜索**。solve函数代表当前矩阵下的最优解，根据上文的建图方式构建好后运行dfs寻找消除最多的情况。dfs除了考虑入度外还考虑是否用过经过点vis，以及这个矩形和哪些矩形产生部分重叠关系，一旦选择了某些节点代表的矩形，和这个矩形产生部分重叠的其他矩形都不会被选中，但是不影响完全重叠的矩形，相反将这些完全重叠的矩形入度-1，用来模拟拓扑排序。但是考虑到不同的选择顺序将会导致完全不同的结果，所以我们没有只使用拓扑排序，而是dfs搜索可能的选择序列。

当选无可选时，我们将当前的矩阵重新传入solve中迭代求解，数轮迭代后，整张图都没有可选的节点了，此时统计成绩并更新最好成绩。
