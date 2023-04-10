## 1. 引言

### 1.1 目标函数：定义模型的优劣程度的度量

### 1.2 优化算法：梯度下降（gradient descent）

### 1.3 监督学习：给定特征预测标签

![image-20230330165001981](/Users/jerseylu/Library/Application Support/typora-user-images/image-20230330165001981.png)

回归、分类、标记问题、搜索、推荐系统、系列学习

### 1.4 无监督学习：不含有“目标”的机器学习问题

聚类、主成分分析（捕捉数据的线性相关属性）、因果关系、生成对抗性网络（检查真实和虚假数据是否相同）

### 1.5 半监督学习：一半知道label，另一半不知道label

### 1.6 强化学习：基于环境的反馈而行动，使得整体收益最大化

![image-20230330170529676](/Users/jerseylu/Library/Application Support/typora-user-images/image-20230330170529676.png)

当环境可被完全观察到时，强化学习问题被称为*马尔可夫决策过程*（markov decision process）。 当状态不依赖于之前的操作时，我们称该问题为*上下文赌博机*（contextual bandit problem）。 当没有状态，只有一组最初未知回报的可用动作时，这个问题就是经典的*多臂赌博机*（multi-armed bandit problem）。

# 2. 预备知识

## 2.1 数据操作

### 2.1.1 广播机制（broadcasting mechanism）：复制元素使张量具有相同的形状

### 2.1.2 索引和切片：第一个元素的索引是0，最后一个元素索引是-1

### 2.1.3 节省内存：创建一个新的矩阵`Z`，其形状与另一个`Y`相同， 使用`zeros_like`来分配一个全0的块

```python
Z = torch.zeros_like(Y)
print('id(Z):', id(Z))
Z[:] = X + Y
print('id(Z):', id(Z))
```

### 2.1.4 转换为其他python对象

*  torch张量和numpy数组

```python
A = X.numpy()
B = torch.tensor(A)
type(A), type(B)
```

* 将大小为1的张量转换为Python标量

```python
a = torch.tensor([3.5])
a, a.item(), float(a), int(a)
```

## 2.2 数据预处理

### 2.2.1 读取数据集

```python
os.makedirs(os.path.join('..', 'data'), exist_ok=True)
data_file = os.path.join('..', 'data', 'house_tiny.csv')
with open(data_file, 'w') as f:
```

### 2.2.2处理缺失值

```python
data.dropna(axis = 1) #删除包含任何缺失值的行
values = {"col":,"col":}
data.fillna（value = values）#填充值
data["col"].fillna(data["col"].mean()) #平均值填充
```

### 2.2.3 转换为张量格式：torch.tensor

## 2.3 线性代数

### 2.3.1 标量：只有一个元素的张量

### 2.3.2 向量：标量值组成的列表，一维张量

### 2.3.3 矩阵：将向量从一阶推广到二阶，具有两个轴的张量

### 2.3.4 张量：描述具有任意数量轴的n维数组的通用方法（向量是一阶张量，矩阵是二阶张量)

### 2.3.5 张量算法的性质：两个矩阵的按元素乘法称为*Hadamard积*（Hadamard product）（eg：A ⊙ B）

```python
A * B
```

### 2.3.6 降维

```python
A.sum() # 所有元素求和
A.sum(axis=1) # axis=1将通过汇总所有列的元素降维
A.sum(axis=[0, 1])  # 结果和A.sum()相同 
A.mean(), A.sum() / A.numel() # 求平均值
```

```python
sum_A = A.sum(axis=1, keepdims=True) 
A / sum_A # 非降维求平均值
A.cumsum(axis=0) # 某个轴计算A元素的累积总和
```

### 2.3.7 点积（Dot product) ： torch.dot( ) 

### 2.3.8 矩阵的向量积：torch.mv(A, x)

### 2.3.9 矩阵-矩阵乘法：torch.mm(A, B)

### 2.3.10 范数：向量的分量大小

* L1范数：向量元素的绝对值之和

  torch.abs( ).sum( )

* L2范数：向量元素平方和的平方根 torch.norm( )

* *Frobenius范数*（Frobenius norm）：矩阵元素平方和的平方根

  torch.norm.(torch.ones(( )))