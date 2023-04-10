# Notes-for-Deep-Learning

This is a personal study note, which includes recent ground-breaking research in the field of deep learning. Individuals who are interested at DL are welcome to discuss and study together.  ps: Only used for personal study!

## 1. AlexNet

[AlexNet](https://papers.nips.cc/paper_files/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html) is the name of a convolutional neural network (CNN) architecture, designed by *Alex Krizhevsky* in collaboration with *Ilya Sutskever* and *Geoffrey Hinton*. Alexnet competed in the ImageNet Large Scale Visual Recognition Challenge on 2012. The network achieved a top-5 error of 15.3%, more than 10.8 percentage points lower than that of the runner up. 

### The Architecture

* ** ReLU Function:**  

  In terms of training time with gradient descent, the non-saturating nonlinearity(ReLU) is faster than these saturating nonlinearities(sigmoid, tanh).

* **Training on Multiple GPUs:**

  Alexnet runs on two GPUs. Half of the kernels (or neurons) are handled by each GPU, and the GPUs only communicate at particular layers.

* **Local Response Normalization:**

  ReLUs have the desirable property that they do not require input normalization to prevent them from saturating. This sort of response normalization implements a form of lateral inhibition, which can improve the generalization ability of neural networks.

* **Overlapping Pooling:**

  overlapping pooling: Pooling size > stride size

  traditional local pooling: Pooling size = stride size

  Model with overlapping pooling has a better performance than the traditional local pooling, and can avoid overfitting.

<div align="center">
<img src="https://user-images.githubusercontent.com/104020492/230723149-71551e46-a06b-4c18-8c00-e7d8d03411e7.png" width="65%" height="40%" />
</div>


The code is from the book--[Dive into Deep Learning](https://d2l.ai/).

```python
import time
import torch
from torch import nn, optim
import torchvision
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 96, 11, 4), # in_channels, out_channels, kernel_size, stride, padding
            nn.ReLU(),
            nn.MaxPool2d(3, 2), # kernel_size, stride
            # 减小卷积窗口，使用填充为2来使得输入与输出的高和宽一致，且增大输出通道数
            nn.Conv2d(96, 256, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(3, 2),
            # 连续3个卷积层，且使用更小的卷积窗口。除了最后的卷积层外，进一步增大了输出通道数。
            # 前两个卷积层后不使用池化层来减小输入的高和宽
            nn.Conv2d(256, 384, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(384, 384, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(384, 256, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(3, 2)
        )
         # 这里全连接层的输出个数比LeNet中的大数倍。使用丢弃层来缓解过拟合
        self.fc = nn.Sequential(
            nn.Linear(256*5*5, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            # 输出层。由于这里使用Fashion-MNIST，所以用类别数为10，而非论文中的1000
            nn.Linear(4096, 10),
        )
def forward(self, img):
    feature = self.conv(img)
    output = self.fc(feature.view(img.shape[0], -1))
    return output
```

## 2. ResNet

[ResNet](https://arxiv.org/pdf/1512.03385.pdf) is the name of a residual learning framework, designed by *Kaiming He*, *Xiangyu Zhang*, *Shaoqing Ren* and *Jian Sun*. ResNet achieved 3.57% error on the ImageNet test set, which won the 1st place on the ILSVRC 2015 classification task. Besides that, it also obtained a 28% relative improvement on the COCO object detection dataset.

### Purpose: Optimizing the "degradation" problem of deeper neural network

* Previous work:  normalized initialization[^1][^2][^3][^4]; intermediate normalization layers[^5]

* This paper: Residual Building Block

<div align="center">
<img src="https://user-images.githubusercontent.com/104020492/230721790-09f6b67a-9400-4e27-bf63-b55c81b74251.png" width="40%" height="40%" />
</div>



Denote the input by x. The desired underlying mapping is f(x), to be used as input to the activation function on the top. 

On the left, the portion within the dotted-line box must directly learn the mapping f(x). On the right, the portion within the dotted-line box needs to learn the residual mapping g(x), which is how the residual block derives its name. 

If the identity mapping f(x)=x is the desired underlying mapping, the residual mapping amounts to g(x)=0 and it is thus easier to learn: we only need to push the weights and biases of the upper weight layer (e.g., fully connected layer and convolutional layer) within the dotted-line box to zero. 


<div align="center">
<img src="https://user-images.githubusercontent.com/104020492/230722383-884eaaff-71fa-4dde-9248-401e0c772b07.png" width="40%" height="50%" />
</div>


### The Architecture

<div align="center">
<img src="https://user-images.githubusercontent.com/104020492/230722972-334438c2-af6a-4583-9d62-ced54af154cf.png" width="40%" height="50%" />
</div>



[^1]: http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf
[^2]: http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf
[^3]: https://arxiv.org/pdf/1312.6120.pdf
[^4]: https://arxiv.org/pdf/1502.01852.pdf
[^5]: https://arxiv.org/pdf/1502.03167.pdf

 The code is available at this [blog](https://blog.csdn.net/weixin_39524208/article/details/124894216?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522168095565816800182751421%2522%252C%2522scm%2522%253A%252220140713.130102334..%2522%257D&request_id=168095565816800182751421&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~top_positive~default-3-124894216-null-null.142^v82^koosearch_v1,201^v4^add_ask,239^v2^insert_chatgpt&utm_term=Resnet&spm=1018.2226.3001.4187).


## 3.Transformer

[Transformer](https://arxiv.org/pdf/1706.03762.pdf) is a neutral network, designed by *Ashish Vaswani*, *Noam Shazeer*, *Niki Parmar*, *Jakob Uszkoreit*, *Llion Jones*, *Aidan N. Gomez*, *Łukasz Kaiser* and *Illia Polosukhin*. Transformer learns context and thus meaning by tracking relationships in sequential data like the words in the sentence.

### The Architecture

<div align="center">
<img src="https://user-images.githubusercontent.com/104020492/230881354-04b00853-31dc-49d3-bfbc-8c534d13250c.png" width="50%" height="50%" />
</div>

* **Encoder and Decoder stacks**

  **Encoder** is composed of a stack of N=6 identical layers. Each layer has 2 sub-layers. The first is a **multi-head self-attention mechanism**, and the second is a simple, **position-wise fully connected feed-forard network**. A **residual connection** is employed around each of the two sub-layers, followed by **layer-normalization**.

  **Decoder** is also composed of a stack of N=6 identical layers.  In addition to the two sub-layers in each encoder layer, the decoder inserts a third sub-layer, which performs **multi-head attention** over the output of the encoder stack. A **residual connection** is also employed around each of the two sub-layers, followed by **layer-normalization**.

* **Attention**

  An attention function can be described as mapping a query and a set of key-value pairs to an output, where the **query**, **keys**, **values**, and output are all vectors. The output is computed as a weighted sum of the values, where the weight assigned to each value is computed by a compatibility function of the query with the corresponding key. 

  <div align="center">
  <img src="https://user-images.githubusercontent.com/104020492/230832590-ad9f53c2-94b1-4489-8227-b9c3af266541.png" width="70%" height="70%" />
  </div>

  * **Scaled Dot-Product Attention**

    The input consists of queries and keys of dimension dk, and values of dimension dv. Computing the dot products of the query with all keys, divide each by pdk, and apply a softmax function to obtain the weights on the values.

  $$
  Attention(Q,K,V)=softmax(\frac{QK^T}{\sqrt{d_k}})V
  $$

  * **Multi-Head Attention**

    Multi-head attention allows the model to jointly attend to information from different representation subspaces at different positions. 

    Method: To linearly project the queries, keys and values h times with different, learned linear projections to dk, dk and dv dimensions, respectively.

$$
\begin{aligned}
  MultiHead(Q,K,V) &= Concat(head_1,…,head_h)W^O \\
  where \quad head_i &= Attention(QW_i^Q,KW_i^K,VW_i^V)
\end{aligned}
$$

* **Position-wise Feed-Forward Networks**

  In addition to attention sub-layers, each of the layers in encoder and decoder contains a **fully connected feed-forward network**, which is applied to each position separately and identically. This consists of two linear transformations with a **ReLU activation** in between.
  $$
  FFN(x)=max(0,xW_1+b_1)W_2+b_2
  $$

* **Embeddings and Softmax**
  * Using learned embeddings to convert the input tokens and output tokens to vectors of dimension dmodel.
  * Using the usual learned linear transformation and softmax function to convert the decoder output to predicted next-token probabilities.

* **Positional Encoding**

  Purpose: To make use of the order of the sequence because this model contains no recurrence and no convolution.

  Solution: Injecting some information about the relative or absolute position of the tokens in the sequence. To this end, adding "positional encodings" to the input embeddings at the bottoms of the encoder and decoder stacks.

  

  ### Highlight

  * Why Layer-Norm instead of Batch-Norm

  <div align="center">
  <img src="https://user-images.githubusercontent.com/104020492/230881098-380d7ed1-33af-4e3f-80d8-509c68811e43.jpeg" width="50%" height="50%" />
  </div>

  * Why Self-Attention instead of RNN

  <div align="center">
  <img src="https://user-images.githubusercontent.com/104020492/230881160-df59e81a-3417-49a6-8e1e-9c27bf624b08.jpeg" width="70%" height="70%" />
  </div>

  The code is available at [this](https://github.com/tensorflow/tensor2tensor).

  




