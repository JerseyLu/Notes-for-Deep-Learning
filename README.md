# Notes-for-Deep-Learning
This is a personal study note, which includes recent ground-breaking research in the field of deep learning. Individuals who are interested at DL are welcome to discuss and study together.  ps: Only used for personal study!

## 1. AlexNet
[AlexNet](https://papers.nips.cc/paper_files/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html) is the name of a convolutional neural network (CNN) architecture, designed by Alex Krizhevsky in collaboration with Ilya Sutskever and Geoffrey Hinton. Alexnet competed in the ImageNet Large Scale Visual Recognition Challenge on 2012. The network achieved a top-5 error of 15.3%, more than 10.8 percentage points lower than that of the runner up. 

### The Architecture

* ReLU Function:  

  In terms of training time with gradient descent, the non-saturating nonlinearity(ReLU) is faster than these saturating nonlinearities(sigmoid, tanh).

* Training on Multiple GPUs:  

  Alexnet runs on two GPUs. Half of the kernels (or neurons) are handled by each GPU, and the GPUs only communicate at particular layers.
  
* Local Response Normalization:

  ReLUs have the desirable property that they do not require input normalization to prevent them from saturating. This sort of response normalization implements a form of lateral inhibition, which can improve the generalization ability of neural networks.

* Overlapping Pooling:

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
[ResNet](https://arxiv.org/pdf/1512.03385.pdf) is the name of a residual learning framework, designed by Kaiming He, Xiangyu Zhang, Shaoqing Ren and Jian Sun. ResNet achieved 3.57% error on the ImageNet test set, which won the 1st place on the ILSVRC 2015 classification task. Besides that, it also obtained a 28% relative improvement on the COCO object detection dataset.

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

### The architecture

<div align="center">
<img src="https://user-images.githubusercontent.com/104020492/230722972-334438c2-af6a-4583-9d62-ced54af154cf.png" width="40%" height="50%" />
</div>


[^1]:http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf
[^2]:http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf
[^3]:https://arxiv.org/pdf/1312.6120.pdf
[^4]:https://arxiv.org/pdf/1502.01852.pdf
[^5]:https://arxiv.org/pdf/1502.03167.pdf
 
 The code is from a [blog](https://blog.csdn.net/weixin_39524208/article/details/124894216?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522168095565816800182751421%2522%252C%2522scm%2522%253A%252220140713.130102334..%2522%257D&request_id=168095565816800182751421&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~top_positive~default-3-124894216-null-null.142^v82^koosearch_v1,201^v4^add_ask,239^v2^insert_chatgpt&utm_term=Resnet&spm=1018.2226.3001.4187).

```python
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, utils
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data.dataset import Dataset
from torchvision.transforms import transforms
from pathlib import Path
import cv2
from PIL import Image
import torch.nn.functional as F
%matplotlib inline
%config InlineBackend.figure_format = 'svg' # 控制显示


transform = transforms.Compose([ToTensor(),
                                transforms.Normalize(
                                    mean=[0.5,0.5,0.5],
                                    std=[0.5,0.5,0.5]
                                ),
                                transforms.Resize((224, 224))
                               ])

training_data = datasets.CIFAR10(
    root="data",
    train=True,
    download=True,
    transform=transform,
)

testing_data = datasets.CIFAR10(
    root="data",
    train=False,
    download=True,
    transform=transform,
)

class BasicBlock(nn.Module):
    def __init__(self,in_channels,out_channels,stride=[1,1],padding=1) -> None:
        super(BasicBlock, self).__init__()
        # 残差部分
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels,out_channels,kernel_size=3,stride=stride[0],padding=padding,bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True), # 原地替换 节省内存开销
            nn.Conv2d(out_channels,out_channels,kernel_size=3,stride=stride[1],padding=padding,bias=False),
            nn.BatchNorm2d(out_channels)
        )

        # shortcut 部分
        # 由于存在维度不一致的情况 所以分情况
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                # 卷积核为1 进行升降维
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride[0], bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
#         print('shape of x: {}'.format(x.shape))
        out = self.layer(x)
#         print('shape of out: {}'.format(out.shape))
#         print('After shortcut shape of x: {}'.format(self.shortcut(x).shape))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

# 采用bn的网络中，卷积层的输出并不加偏置
class ResNet18(nn.Module):
    def __init__(self, BasicBlock, num_classes=10) -> None:
        super(ResNet18, self).__init__()
        self.in_channels = 64
        # 第一层作为单独的 因为没有残差快
        self.conv1 = nn.Sequential(
            nn.Conv2d(3,64,kernel_size=7,stride=2,padding=3,bias=False),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        # conv2_x
        self.conv2 = self._make_layer(BasicBlock,64,[[1,1],[1,1]])
        # self.conv2_2 = self._make_layer(BasicBlock,64,[1,1])

        # conv3_x
        self.conv3 = self._make_layer(BasicBlock,128,[[2,1],[1,1]])
        # self.conv3_2 = self._make_layer(BasicBlock,128,[1,1])

        # conv4_x
        self.conv4 = self._make_layer(BasicBlock,256,[[2,1],[1,1]])
        # self.conv4_2 = self._make_layer(BasicBlock,256,[1,1])

        # conv5_x
        self.conv5 = self._make_layer(BasicBlock,512,[[2,1],[1,1]])
        # self.conv5_2 = self._make_layer(BasicBlock,512,[1,1])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    #这个函数主要是用来，重复同一个残差块
    def _make_layer(self, block, out_channels, strides):
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels
        return nn.Sequential(*layers)
    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)

#         out = F.avg_pool2d(out,7)
        out = self.avgpool(out)
        out = out.reshape(x.shape[0], -1)
        out = self.fc(out)
        return out
    
# 保持数据集和测试机能完整划分
batch_size=100
train_data = DataLoader(dataset=training_data,batch_size=batch_size,shuffle=True,drop_last=True)
test_data = DataLoader(dataset=testing_data,batch_size=batch_size,shuffle=True,drop_last=True)

images,labels = next(iter(train_data))
print(images.shape)
img = utils.make_grid(images)
img = img.numpy().transpose(1,2,0)
mean=[0.5,0.5,0.5]
std=[0.5,0.5,0.5]
img = img * std + mean
print([labels[i] for i in range(64)])
plt.imshow(img)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = res18.to(device)
cost = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())

print(len(train_data))
print(len(test_data))
epochs = 10
for epoch in range(epochs):
    running_loss = 0.0
    running_correct = 0.0
    model.train()
    print("Epoch {}/{}".format(epoch+1,epochs))
    print("-"*10)
    for X_train,y_train in train_data:
        # X_train,y_train = torch.autograd.Variable(X_train),torch.autograd.Variable(y_train)
        X_train,y_train = X_train.to(device), y_train.to(device)
        outputs = model(X_train)
        _,pred = torch.max(outputs.data,1)
        optimizer.zero_grad()
        loss = cost(outputs,y_train)

        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        running_correct += torch.sum(pred == y_train.data)

    testing_correct = 0
    test_loss = 0
    model.eval()
    for X_test,y_test in test_data:
        # X_test,y_test = torch.autograd.Variable(X_test),torch.autograd.Variable(y_test)
        X_test,y_test = X_test.to(device), y_test.to(device)
        outputs = model(X_test)
        loss = cost(outputs,y_test)
        _,pred = torch.max(outputs.data,1)
        testing_correct += torch.sum(pred == y_test.data)
        test_loss += loss.item()
    print("Train Loss is:{:.4f}, Train Accuracy is:{:.4f}%, Test Loss is::{:.4f} Test Accuracy is:{:.4f}%".format(
        running_loss/len(training_data), 100*running_correct/len(training_data),
        test_loss/len(testing_data),
        100*testing_correct/len(testing_data)
    ))
```
