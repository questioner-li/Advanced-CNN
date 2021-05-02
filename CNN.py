import torch
import torch.nn as nn
from torchvision import transforms  # 图片处理
from torchvision import datasets  # 加载数据时使用
from torch.utils.data import DataLoader  # 加载数据集时使用
import torch.nn.functional as F  # 使用relu函数中使用
import torch.optim as optim  # 优化器使用
import matplotlib.pyplot as plt  # 作图工具

# prepare dataset

batch_size = 64
transform = transforms.Compose([transforms.ToTensor(),  # 将读进来的图片转变成Pytorch使用的张量，包括像素的映射和通道的改变
                                transforms.Normalize((0.1307,), (0.3081,))])  # 归一化,均值和方差，方便进行神经网络计算

# 利用datasets中的MNIST类来构造mnist实例，用于训练，root表示的是路径，后面表示是训练集，下载，
train_dataset = datasets.MNIST(root='./mnist/',  # 输入数据集的路径
                               train=True,  # 用于神经网络的计算
                               download=True,  # 若当前路径没有数据集MNIST,就进行下载，有的话就可以不用下载了
                               transform=transform)  # 表示按照transform的设置将图片读进来
train_loader = DataLoader(train_dataset,  # 加载mini-batch所使用的数据集
                          shuffle=True,  # 将数据排序打乱再进行loader，增加训练的随机性
                          batch_size=batch_size)  # 设置每一个mini-batch的大小为batach_size
test_dataset = datasets.MNIST(root='./mnist/', train=False, download=True, transform=transform)  # 输入测试集
test_loader = DataLoader(test_dataset, shuffle=False, batch_size=batch_size)  # 加载mini-batch使用的数据集


# 构造InceptionA子类
class InceptionA(nn.Module):
    def __init__(self, in_channels):
        super(InceptionA, self).__init__()
        self.branch1x1 = nn.Conv2d(in_channels, 16, kernel_size=1)

        self.branch5x5_1 = nn.Conv2d(in_channels, 16, kernel_size=1)
        self.branch5x5_2 = nn.Conv2d(16, 24, kernel_size=5, padding=2)  # padding=2是因为kernel=5,要保证数据高宽不变

        self.branch3x3_1 = nn.Conv2d(in_channels, 16, kernel_size=1)
        self.branch3x3_2 = nn.Conv2d(16, 24, kernel_size=3, padding=1)
        self.branch3x3_3 = nn.Conv2d(24, 24, kernel_size=3, padding=1)

        self.branch_pool = nn.Conv2d(in_channels, 24, kernel_size=1)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)  # 路径1

        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)  # 路径2

        branch3x3 = self.branch3x3_1(x)
        branch3x3 = self.branch3x3_2(branch3x3)
        branch3x3 = self.branch3x3_3(branch3x3)  # 路径3

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)  # 均值迟化，所有参数可以保证我们的输入层和输出层高宽不变
        branch_pool = self.branch_pool(branch_pool)  # 路径4

        outputs = [branch1x1, branch5x5, branch3x3, branch_pool]
        return torch.cat(outputs, dim=1)  # b,c,w,h  c对应的是dim=1 也就是通道的排列方向


# 构造residual子类
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()  # 继承nn.Module类，进行初始化
        self.channels = channels  # 设置通道的数目
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)  # 利用卷积方法构造通道数相同，大小不变的输出层
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)  # 利用卷积方法构造通道数相同，大小不变的输出层

    def forward(self, x):
        y = F.relu(self.conv1(x))  # 先进行第一层的卷积，再进行激活
        y = self.conv2(y)  # 进行第二层的卷积，先不激活
        return F.relu(x + y)  # 将第二层的输出层和输入层相加，然后再进行relu的非线性激活


# 构造模型Net类
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5)  # 输入层1通道，输出层16通道，卷积核为1*5*5，卷积核数为16
        self.conv2 = nn.Conv2d(88, 32, kernel_size=5)  # 88 = 24x3 + 16

        self.rblock1 = ResidualBlock(16)
        self.rblock2 = ResidualBlock(32)

        self.incep1 = InceptionA(in_channels=16)
        self.incep2 = InceptionA(in_channels=32)

        self.mp = nn.MaxPool2d(2)
        self.linear = nn.Linear(8800, 10)  # 暂时不知道1408咋能自动出来的

    def forward(self, x):
        in_size = x.size(0)  # 我们把batchsize给求出来，拿出它的维度出来

        x = F.relu(self.conv1(x))  # 先做卷积，在做relu，最后做迟化
        x = self.rblock1(x)  # 进行依次
        x = self.incep1(x)
        x = self.mp(F.relu(self.conv2(x)))
        x = self.rblock2(x)
        x = self.incep2(x)

        x = x.view(in_size, -1)  # 将数据转化成全连接层需要的输入
        x = self.linear(x)  # 然后计算它最后的全连接层，不做激活
        return x

model = Net()  # 模型实例化
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 将模型传输到GPU中进行计算
model.to(device)  # to方法的功能是把模型和参数都放在GPU上


# 构造误差和优化器
criterion = torch.nn.CrossEntropyLoss()  # 误差计算使用交叉熵损失
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)  # 使用带冲量的优化器，学习步长设置为0.01


def train(epoch):  # 为了train cycle代码简介，这样将一个样本依次的训练进行函数的封装
    running_loss = 0.0  # 函数损失初始化
    for batch_idx, data in enumerate(train_loader, 0):  # 使用mini-batch的方式进行循环训练
        # 1. Prepare data
        inputs, target = data  # inputs作为样本， target作为标签
        inputs, target = inputs.to(device), target.to(device)  # 将数据集传送到GPU中进行计算
        optimizer.zero_grad()  # 清空梯度
        # 2. Forward
        outputs = model(inputs)  # 计算预测输出值
        loss = criterion(outputs, target)  # 计算误差
        # 3. Backward
        loss.backward()  # 误差反向传播
        # 4. Update
        optimizer.step()  # 优化器进行更新

        running_loss += loss.item()  # 将所有的误差进行相加，利用item()取出数值
        if batch_idx % 300 == 299:  # 每300次进行依次误差的输出
            print('[%d, %5d] loss: %.3f' % (epoch + 1, batch_idx + 1, running_loss / 300))
            running_loss = 0.0  # 每300次之后，清零后累计下一个300次的误差


def test():
    correct = 0
    total = 0
    with torch.no_grad():  # 不需要进行反向传播，故不用计算梯度
        for data in test_loader:  # 直接从test_loader中拿出数据
            images, labels = data  # 每一个样本放入image, label
            images, labels = images.to(device), labels.to(device)  # 将测试集导入GPU中进行计算
            outputs = model(images)  # 预测进行输出值的计算
            _, predicted = torch.max(outputs.data, dim=1)  # 取每一行中最大的下标出来，拿出每个样本中最大的下标出来，行是1
            total += labels.size(0)  # 将所有的样本进行累计，计算出所有的样本出来
            correct += (predicted == labels).sum().item()  # 预测与实际值相同，表示正确，进行求和
    print('accuracy on test set: %.4f %% ' % (100 * correct / total))
    return correct / total  # 返回正确率


if __name__ == '__main__':  # 将所有train cycle封装起来，避免因为windows建立进程swap导致错误
    epoch_list = []  # 建立epoch_list进行训练周期次数收集
    acc_list = []  # 建立acc_list进行正确率数据的收集

    for epoch in range(10):  # 循环周期设置为10
        train(epoch)  # 训练
        acc = test()  # 测试
        epoch_list.append(epoch)  # 收集训练周期的次数
        acc_list.append(acc)  # 收集第i次训练后模型预测测试集的正确性

    plt.plot(epoch_list, acc_list)  # 绘制误差图表
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.grid()  # 打开网格
    plt.show()  # 显示图片
