"""
  * FileName: 1.baseRNN
  * Author:   Slatter
  * Date:     2022/7/31 13:54
  * Description: Implement a basic RNN model to classify which country a name belongs to(More specific)
  * History:
"""
import glob
import os.path
import string
import random

import torch
from torch import nn
import unicodedata

print("---------------------Set hyperpameters---------------------")
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
hidden_size = 64
learning_rate = 0.5
epochs = 20
iters = 3000

print("---------------------Preparing the Data---------------------")


# 1. 加载数据以及将数据格式调整一致
def unicodeToAscii(s):
    # Félix -> Felix   é = e + \u..
    res = str()
    for c in unicodedata.normalize('NFD', s):
        if unicodedata.category(c) != 'Mn':
            res += c
    return res


def readFile(filepath):
    with open(filepath, 'r', encoding='UTF-8') as input_file:
        lines = input_file.readlines()  # 'Cheng\n'
        res = [unicodeToAscii(line.strip()) for line in lines]
        return res


category_names = {}  # {‘Chinese’: [...]}
categorys = []

for filepath in glob.glob('data/names/*.txt'):
    category = os.path.splitext(os.path.basename(filepath))[0]
    categorys.append(category)
    names = readFile(filepath)
    category_names[category] = names

# print(len(category_names['Chinese']))

# 2. 把字符数据转换为Tensor
letters = string.ascii_letters + " .,;"  # 30  'a'  [1 0 0 0 ...]
total_letters = len(letters)


def namesToTensor(names):
    # 把一个名字转换为tensor
    res = torch.zeros(len(names), total_letters)
    for idx, letter in enumerate(names):
        res[idx][letters.find(letter)] = 1
    return res.to(DEVICE)


# print(namesToTensor('Cheng'))

print("---------------------Creating the Network---------------------")


class RNNBlock(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNNBlock, self).__init__()
        self.hidden_size = hidden_size
        self.i2h = nn.Linear(input_size, hidden_size)
        self.h2h = nn.Linear(hidden_size, hidden_size)
        self.sigmoid = nn.Sigmoid()

        self.h2o = nn.Linear(hidden_size, output_size)

    def forward(self, input, hidden):
        ih = self.i2h(input)
        hh = self.h2h(hidden)
        hidden = self.sigmoid(ih + hh)
        out = self.h2o(hidden)
        return out, hidden


print("----------------------------Training----------------------------")


def randomChoose(l):
    return l[random.randint(0, len(l) - 1)]


def randomTrainExample():
    category = randomChoose(categorys)  # 随机选择一个类别
    name = randomChoose(category_names[category])  # 随机选取该类别中的一个名字
    category_tensor = torch.tensor(categorys.index(category))
    nameTensor = namesToTensor(name)
    return category, name, category_tensor.to(DEVICE), nameTensor


# for i in range(5):
#     category, name, category_tensor, name_tensor = randomTrainExample()
#     print(category, name, category_tensor, name_tensor)


model = RNNBlock(input_size=total_letters, hidden_size=hidden_size, output_size=len(categorys))
model.to(DEVICE)

criterion = nn.CrossEntropyLoss()


# 一次train_iter()走完整个名字
def train_iter(category, name_tensor):
    hidden = torch.zeros(hidden_size).to(DEVICE)
    model.zero_grad()

    for i in range(name_tensor.size(0)):
        out, hidden = model(name_tensor[i], hidden)

    flag = False
    predict_label = out.argmax().item()
    truth_label = category.item()
    if predict_label == truth_label:
        flag = True

    loss = criterion(out, category)
    loss.backward()

    for p in model.parameters():
        p.data.add_(p.grad.data, alpha=-learning_rate)
    return loss.item(), flag


def train(epochs):
    for epoch in range(1, epochs + 1):
        right_count = 0
        total_count = iters
        total_loss = 0
        accuracy = 0
        for iter in range(iters):
            category, name, category_tensor, name_tensor = randomTrainExample()
            loss, flag = train_iter(category_tensor, name_tensor)
            total_loss += loss
            if flag == True:
                right_count += 1

        cur_accuracy = right_count / total_count
        total_loss /= total_count
        print("Epoch={} | loss={} | total_count={} | accuracy={}".format(epoch, total_loss, total_count, cur_accuracy))

        if cur_accuracy > accuracy:
            accuracy = cur_accuracy
            torch.save(model.state_dict(), 'best.pth')  # 更改：保存网络中的参数

    print('The best accuracy is:{}'.format(accuracy))   # 更改：accurage 改为了 accuracy


train(epochs)  # 更改：这里只是提醒一下如果训练好一轮不想再训练一次，注释掉改行直接运行即可
print("----------------------------Evaluating----------------------------")
model.load_state_dict(torch.load('best.pth'))  # 更改：g加载模型参数
while True:
    name = input("\nInput a name to predict(enter 0 to quit):")
    if name == '0':
        break
    else:
        name_tensor = namesToTensor(name)

        hidden = torch.zeros(hidden_size).to(DEVICE)
        for i in range(name_tensor.size(0)):
            out, hidden = model(name_tensor[i], hidden)
        predict = out.argmax().item()

        print("This name might belongs to:", categorys[predict])  # 更改：把category改为了categories
