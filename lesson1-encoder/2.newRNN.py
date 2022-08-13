"""
  * FileName: 2.newRNN
  * Author:   Slatter
  * Date:     2022/8/7 23:38
  * Description: Implement a basic RNN model to classify which country a name belongs to(More Concise)
  * History:
"""
import os
import glob
import string

import unicodedata
import torch
from torch import nn

print("---------------------Set hyperparameters---------------------")
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
Epochs = 5
input_size = 30
hidden_size = 40
learning_rate = 0.05

print("---------------------Preparing the Data---------------------")

# 1.读入数据
letters = []


def unicodeToAscii(s):
    res = str()
    for c in unicodedata.normalize('NFD', s):
        if unicodedata.category(c) != 'Mn':
            res += c
    return res


def readFile(letters, path):
    res = []
    with open(path, 'r', encoding='UTF-8') as input_file:
        lines = input_file.readlines()
        for line in lines:
            name = unicodeToAscii(line.strip())
            res.append(name)
            letters += list(name)
    return res


train_data = []  # ('Cheng', 'Chinese')
countries = []  # 记录所有国家名字

for filepath in glob.glob('data/names/*.txt'):
    country = os.path.splitext(os.path.basename(filepath))[0]
    countries.append(country)
    names = readFile(letters, filepath)
    for name in names:
        train_data.append((name, country))

letters = list(set(letters))


# 2.把数据转换为tensor
def get_nameIdx(name):
    name_idx = torch.tensor([letters.index(c) for c in name]).to(DEVICE)
    return name_idx


print("---------------------Creating the Network---------------------")


class RNNModel(nn.Module):
    def __init__(self, vocab_size, input_size, hidden_size, output_size):
        super(RNNModel, self).__init__()
        self.embedding_layer = nn.Embedding(vocab_size, input_size)
        self.rnncell = nn.RNNCell(input_size=input_size, hidden_size=hidden_size)
        self.fc_layer = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_idx, hidden):
        """
        :param input_idx: 输入名字对应的idx list ’Cheng‘ -> tensor([5, 20, 27, 36, 51])
        """
        embed = self.embedding_layer(input_idx)
        for i in range(embed.size(0)):
            hidden = self.rnncell(embed[i], hidden)

        out = self.sigmoid(self.fc_layer(hidden))
        return out


print("----------------------------Training----------------------------")
model = RNNModel(vocab_size=len(letters), input_size=input_size, hidden_size=hidden_size, output_size=len(countries))
model.to(DEVICE)

criterion = nn.CrossEntropyLoss()


def data_to_tensor(name, country):
    name_idx = get_nameIdx(name)
    country_tensor = torch.tensor(countries.index(country)).to(DEVICE)
    return name_idx, country_tensor


def train_iter(name_tensor, country_tensor):
    hidden = torch.zeros(hidden_size).to(DEVICE)
    model.zero_grad()

    out = model(name_tensor, hidden)
    flag = False
    predict_label = out.argmax()
    if predict_label == country_tensor:
        flag = True

    loss = criterion(out, country_tensor)
    loss.backward()
    for p in model.parameters():
        p.data.add_(p.grad.data, alpha=-learning_rate)
    return loss.item(), flag


def train():
    accuracy_max = 0
    for epoch in range(1, Epochs + 1):
        total_iters = len(train_data)
        right_count = 0
        total_loss = 0
        for i in range(len(train_data)):
            name, country = train_data[i]
            name_tensor, country_tensor = data_to_tensor(name, country)
            loss, flag = train_iter(name_tensor, country_tensor)
            if flag:
                right_count += 1
            total_loss += loss

        accuracy = right_count / total_iters
        avg_loss = total_loss / total_iters
        print("Epoch:{} | avg_loss:{} | accuracy:{}".format(epoch, avg_loss, accuracy))

        if accuracy > accuracy_max:
            accuracy_max = accuracy
            torch.save(model.state_dict(), 'best.pth')

    print("The best accuracy is:{}".format(accuracy_max))


train()
print("----------------------------Evaluating----------------------------")
model.load_state_dict(torch.load('best.pth'))
while True:
    name = input('\nInput a name to predict(enter 0 to quit):')
    if name == '0':
        break
    else:
        name_tensor = get_nameIdx(name)
        hidden = torch.zeros(hidden_size).to(DEVICE)
        out = model(name_tensor, hidden)
        predict = out.argmax().item()

        print('This name might belongs to:', countries[predict])
