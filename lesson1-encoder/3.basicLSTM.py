"""
  * FileName: 3.basicLSTM
  * Author:   Slatter
  * Date:     2022/8/13 14:31
  * Description: Implement a LSTM model to classify which country a name belongs to
  * History:
"""

"""
  * FileName: 1.baseRNN
  * Author:   Slatter
  * Date:     2022/7/31 13:54
  * Description: Implement a basic RNN model to classify which country a name belongs to(More specific)
  * History:
"""
import glob
import os.path

import torch
from torch import nn, optim
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import DataLoader
import unicodedata

print("---------------------Set hyperparameters---------------------")  # 更改：parameters拼写错误已更改
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
embed_size = 20
hidden_size = 40
batch_size = 128
learning_rate = 5e-3
Epochs = 20

print("---------------------Preparing the Data---------------------")
letters = ["<pad>"]
MAX_LEN = 0


# 1.读入数据
def unicodeToAscii(s):
    res = str()
    for c in unicodedata.normalize('NFD', s):
        if unicodedata.category(c) != 'Mn':
            res += c
    return res


def readFile(path):
    res = []
    global letters
    global MAX_LEN
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            convert = unicodeToAscii(line.strip())
            MAX_LEN = max(MAX_LEN, len(convert))
            res.append(convert)
            letters += list(convert)
        return res


# 20000 0.9:0.05:0.05
train_dataset = []
val_dataset = []
test_dataset = []
countries = []

for filepath in glob.glob('data/names/*.txt'):
    country_name = os.path.splitext(os.path.basename(filepath))[0]
    countries.append(country_name)
    names = readFile(filepath)
    length = len(names)
    for i in range(length):
        if i <= 0.9 * length:
            train_dataset.append((names[i], country_name))
        elif i <= 0.95 * length:
            val_dataset.append((names[i], country_name))
        else:
            test_dataset.append((names[i], country_name))

letters = sorted(list(set(letters)))  # 一定要排序固定词汇表，因为set每次的结果都不一样，所以不固定会导致每次运行程序的词汇表都不一样


# 2.构造dataloader
def collate_fn(batch):
    # 128  (name, country)
    names_list, length_list, countries_list = [], [], []
    for name, country in batch:
        # "cheng"
        padded_name = [letters.index(c) for c in name] + [letters.index("<pad>")] * (MAX_LEN - len(name))
        names_list.append(padded_name)
        length_list.append(len(name))
        countries_list.append(countries.index(country))
    names_tensor = torch.tensor(names_list)
    countries_tensor = torch.tensor(countries_list)
    return names_tensor.to(DEVICE), length_list, countries_tensor.to(DEVICE)


train_dataloader = DataLoader(dataset=train_dataset, shuffle=True, batch_size=batch_size, collate_fn=collate_fn)
val_dataloader = DataLoader(dataset=val_dataset, batch_size=batch_size, collate_fn=collate_fn)
test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size, collate_fn=collate_fn)

print("---------------------Creating the Network---------------------")


class BasicLSTM(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, output_size):
        super(BasicLSTM, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_size)
        self.lstm = nn.LSTM(input_size=embed_size, hidden_size=hidden_size, num_layers=1)
        self.fc_layer = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, name, length):
        # name: (batch, MAX_LEN) length: (batch)
        name = name.transpose(0, 1)  # (batch, MAX_LEN) -> (MAX_LEN, batch)
        embed = self.embedding(name)  # (MAX_LEN, batch, embed_size)
        embed = pack_padded_sequence(embed, length, enforce_sorted=False)
        out, (final_hidden, final_cell) = self.lstm(embed)
        result = self.sigmoid(self.fc_layer(final_hidden))  # (1, batch, MAX_LEN)
        return result.squeeze()  # (batch, MAX_LEN)


print("----------------------------Training----------------------------")
model = BasicLSTM(vocab_size=len(letters), embed_size=embed_size, hidden_size=hidden_size,
                  output_size=len(countries)).to(DEVICE)
optimizer = optim.Adam(params=model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()


def train():
    accuracy_max = 0  # 也可以根据loss值去选择最好的模型
    for epoch in range(1, Epochs + 1):
        total_train_iter = len(train_dataset)
        total_train_loss = 0
        # training
        model.train()
        for idx, batch in enumerate(train_dataloader):
            name_tensor, length, country_tensor = batch
            optimizer.zero_grad()
            logits = model(name_tensor, length)
            loss = criterion(logits, country_tensor)
            total_train_loss += loss.item()

            loss.backward()
            optimizer.step()

        train_loss = total_train_loss / total_train_iter
        print("Training on Epoch:{} | train_loss:{}".format(epoch, train_loss))

        # evaluating
        total_val_iter = len(val_dataset)
        total_val_loss = 0
        right_count = 0
        model.eval()
        with torch.no_grad():
            for idx, batch in enumerate(val_dataloader):
                name_tensor, length, country_tensor = batch
                logits = model(name_tensor, length)
                loss = criterion(logits, country_tensor)
                total_val_loss += loss.item()

                prediction = logits.argmax(dim=-1)
                right_count += (prediction == country_tensor).sum().item()

        accuracy = right_count / total_val_iter
        val_loss = total_val_loss / total_val_iter
        print("Evaluating on Epoch:{} | val_loss:{} | accuracy:{}".format(epoch, val_loss, accuracy))

        if accuracy > accuracy_max:
            accuracy_max = accuracy
            torch.save(model.state_dict(), 'best.pth')

    print("The best accuracy is:", accuracy_max)


train()
print("----------------------------Evaluating----------------------------")
model.load_state_dict(torch.load("best.pth"))


def test():
    # evaluating on test dataset
    total_test_iter = len(test_dataset)
    total_test_loss = 0
    right_count = 0
    model.eval()
    with torch.no_grad():
        for idx, batch in enumerate(test_dataloader):
            name_tensor, length, country_tensor = batch
            logits = model(name_tensor, length)
            loss = criterion(logits, country_tensor)
            total_test_loss += loss.item()

            prediction = logits.argmax(dim=-1)
            right_count += (prediction == country_tensor).sum().item()

    accuracy = right_count / total_test_iter
    test_loss = total_test_loss / total_test_iter
    print("Evaluating on Test dataset: test_loss:{} | accuracy:{}".format(test_loss, accuracy))

test()