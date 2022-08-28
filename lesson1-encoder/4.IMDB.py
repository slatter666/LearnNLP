"""
  * FileName: 4.IMDB.py
  * Author:   Slatter
  * Date:     2022/8/28 20:00
  * Description: Implement a model based on average of word vector to classify IMDB dataset
  * History:
"""
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchtext.datasets import IMDB
from torchtext.data.functional import to_map_style_dataset
from typing import List

print("-----------------------Hyperparameters-----------------------")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
embed_size = 300
batch_size = 128
Epochs = 10
learning_rate = 5

print("-----------------------Prepare data-----------------------")


def tokenize(text):
    return text.lower().split()


def build_vocab(data):
    vocab = []
    for label, text in data:
        vocab += tokenize(text)
    vocab.append("<unk>")
    vocab = sorted(list(set(vocab)))
    return vocab


def token_id_convert(vocab: List[str]):
    token2id, id2token = {}, {}
    for i in range(len(vocab)):
        token2id[vocab[i]] = i
        id2token[i] = vocab[i]
    return token2id, id2token


train = to_map_style_dataset(IMDB(split="train"))
test = to_map_style_dataset(IMDB(split="test"))

vocab = build_vocab(train)
token2id, id2token = token_id_convert(vocab)


def label_pipeline(label):
    if label == 'neg':
        return 0
    else:
        return 1


def text_pipeline(text: str):
    res = []
    tokens = tokenize(text)
    for i in range(len(tokens)):
        res.append(token2id.get(tokens[i], token2id["<unk>"]))
    assert len(res) == len(tokens), "生成的id数和token数不一致"
    return res


def collate_batch(batch):
    label_list, text_list, offsets = [], [], [0]
    for label, text in batch:
        label_list.append(label_pipeline(label))
        processed_text = torch.tensor(text_pipeline(text), dtype=torch.int64)
        text_list.append(processed_text)
        offsets.append(processed_text.size(0))
    label_list = torch.tensor(label_list, dtype=torch.int64)
    offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
    text_list = torch.cat(text_list)
    return label_list.to(DEVICE), text_list.to(DEVICE), offsets.to(DEVICE)


train_dataset = train[:int(len(train) * 0.9)]
val_dataset = train[int(len(train) * 0.9):]
test_dataset = test

train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, collate_fn=collate_batch)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=collate_batch)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=collate_batch)

print("-----------------------Build model-----------------------")


class IMDBClassifier(nn.Module):
    def __init__(self, vocab_size, embed_size, num_class):
        super(IMDBClassifier, self).__init__()
        self.embedding = nn.EmbeddingBag(num_embeddings=vocab_size, embedding_dim=embed_size, mode="mean", sparse=True)
        self.fc = nn.Linear(embed_size, num_class)

    def forward(self, text, offset):
        embed = self.embedding(text, offset)
        return self.fc(embed)


print("-----------------------Train the model-----------------------")
model = IMDBClassifier(vocab_size=len(vocab), embed_size=embed_size, num_class=2).to(DEVICE)
optimizer = optim.SGD(params=model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()


def train():
    accuracy_max = 0  # 也可以根据loss值去选择最好的模型
    for epoch in range(1, Epochs + 1):
        total_train_iter = len(train_dataset)
        total_train_loss = 0
        # training
        model.train()
        for idx, batch in enumerate(train_dataloader):
            optimizer.zero_grad()
            label, text, offset = batch
            logits = model(text, offset)
            loss = criterion(logits, label)
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
                label, text, offset = batch
                logits = model(text, offset)
                loss = criterion(logits, label)
                total_val_loss += loss.item()

                prediction = logits.argmax(dim=-1)
                right_count += (prediction == label).sum().item()

        accuracy = right_count / total_val_iter
        val_loss = total_val_loss / total_val_iter
        print("Evaluating on Epoch:{} | val_loss:{} | accuracy:{}".format(epoch, val_loss, accuracy))

        if accuracy > accuracy_max:
            accuracy_max = accuracy
            torch.save(model.state_dict(), 'best.pth')

    print("The best accuracy is:", accuracy_max)

train()
print("-----------------------Evaluate-----------------------")
model.load_state_dict(torch.load("best.pth"))


def test():
    # evaluating on test dataset
    total_test_iter = len(test_dataset)
    total_test_loss = 0
    right_count = 0
    model.eval()
    with torch.no_grad():
        for idx, batch in enumerate(test_dataloader):
            label, text, offset = batch
            logits = model(text, offset)
            loss = criterion(logits, label)
            total_test_loss += loss.item()

            prediction = logits.argmax(dim=-1)
            right_count += (prediction == label).sum().item()

    accuracy = right_count / total_test_iter
    test_loss = total_test_loss / total_test_iter
    print("Evaluating on Test dataset: test_loss:{} | accuracy:{}".format(test_loss, accuracy))


test()
