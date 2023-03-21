import torch
from torch import nn
import pandas as pd
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.feature_extraction.text import CountVectorizer
from tqdm import tqdm
import random


def get_device():
    """ 设备 """
    return 'cuda' if torch.cuda.is_available() else 'cpu'


device = get_device()


class Sequence(Dataset):
    """ 文本预处理 """

    def __init__(self, data, max_seq_len):
        # 句子的最长长度 少于: 用<pad>填补 多余: 截断
        self.max_seq_len = max_seq_len
        df = data

        # BOW 以后换Bert
        vectorizer = CountVectorizer(stop_words="english", min_df=0.015)
        vectorizer.fit(df.review.tolist())

        # 创建词汇表
        self.token2idx = vectorizer.vocabulary_
        self.token2idx['<pad>'] = max(self.token2idx.values()) + 1

        # 词元化工具
        tokenizer = vectorizer.build_analyzer()

        # 词元转换为整数索引
        def encode(seq):
            return [self.token2idx[token] for token in tokenizer(seq) if token in self.token2idx]
        self.encode = encode
        # 将少于max_seq_len的地方用特殊符号 <pad> 补齐
        def pad(token):
            return token + (max_seq_len - len(token)) * [self.token2idx['<pad>']]
        self.pad = pad
        sequences = [encode(sequence)[:max_seq_len] for sequence in df.review.tolist()]
        sequences, self.labels = zip(
            *[(sequence, label) for sequence, label in zip(sequences, df.label.tolist()) if sequence])

        # 填补
        self.sequences = [pad(sequence) for sequence in sequences]

    def __getitem__(self, index):
        assert len(self.sequences[index]) == self.max_seq_len
        return self.sequences[index], self.labels[index]

    def __len__(self):
        return len(self.sequences)


data = pd.read_csv("./IMDB Dataset.csv")
data['label'] = data['sentiment']
del data['sentiment']

labeling = {
    'positive': 1,
    'negative': 0
}

data['label'] = data['label'].apply(lambda x: labeling[x])

# 实例化
dataset = Sequence(data, max_seq_len=128)


def collate(batch):
    inputs = torch.LongTensor([item[0] for item in batch])
    target = torch.FloatTensor([item[1] for item in batch])
    return inputs, target


batch_size = 2048
train_loader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate, num_workers=0)


class RNN(nn.Module):
    """ LSTM """

    def __init__(self, vocab_size, batch_size, embedding_size=100, hidden_size=128, n_layers=1, device='cpu'):
        super(RNN, self).__init__()
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.device = device
        self.encoder = nn.Embedding(vocab_size, embedding_size)
        self.rnn = nn.LSTM(embedding_size, hidden_size, num_layers=n_layers, batch_first=True)
        self.decoder = nn.Linear(hidden_size, 1)

    def init_hidden(self):
        """ 初始化隐藏状态 hidden-state cell """
        return (torch.randn(self.n_layers, self.batch_size, self.hidden_size).to(self.device),
                torch.randn(self.n_layers, self.batch_size, self.hidden_size).to(self.device))

    def forward(self, inputs):
        """ 传播 """
        batch_size = inputs.size(0)
        if batch_size != self.batch_size:
            self.batch_size = batch_size
        X = self.encoder(inputs)
        output, state = self.rnn(X, self.init_hidden())
        # 取最后一个输出
        output = self.decoder(output[:, :, -1]).squeeze()
        return output


model = RNN(
    vocab_size=len(dataset.token2idx),
    batch_size=batch_size,
    hidden_size=128,
    device=device
)
model = model.to(device)

# 超参数
lr = 1e-4 # lr = 0.001(V0)
num_epochs = 50
wd = 3e-3

criterion = nn.BCEWithLogitsLoss()

optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)

train_losses = []
model_path = "D:\\AI_Study\\Kaggle\\Sentiment Analysis\\models\\sentiment_analysis_v1.pth"
best_loss = 1000.0

if __name__ == '__main__':
    # 训练
    '''for epoch in range(num_epochs):
        model.train()
        progress_bar = tqdm(train_loader)
        losses = []
        total = 0
        for inputs, target in progress_bar:
            inputs, target = inputs.to(device), target.to(device)
            # 梯度清零 因为pytorch会保留上一次计算的梯度
            model.zero_grad()
            output = model(inputs)
            # 损失
            loss = criterion(output, target)
            # 计算图反向传播
            loss.backward()
            # 梯度裁剪 以防梯度爆炸等问题
            nn.utils.clip_grad_norm_(model.parameters(), 3)
            # 更新参数
            optimizer.step()
            # 保存损失
            losses.append(loss)
            # 进度条显示loss
            progress_bar.set_description(f'Loss {loss.item():.3f}')
            # 迭代次数 +1
            total += 1

        # 每个epoch的损失
        epoch_loss = sum(losses) / total
        train_losses.append(epoch_loss)
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(model.state_dict(), model_path)
            print('saving model with loss {:.3f} in epoch {}'.format(best_loss, epoch + 1))

        # 进度条显示迭代信息
        tqdm.write(f'Epoch #{epoch + 1}\tTrain Loss: {epoch_loss:.3f}')
    print('Done!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n')'''

    # 预测
    def predict_sentiment(text):
        model.load_state_dict(torch.load(model_path))
        model.eval()
        with torch.no_grad():
            test_vector = torch.LongTensor([dataset.pad(dataset.encode(text))]).to(device)

            output = model(test_vector)
            prediction = torch.sigmoid(output).item()

            # print_random(prediction)
            if prediction > 0.5:
                print(f'{prediction:0.3}: Positive sentiment')
            else:
                print(f'{prediction:0.3}: Negative sentiment')


    def print_random(prediction):
        p_answer = [
            "That sounds good!",
            "You look happy!",
            "Very nice!",
            "This is good!"
        ]
        n_answer = [
            "That's too bad!",
            "Kind of bad.",
            "Very bad.",
            "This is bad."
        ]
        rdm_index_p = random.randint(0, len(p_answer) - 1)
        rdm_index_n = random.randint(0, len(n_answer) - 1)
        if prediction > 0.5:
            # print(f'\n\nme: {text}\n')
            print(f"sam: {p_answer[rdm_index_p]}")
            print("\n")
        else:
            # print(f'\n\nme: {text}\n')
            print(f"sam: {n_answer[rdm_index_n]}")
            print("\n")


#输入
text = "I'm too full of life to be half-loved."
predict_sentiment(text)
while True:
    text = str(input("me: "))
    if text == "exit()":
        raise KeyboardInterrupt
    predict_sentiment(text)