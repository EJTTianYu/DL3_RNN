# coding: utf-8
import argparse
import time
import math
import torch
import torch.nn as nn
import src.data as data
from src.model import LMModel
import os
import os.path as osp
import torch.optim as optim

parser = argparse.ArgumentParser(description='PyTorch ptb Language Model')
parser.add_argument('--epochs', type=int, default=40,
                    help='upper epoch limit')
parser.add_argument('--train_batch_size', type=int, default=20, metavar='N',
                    help='batch size')
parser.add_argument('--eval_batch_size', type=int, default=20, metavar='N',
                    help='eval batch size')
parser.add_argument('--max_sql', type=int, default=35,
                    help='sequence length')
parser.add_argument('--seed', type=int, default=1234,
                    help='set random seed')
parser.add_argument('--cuda', action='store_true', help='use CUDA device')
parser.add_argument('--gpu_id', type=int, help='GPU device id used')

args = parser.parse_args()

# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)

# Use gpu or cpu to train
use_gpu = False

if use_gpu:
    # torch.cuda.set_device(args.gpu_id)
    # device = torch.device(args.gpu_id)
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")

# load data
train_batch_size = args.train_batch_size
eval_batch_size = args.eval_batch_size
batch_size = {'train': train_batch_size, 'valid': eval_batch_size}
data_loader = data.Corpus("../data/ptb", batch_size, args.max_sql)
# 获取语料库的总词数
total_voc = len(data_loader.vocabulary)
# 用于指定每个单词的embedding维数
word_vector_num = 10000
# 用于指定RNN的hidden state的数量
hidden_num = 30
# 用于指定RNN的层数
layer_num = 2
# 用于指定学习率
LEARNING_RATE = 0.001
# 用于指定输出的view压缩
# target_size = args.max_sql * train_batch_size
# print(total_voc)

# WRITE CODE HERE within two '#' bar
########################################
# Build LMModel model (bulid your language model here)

########################################
rnn = LMModel(total_voc, word_vector_num, hidden_num, layer_num)
rnn = rnn.to(device)
criterion = nn.CrossEntropyLoss()
# optimizer_ft = optim.Adam(rnn.parameters(), lr=0.001)
optimizer = optim.SGD(rnn.parameters(), lr=LEARNING_RATE, momentum=0.9)


# WRITE CODE HERE within two '#' bar
########################################
# Evaluation Function
# Calculate the average cross-entropy loss between the prediction and the ground truth word.
# And then exp(average cross-entropy loss) is perplexity.

def evaluate():
    # valid模式
    rnn.eval()
    data_loader.set_valid()
    running_loss_test = 0.0
    running_corrects_test = 0
    count_test = 0
    while True:
        input, target, end_flag = data_loader.get_batch()
        input = input.to(device)
        target = target.to(device)
        # optimizer.zero_grad()
        output, hidden = rnn(input)
        _, preds = torch.max(output, 2)
        loss = criterion(output.view(target.size(0), -1), target)
        running_loss_test += loss.item()
        running_corrects_test += torch.sum(preds.view(-1) == target.data)
        # loss.backward()
        # optimizer.step()
        count_test += 1
        if end_flag:
            break
    epoch_loss_test = running_loss_test / count_test
    # epoch_acc_test = running_corrects_test.double() / count_test
    epoch_pp = math.exp(epoch_loss_test)
    print('Valid Loss: {:.4f} Perplexity: {:.4f}'.format(epoch_loss_test, epoch_pp))


########################################


# WRITE CODE HERE within two '#' bar
########################################
# Train Function
def train():
    # 先train再valid
    rnn.train()
    data_loader.set_train()
    running_loss_train = 0.0
    running_corrects_train = 0
    count = 0
    while True:
        input, target, end_flag = data_loader.get_batch()
        input = input.to(device)
        target = target.to(device)
        optimizer.zero_grad()
        output, hidden = rnn(input)
        _, preds = torch.max(output, 2)
        loss = criterion(output.view(target.size(0), -1), target)
        loss.backward()
        optimizer.step()  # statistics
        running_loss_train += loss.item()
        running_corrects_train += torch.sum(preds.view(-1) == target.data)
        count += 1
        if end_flag:
            break
    epoch_loss_train = running_loss_train / count
    # epoch_acc_train = running_corrects_train.double() / count
    epoch_pp = math.exp(epoch_loss_train)
    print('Train Loss: {:.4f} Perplexity: {:.4f}'.format(epoch_loss_train, epoch_pp))


########################################


# Loop over epochs.
for epoch in range(1, args.epochs + 1):
    train()
    evaluate()
