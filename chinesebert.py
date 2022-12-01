import numpy as np
import random
import torch
import pkuseg
import math
import random
from tqdm import tqdm,trange
import pandas as pd
import matplotlib.pylab as plt
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from transformers import get_linear_schedule_with_warmup
try:
    from torch.utils.tensorboard import SummaryWriter
except:
    from tensorboardX import SummaryWriter
from torch.utils.data.distributed import DistributedSampler
import argparse
import glob
import logging
import os
import random
from sklearn.metrics import f1_score


torch.cuda.set_device(-1)
SEED = 123
BATCH_SIZE = 4
learning_rate = 2e-5
weight_decay = 1e-2
epsilon = 1e-8

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
logger = logging.getLogger(__name__)

def readFile(filename):
    with open(filename, encoding='utf-8') as f:
        content = f.readlines()
        return content

def read_corpus(file_path):
    """读取语料
    :param file_path:
    :param type:
    :return:
    """
    src_data = []
    labels = []
    seg = pkuseg.pkuseg()
    fout = pd.read_excel(file_path,usecols=[10, 11])
    # dataframe转化成list
    fout = fout.values.tolist()
    # 去掉第一行
    fout = fout[1:]
    i=1
    lmax = 0
    for line in tqdm(fout,desc='precessing data'):
        pair = line[0]
        label = line[1] #+ 1
        lp = len(pair)
        # if (type(pair) != str) or ((type(label) != int) and (type(label) != float)):
        #     print(i)
        #     print(type(pair), type(label),label)
        #     print(pair)
            # i += 1
        #     continue
        # else:
        #     # pair = pair.strip(u'\u200b').split('\t')
        #     # src_data.append(seg.cut(pair[0]))
        #     # if label == 4: label = 1
        #     # else: label = 0
        #     src_data.append(pair)
        #     labels.append(label)
        #     if lp > lmax: lmax = lp
        try:
            labels.append(int(label))
            src_data.append(pair)
            i += 1
        except:
            i += 1
            print(i,pair,label)
            print(type(label))
            continue
    print(lmax)
    # print(labels)
    return (src_data, labels)


train = read_corpus('./1w_process.xlsx')
# for i in range(400, len(train)):
#     if 0 <= train[i][1] and train[i][1] <= 5: continue
#     else: print(i)
sentences = train[:][0]
targets = train[:][1]
total_targets = torch.tensor(targets)
total_targets = total_targets.view(9993, 1)

# 设定标签

# targets = np.concatenate((pos_targets, neg_targets), axis=0).reshape(-1, 1)  # (10000, 1)
# total_targets = torch.tensor(targets)

model_name = './bert-base-chinese'
cache_dir = './sample_data/'

tokenizer = BertTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
# print(len(sentences[2]), sentences[2])
# print(tokenizer.tokenize(sentences[2]))
# print(tokenizer.encode(sentences[2]))
# print(tokenizer.convert_ids_to_tokens(tokenizer.encode(sentences[2])))


# 将每一句转成数字 （大于510做截断，小于510做 Padding，加上首位两个标识，长度总共等于512）
def convert_text_to_token(tokenizer, sentence, limit_size=510):
    tokens = tokenizer.encode(sentence[:limit_size])  # 直接截断
    if len(tokens) < limit_size + 2:  # 补齐（pad的索引号就是0）
        tokens.extend([0] * (limit_size + 2 - len(tokens)))
    return tokens


input_ids = [convert_text_to_token(tokenizer, sen) for sen in sentences]

input_tokens = torch.tensor(input_ids)
# print(input_tokens.shape)  # torch.Size([5197, 1000])

# 建立mask
def attention_masks(input_ids):
    atten_masks = []
    for seq in input_ids:  # [10000, 128]
        seq_mask = [float(i > 0) for i in seq]  # PAD: 0; 否则: 1
        atten_masks.append(seq_mask)
    return atten_masks


atten_masks = attention_masks(input_ids)
attention_tokens = torch.tensor(atten_masks)
# print(attention_tokens.shape)  # torch.Size([5197, 1032])

from sklearn.model_selection import train_test_split

train_inputs, test_inputs, train_labels, test_labels = train_test_split(input_tokens, total_targets,
                                                                        random_state=666, test_size=0.1)#训练集 验证集 比例
train_masks, test_masks, _, _ = train_test_split(attention_tokens, input_tokens,
                                                 random_state=666, test_size=0.1)
# print(train_inputs.shape, test_inputs.shape)  # torch.Size([4156, 1032]) torch.Size([1039, 1032])
# print(train_masks.shape)  # torch.Size([4156, 1032])和train_inputs形状一样
#
# print(train_inputs[0]) #tensor([ 101, 2792,  809,  ...,    0,    0,    0])
# print(train_masks[0]) #tensor([1., 1., 1.,  ..., 0., 0., 0.])

train_data = TensorDataset(train_inputs, train_masks, train_labels)
# train_data = TensorDataset(train_inputs, train_labels)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, batch_size=BATCH_SIZE)

test_data = TensorDataset(test_inputs, test_masks, test_labels)
# test_data = TensorDataset(test_inputs, test_labels)
test_sampler = RandomSampler(test_data)
test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=BATCH_SIZE)

# for i, (train, mask, label) in enumerate(train_dataloader):
#     # torch.Size([16, 1032]) torch.Size([16, 1032]) torch.Size([16, 1])
#     print(train.shape, mask.shape, label.shape)
#     break

# print('len(train_dataloader) = ', len(train_dataloader))  # 276

model = BertForSequenceClassification.from_pretrained(model_name, num_labels=6) # num_labels表示2个分类,好评和差评 标签分类总数
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

optimizer = AdamW(model.parameters(), lr = learning_rate, eps = epsilon)

no_decay = ['bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
     'weight_decay': weight_decay},
    {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
     'weight_decay': 0.0}
]

optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate, eps=epsilon)

epochs = 5
# training steps 的数量: [number of batches] x [number of epochs].
total_steps = len(train_dataloader) * epochs

# 设计 learning rate scheduler.
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

def binary_acc(preds, labels):
    correct = torch.eq(torch.max(preds, dim=1)[1], labels.flatten()).float() # eq里面的两个参数的shape=torch.Size([16])
    acc = correct.sum().item() / len(correct)
    return acc
import time
import datetime
def format_time(elapsed):
    elapsed_rounded = int(round((elapsed)))
    return str(datetime.timedelta(seconds=elapsed_rounded))   #返回 hh:mm:ss 形式的时间

def train(model, optimizer):#定义训练
    t0 = time.time()
    avg_loss, avg_acc = [],[]

    model.train()
    for step, batch in enumerate(train_dataloader):
        optimizer.zero_grad()
        # 每隔40个batch 输出一下所用时间.
        if step % 40 == 0 and not step == 0:#40个batch输出一下时间 batch分批处理 epoch训练所有数据
            elapsed = format_time(time.time() - t0)
            print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))

        b_input_ids, b_input_mask, b_labels = batch[0].long().to(device), batch[1].long().to(device), batch[2].long().to(device)

        output = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
        loss, logits = output[0], output[1]
        # print(output)

        avg_loss.append(loss.item())

        acc = binary_acc(logits, b_labels)
        avg_acc.append(acc)
        loss.backward()

        clip_grad_norm_(model.parameters(), 1.0)      #大于1的梯度将其设为1.0, 以防梯度爆炸
        optimizer.step()              #更新模型参数
        scheduler.step()              #更新learning rate

    avg_acc = np.array(avg_acc).mean()
    avg_loss = np.array(avg_loss).mean()
    return avg_loss, avg_acc


from sklearn.metrics import classification_report
def evaluate(model):
    avg_acc, pre, labels = [],[], []
    model.eval()         #表示进入测试模式

    with torch.no_grad():
        for batch in test_dataloader:
            b_input_ids, b_input_mask, b_labels = batch[0].long().to(device), batch[1].long().to(device), batch[2].long().to(device)

            output = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)
            acc = binary_acc(output[0], b_labels)
            avg_acc.append(acc)
            # print(type(output[0].cpu().numpy()),type(b_labels.cpu().numpy()))
            # print(output[0].cpu().numpy(), b_labels.cpu().numpy())
            # preans = output[0].cpu().numpy()
            # preans = np.argmax(preans,axis=1)
            preans = torch.max(output[0], dim=1)[1]
            pre.extend(preans.cpu().numpy())
            labels.extend(b_labels.cpu().numpy())
    avg_acc = np.array(avg_acc).mean()
    f1_weighted = f1_score(labels, pre, average='weighted')
    target_names = ["0","1"]#需要修改
    f_all = classification_report(labels, pre, target_names=target_names)
    return avg_acc, f1_weighted, f_all


for epoch in range(epochs):
    train_loss, train_acc = train(model, optimizer)
    print('epoch={},训练准确率={}，损失={}'.format(epoch, train_acc, train_loss))
    test = evaluate(model)
    test_acc = test[0]
    f_score = test[1]
    f_all = test[2]
    print("epoch={},测试准确率={},f1加权值={}".format(epoch, test_acc,f_score))
    print("各个分类的精确度、召回、f得分如下:{}".format(f_all))
    # if test_acc > 0.78 or train_loss < 0.1 or train_acc > 0.97: break




def predict(sen):

    input_id = convert_text_to_token(tokenizer, sen)
    input_token = torch.tensor(input_id).long().to(device)            #torch.Size([128])

    atten_mask = [float(i>0) for i in input_id]
    attention_token = torch.tensor(atten_mask).long().to(device)       #torch.Size([128])

    output = model(input_token.view(1, -1), token_type_ids=None, attention_mask=attention_token.view(1, -1))     #torch.Size([128])->torch.Size([1, 128])否则会报错
    # print(output[0])

    return torch.max(output[0], dim=1)[1]

def read_test(file_path):
    src_data = []
    fout = pd.read_excel(file_path,usecols=[10])#预测第几列的数据就填几
    # dataframe转化成list
    fout = fout.values.tolist()
    # 去掉第一行
    fout = fout[1:]
    for line in tqdm(fout,desc='precessing data'):
        pair = line[0]
        lp = len(pair)
        if type(pair) is not str:
            print(type(pair))
            continue
        else:
            # pair = pair.strip(u'\u200b').split('\t')
            # src_data.append(seg.cut(pair[0]))
            src_data.append(pair)
    return (src_data)

def deal(ans):
    # 列表
    # list转dataframe
    df = pd.DataFrame(ans, columns=['final'])
    # 保存到本地excel
    df.to_excel("predict_5000.xlsx", index=False)

ans = []
test = read_test('5000.xlsx')
i = 0
for sentence in test:
    # if i == 0: print(sentence)
    i += 1
    ans.append(predict(sentence).item())
deal(ans)
