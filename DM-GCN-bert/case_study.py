import os
import random
import argparse
import numpy as np
import pickle
from sklearn import metrics
from loader import DataLoader
from trainer import GCNTrainer
from utils import helper
import torch
import torch.nn.functional as F
from copy import deepcopy
import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib as mpl


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='Restaurants')
parser.add_argument('--emb_dim', type=int, default=300, help='Word embedding dimension.')
parser.add_argument('--post_dim', type=int, default=30, help='Position embedding dimension.')
parser.add_argument('--pos_dim', type=int, default=30, help='Pos embedding dimension.')
parser.add_argument('--hidden_dim', type=int, default=300, help='GCN mem dim.')
parser.add_argument('--rnn_hidden', type=int, default=300, help='RNN hidden state size.')
parser.add_argument('--num_layers', type=int, default=2, help='Num of GCN layers.')
parser.add_argument('--num_class', type=int, default=3, help='Num of sentiment class.')
parser.add_argument('--input_dropout', type=float, default=0, help='Input dropout rate.')
parser.add_argument('--gcn_dropout', type=float, default=0., help='GCN layer dropout rate.')
parser.add_argument('--lower', default=True, help='Lowercase all words.')
parser.add_argument('--direct', default=False)
parser.add_argument('--loop', default=True)
parser.add_argument('--lr', type=float, default=0.01, help='learning rate.')
parser.add_argument('--l2reg', type=float, default=1e-5, help='l2 .')
# parser.add_argument('--num_epoch', type=int, default=100, help='Number of total training epochs.')
parser.add_argument('--save_dir', type=str, default='./saved_models/best_model_rest.pt', help='Root dir for saving models.')
parser.add_argument('--head_num', default=3, type=int, help='head_num must be a multiple of 3')
parser.add_argument('--top_k', default=2, type=int)
parser.add_argument('--beta', default=1.0, type=float)
parser.add_argument('--theta', default=1.0, type=float)
parser.add_argument('--second_layer', default='max', type=str)
parser.add_argument('--DEVICE', default='cuda:0', type=str)
parser.add_argument('--batch_size', default=8, type=int)
args = parser.parse_args()

# load contants
dicts = eval(open('./dataset/'+args.dataset+'/constant.py', 'r').read())
vocab_file = './dataset/'+args.dataset+'/vocab.pkl'
token_vocab = dict()
with open(vocab_file, 'rb') as infile:
    token_vocab['i2w'] = pickle.load(infile)
    token_vocab['w2i'] = {token_vocab['i2w'][i]:i for i in range(len(token_vocab['i2w']))}
emb_file = './dataset/'+args.dataset+'/embedding.npy'
emb_matrix = np.load(emb_file)
assert emb_matrix.shape[0] == len(token_vocab['i2w'])
assert emb_matrix.shape[1] == args.emb_dim

args.token_vocab_size = len(token_vocab['i2w'])
args.post_vocab_size = len(dicts['post'])
args.pos_vocab_size = len(dicts['pos'])

dicts['token'] = token_vocab['w2i']

# load training set and test set
print("Loading data from {} with batch size {}...".format(args.dataset, args.batch_size))
test_batch = DataLoader('./dataset/'+args.dataset+'/case_study1.json', args.batch_size, args, dicts)

with open('./dataset/'+args.dataset+'/case_study1.json') as infile:
    data = json.load(infile)
sentence = data[0]["token"]
print(sentence)

# create the model
trainer = GCNTrainer(args, emb_matrix=emb_matrix)

# 加载模型
print("Loading model from {}".format(args.save_dir))
#trainer.load(args.save_dir)
DEVICE_ID = 0
DEVICE = torch.device(args.DEVICE if torch.cuda.is_available() else 'cpu')
mdict = torch.load(args.save_dir, map_location=DEVICE)
print(mdict['config'])
model_dict = trainer.model.state_dict()
pretrained_dict = {k: v for k, v in mdict['model'].items() if k in model_dict}
model_dict.update(pretrained_dict)
trainer.model.load_state_dict(model_dict)

# 计算每个单词的贡献度
# 这段代码之前嵌在模型内部，如果需要改成独立于模型的函数，可以将h,h_w作为模型计算得到的参数传入。
batch = [b.cuda() for b in test_batch[0]]
inputs = batch[0:9]
length = inputs[0].size(1)    #num of tokens

trainer.model.eval()
logits, h, _, _, _, _ = trainer.model(inputs)      #conventional procedure   size of h:(1,50)
predprob = F.softmax(logits, dim=1).data.cpu().numpy().tolist()
h = h[0].cpu().detach().numpy()   #convert to numpy h为正常得分（所有词都没有mask）
print(predprob)

h_w = list(range(length))
r = [0.00 for _ in range(length)]    #score

for i in range(length):
    batch = [deepcopy(b).cuda() for b in test_batch[0]]
    tok, asp, pos, head, post, dep, mask, l, adj = batch[0:9]
    tok[0][i] = 0
    inputs = [tok, asp, pos, head, post, dep, mask, l, adj]

    logits, h_w[i], _, _, _, _ = trainer.model(inputs)   #h_w[i]是第i个词被mask掉的得分
    predprob_ = F.softmax(logits, dim=1).data.cpu().numpy().tolist()
    print(predprob)
    h_w[i] = h_w[i].squeeze(0).cpu().detach().numpy()
    for dim in range(len(h)):
        r[i] += abs(h[dim]-h_w[i][dim])
    # r[i] = abs(predprob_[0][0] - predprob[0][0])

max_r = max(r)

r = [r[i]/max_r for i in range(length)]

print(r)


# draw heatmap
mpl.rcParams['figure.subplot.top'] = 0.50
mpl.rcParams['figure.subplot.bottom'] = 0.44
mpl.rcParams['figure.subplot.left'] = 0.001
mpl.rcParams['figure.subplot.right'] = 1.0

#plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.family"] = "DejaVu Sans"
#sns.set(font="Times New Roman")

def store_heat(df_data):
    plt.subplots(figsize=(df_data.shape[1], df_data.shape[0]))
    cmap = sns.light_palette((260, 75, 60), input="husl")
    sns.heatmap(df_data, annot=True, linewidths=0.5, xticklabels=True, yticklabels=True, cmap=cmap, cbar=False)
    plt.show()

#一个使用例子，df_data为DataFrame类型的数据，列名（columns需设置为对应显示的字符串）
#path为保存路径
df_data = pd.DataFrame([r,r], index=['1', '2'], columns=[word for word in sentence])
print(df_data)

store_heat(df_data)

