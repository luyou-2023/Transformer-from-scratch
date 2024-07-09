import os
import requests
import pandas as pd
import matplotlib.pyplot as plt
import math
import tiktoken
import torch
import torch.nn as nn


# Hyperparameters
batch_size = 4  # How many batches per training step
context_length = 16  # Length of the token chunk each batch
d_model = 64  # The vector size of the token embeddings
num_layers = 8  # Number of transformer blocks
num_heads = 4  # Number of heads in Multi-head attention # 我们的代码中通过 d_model / num_heads = 来获取 head_size
learning_rate = 1e-3  # 0.001
dropout = 0.1 # Dropout rate
max_iters = 5000  # Total of training iterations
eval_interval = 50  # How often to evaluate the model
eval_iters = 20  # How many iterations to average the loss over when evaluating the model
device = 'cuda' if torch.cuda.is_available() else 'cpu'  # Instead of using the cpu, we'll use the GPU if it's available.

TORCH_SEED = 1337
torch.manual_seed(TORCH_SEED)

# download a sample txt file from https://huggingface.co/datasets/goendalf666/sales-textbook_for_convincing_and_selling/raw/main/sales_textbook.txt
if not os.path.exists('sales_textbook.txt'):
    url = 'https://huggingface.co/datasets/goendalf666/sales-textbook_for_convincing_and_selling/raw/main/sales_textbook.txt'
    with open('sales_textbook.txt', 'w') as f:
        f.write(requests.get(url).text)

with open('sales_textbook.txt', 'r', encoding='utf-8') as f:
    text = f.read()

encoding = tiktoken.get_encoding("cl100k_base")
tokenized_text = encoding.encode(text) # size of tokenized source text is 77,919
vocab_size = len(set(tokenized_text)) # size of vocabulary is 3,771
max_token_value = max(tokenized_text)

print(f"Tokenized text size: {len(tokenized_text)}")
print(f"Vocabulary size: {vocab_size}")
print(f"The maximum value in the tokenized text is: {max_token_value}")


# Split train and validation
split_idx = int(len(tokenized_text) * 0.8)
train_data = tokenized_text[:split_idx]
val_data = tokenized_text[split_idx:]

# Prepare data for training batch
data = train_data
idxs = torch.randint(low=0, high=len(data) - context_length, size=(batch_size,))
x_batch = torch.stack([torch.tensor(data[idx:idx + context_length]) for idx in idxs])
y_batch = torch.stack([torch.tensor(data[idx + 1:idx + context_length + 1]) for idx in idxs])
print(x_batch.shape, y_batch.shape)


# Define Token Embedding look-up table
token_embedding_lookup_table = nn.Embedding(max_token_value, d_model)

# Get X and Y embedding
x = token_embedding_lookup_table(x_batch.data)
y = token_embedding_lookup_table(y_batch.data)

# Define Position Encoding look-up table
position_encoding_lookup_table = torch.zeros(context_length, d_model) # initial with zeros with shape (context_length, d_model)
position = torch.arange(0, context_length, dtype=torch.float).unsqueeze(1)
# apply the sine & cosine
div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
position_encoding_lookup_table[:, 0::2] = torch.sin(position * div_term)
position_encoding_lookup_table[:, 1::2] = torch.cos(position * div_term)
position_encoding_lookup_table = position_encoding_lookup_table.unsqueeze(0).expand(batch_size, -1, -1) #add batch to the first dimension

print("Position Encoding Look-up Table: ", position_encoding_lookup_table.shape)


# Add positional encoding into the input embedding vector
input_embedding_x = x + position_encoding_lookup_table # [4, 16, 64] [batch_size, context_length, d_model]
input_embedding_y = y + position_encoding_lookup_table

X = input_embedding_x

x_plot = input_embedding_x[0].detach().cpu().numpy()
print("Final Input Embedding of x: \n", pd.DataFrame(x_plot))

# 多头注意力概述﻿ 现在我们有了输入嵌入X，我们可以开始实现多头注意力模块了。实现多头注意力模块需要一系列步骤。让我们一一编写代码。
#准备Q、K、V﻿

# Prepare Query, Key, Value for Multi-head Attention

query = key = value = X # [4, 16, 64] [batch_size, context_length, d_model]

# Define Query, Key, Value weight matrices
Wq = nn.Linear(d_model, d_model)
Wk = nn.Linear(d_model, d_model)
Wv = nn.Linear(d_model, d_model)

Q = Wq(query) #[4, 16, 64]
Q = Q.view(batch_size, -1, num_heads, d_model // num_heads)  #[4, 16, 4, 16]

K = Wk(key) #[4, 16, 64]
K = K.view(batch_size, -1, num_heads, d_model // num_heads)  #[4, 16, 4, 16]

V = Wv(value) #[4, 16, 64]
V = V.view(batch_size, -1, num_heads, d_model // num_heads)  #[4, 16, 4, 16]

#将 Q、K、V 重塑为 [batch_size、num_heads、context_length、head_size] 以进行进一步计算。
# Transpose q,k,v from [batch_size, context_length, num_heads, head_size] to [batch_size, num_heads, context_length, head_size]
# The reason is that treat each batch with "num_heads" as its first dimension.
Q = Q.transpose(1, 2) # [4, 4, 16, 16]
K = K.transpose(1, 2) # [4, 4, 16, 16]
V = V.transpose(1, 2) # [4, 4, 16, 16]

#计算QK^T Attention
# Calculate the attention score betwee Q and K^T
# attention_score = torch.matmul(Q, K.transpose(-2, -1))

# Then Scale the attention score by the square root of the head size
# attention_score = attention_score / math.sqrt(d_model // num_heads)

attention_score = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_model // num_heads) # [4, 4, 16, 16] #[4, 4, 16, 16] [batch_size, num_heads, context_length, context_length]
print(pd.DataFrame(attention_score[0][0].detach().cpu().numpy()))

# Apply Mask to attention scores
attention_score = attention_score.masked_fill(torch.triu(torch.ones(attention_score.shape[-2:]), diagonal=1).bool(), float('-inf')) #[4, 4, 16, 16] [batch_size, num_heads, context_length, context_length]
print(pd.DataFrame(attention_score[0][0].detach().cpu().numpy()))

# Softmax the attention score
attention_score = torch.softmax(attention_score, dim=-1)  #[4, 4, 16, 16] [batch_size, num_heads, context_length, context_length]
print(pd.DataFrame(attention_score[0][0].detach().cpu().numpy()))

# 计算V注意力
# Calculate the V attention output
A = torch.matmul(attention_score, V) # [4, 4, 16, 16] [batch_size, num_heads, context_length, head_size]
print(A.shape)

#连接并输出 需要连接多头注意力块的输出并将其输入到线性层
A = A.transpose(1, 2) # [4, 16, 4, 16] [batch_size, context_length, num_heads, head_size]
A = A.reshape(batch_size, -1, d_model) # [4, 16, 64] [batch_size, context_length, d_model]

#可以应用另一个 [64,64] 线性层Wo （这是训练期间学习到的权重）并得到多​​头注意力模块的最终输出：

# Define the output weight matrix
Wo = nn.Linear(d_model, d_model)
output = Wo(A) # [4, 16, 64] [batch_size, context_length, d_model]

print(output.shape)

#残差连接和层归一化

# Add residual connection
output = output + X

# Add Layer Normalization
layer_norm = nn.LayerNorm(d_model)
output = layer_norm(output)

#前馈网络﻿
# Define Feed Forward Network
output = nn.Linear(d_model, d_model * 4)(output)
output = nn.ReLU()(output)
output = nn.Linear(d_model * 4, d_model)(output)
output = torch.dropout(output, p=dropout, train=True)

# Add residual connection
output = output + X
# Add Layer Normalization
layer_norm = nn.LayerNorm(d_model)
output = layer_norm(output)

#上面我们完成的只是一个 Transformer 块。实际操作中，我们会将多个 Transformer 块堆叠在一起，形成一个 Transformer 解码器。
# 我们实际上应该将代码打包成类并使用 PyTorch nn.Module 来构建我们的 Transformer 解码器。但为了演示，我们只保留一个块。

logits = nn.Linear(d_model, max_token_value)(output)
print(pd.DataFrame(logits[0].detach().cpu().numpy()))

# 最后一步是对logits 进行softmax以获得每个 token 的概率：
# torch.softmax usually used during inference, during training we use torch.nn.CrossEntropyLoss
# but for illustration purpose, we'll use torch.softmax here
probabilities = torch.softmax(logits, dim=-1)
