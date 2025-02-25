# 导入必要模块
import os
import requests
import math
import tiktoken
import torch
import torch.nn as nn
from torch.nn import functional as F

# 超参数设置
batch_size = 4  # 每次训练步骤的批量大小
context_length = 16  # 每个批次处理的上下文长度（序列长度）
d_model = 64  # 模型中 token 嵌入的维度
num_blocks = 8  # Transformer 的层数（block 数量）
num_heads = 4  # 多头注意力中的头数
learning_rate = 1e-3  # 学习率
dropout = 0.1  # Dropout 比例
max_iters = 5000  # 最大训练迭代次数
eval_interval = 50  # 评估间隔（每隔多少步评估模型）
eval_iters = 20  # 评估时的迭代次数（计算评估损失的平均值）
device = 'cuda' if torch.cuda.is_available() else 'cpu'  # 优先使用 GPU
TORCH_SEED = 1337  # 设置随机种子以确保结果可复现
torch.manual_seed(TORCH_SEED)  # 初始化 PyTorch 的随机种子

# 加载训练数据
if not os.path.exists('data/sales_textbook.txt'):
    # 如果文件不存在，从 Hugging Face 下载文本数据
    url = 'https://huggingface.co/datasets/goendalf666/sales-textbook_for_convincing_and_selling/raw/main/sales_textbook.txt'
    with open('data/sales_textbook.txt', 'w') as f:
        f.write(requests.get(url).text)

# 读取文本文件
with open('data/sales_textbook.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# 使用 TikToken 对文本进行分词（与 GPT-3 的分词器一致）
encoding = tiktoken.get_encoding("cl100k_base")
tokenized_text = encoding.encode(text)
max_token_value = max(tokenized_text) + 1  # 获取最大 token 值，用于嵌入层的定义
tokenized_text = torch.tensor(tokenized_text, dtype=torch.long, device=device)  # 转换为 PyTorch 张量

# 数据集划分：90% 用于训练，10% 用于验证
split_idx = int(len(tokenized_text) * 0.9)
train_data = tokenized_text[:split_idx]
val_data = tokenized_text[split_idx:]

'''
前馈神经网络（FeedForward Neural Network, FFN）是神经网络中的一种基础结构，主要用于处理输入的特征并通过全连接层（Linear Layers）进行信息的变换。其作用可以概括为：

1. 特征变换与非线性映射
前馈网络通过一系列的全连接层进行线性变换，将输入的特征空间从原始的维度映射到更高或更低的维度。通常会通过激活函数（如 ReLU）加入非线性，使得网络能够学习到复杂的特征。

2. 数据的扩展与压缩
在你提供的代码中，前馈网络的设计将输入 d_model 映射到 d_model * 4（扩展至更高维度），然后通过激活函数学习非线性特征，再将维度压缩回 d_model。这种结构使得网络能够在较大的特征空间中学习更多的表示，然后再将其压缩回原来的维度，以便与其他网络模块结合。

3. 防止过拟合
Dropout 操作在前馈网络中被用来减少过拟合，通过随机丢弃部分神经元的输出，增加模型的泛化能力。

4. 作用在注意力机制中的配合
在你提供的代码中，前馈网络被嵌入到更复杂的架构中，例如与注意力机制（Attention）结合。通常，前馈网络用于对每个注意力模块的输出进行进一步的处理。具体来说，注意力模块主要关注输入特征之间的关系，而前馈网络则负责通过非线性变换进一步提取特征。
'''


# 定义前馈神经网络模块（Feed Forward Network）
class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        self.d_model = d_model
        self.dropout = dropout
        self.ffn = nn.Sequential(
            nn.Linear(in_features=self.d_model, out_features=self.d_model * 4),  # 扩展至更高维度
            nn.ReLU(),  # 激活函数
            nn.Linear(in_features=self.d_model * 4, out_features=self.d_model),  # 缩减回 d_model 维度
            nn.Dropout(dropout),  # 添加 Dropout 以减少过拟合
        )

    def forward(self, x):
        return self.ffn(x)  # 前向传播


# 定义单头注意力机制
class Attention(nn.Module):
    def __init__(self, head_size: int):
        super().__init__()
        self.d_model = d_model
        self.head_size = head_size
        self.context_length = context_length
        self.dropout = dropout

        # 定义键（Key）、查询（Query）和值（Value）的线性变换
        self.key_layer = nn.Linear(in_features=self.d_model, out_features=self.head_size, bias=False)
        self.query_layer = nn.Linear(in_features=self.d_model, out_features=self.head_size, bias=False)
        self.value_layer = nn.Linear(in_features=self.d_model, out_features=self.head_size, bias=False)

        # 创建一个下三角矩阵，用于遮掩未来时间步的注意力
        self.register_buffer('tril', torch.tril(torch.ones((self.context_length, self.context_length))))
        self.dropout_layer = nn.Dropout(self.dropout)

    def forward(self, x):
        # x 的维度为 (batch_size, sequence_length, embedding_dim)
        B, T, C = x.shape
        assert T <= self.context_length
        assert C == self.d_model

        '''
        理解为什么不同的头会关注不同的上下文信息，关键在于线性变换的作用。让我从更具体的角度来解释：

        1. 线性变换的作用
        每个注意力头有独立的查询（Q）、键（K）和值（V）权重矩阵，实际上是对输入数据应用不同的线性变换。简单来说，线性变换就是通过一个权重矩阵（例如 self.query_layer）对输入数据做一次映射，将输入向量转换为新的向量。
        
        假设我们有输入 x = [我, 是, 张, 三, 他, 李]，每个词向量的维度为 d_model，然后通过以下方式得到查询、键和值：
        
        q = self.query_layer(x)
        k = self.key_layer(x)
        v = self.value_layer(x)
        这里，q, k, v 都是通过线性变换生成的，每个变换都应用了不同的权重矩阵。换句话说，每个头的 q, k, v 对输入的映射都是不同的。尽管输入 x 是相同的，生成的 q, k, v 仍然会不同，因为每个头的权重矩阵不一样。
        
        2. 不同的头关注不同的上下文信息
        在多头注意力中，每个头都有一个独立的线性变换，这意味着每个头会学习到不同的特征。你可以想象每个头像是一个“专家”，它专门学习不同类型的关系。
        
        头1：可能专注于近距离的词对关系（比如“我”和“是”之间的关系）。这可能是因为头1的权重矩阵学习到了这种近距离依赖的特征。
        
        比如，假设 q_1 是针对当前词“是”生成的查询向量，k_1 是“我”的键向量，计算点积时，可能会得到更高的注意力权重，因为“我”和“是”在语法上通常是紧密联系的。
        
        头2：可能专注于长距离的依赖关系（比如“张”和“他”之间的关系）。这个头的权重矩阵可能学习到了长距离依赖的特征，使得它能够捕捉到像“张”和“他”之间的关系。
        
        比如，假设 q_2 是针对“是”生成的查询向量，k_2 是“张”的键向量，尽管“是”和“张”之间的距离较远，头2可能仍然能学习到它们之间的依赖关系，因此在计算点积时，它可能会给“张”更高的注意力权重。
        
        3. 为何会有不同的关注点
        关键在于 线性变换的参数（权重矩阵）是不同的，这些参数是在训练过程中优化的。训练时，模型会通过反向传播不断调整这些权重，以便每个头能学习到最有用的表示。
        
        头1 的权重矩阵可能会变得擅长捕捉近距离的词对关系，因为在训练数据中，“我”和“是”这样的词对经常一起出现，模型因此学会了这种模式。
        
        头2 的权重矩阵可能会变得擅长捕捉长距离的依赖关系，因为在训练数据中，“张”和“他”这样的关系较为重要，模型因此学会了这种模式。
        
        这些差异是通过训练过程中模型的学习（即优化权重矩阵）自动实现的。
        
        4. 一个具体例子
        假设我们处理的是句子 "我 是 张 三 他 李"，在训练时：
        
        头1 可能学习到的是短距离关系。例如，词语 "我" 和 "是" 紧密相连，"张" 和 "三" 紧密相连。这意味着头1会更多关注这些近距离词对。
        
        头2 可能学习到的是长距离依赖。例如，词语 "是" 和 "张"，或 "三" 和 "李" 之间的关系，模型通过训练发现这些长距离关系是重要的，因此头2会关注这些长距离的依赖。
        
        5. 总结
        每个注意力头的 q, k, v 都是通过不同的线性变换计算的，权重矩阵在训练过程中优化，导致每个头的表示会有所不同。最终，多个头会并行计算注意力，这样模型就能从多个角度（例如近距离关系和长距离依赖）同时捕捉上下文信息，从而增强模型的表达能力。
        
        希望这样的解释能帮助你理解多头注意力为何会关注不同的上下文信息！
        '''

        # 通过线性变换生成 Q, K, V
        q = self.query_layer(x)
        k = self.key_layer(x)
        v = self.value_layer(x)

        # 计算缩放点积注意力
        weights = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        weights = weights.masked_fill(self.tril[:T, :T] == 0, float('-inf'))  # 应用遮掩
        weights = F.softmax(input=weights, dim=-1)  # 归一化注意力权重
        weights = self.dropout_layer(weights)

        # 使用权重对 V 进行加权平均
        out = weights @ v
        return out


# 定义多头注意力模块
class MultiHeadAttention(nn.Module):
    def __init__(self, head_size: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_size = head_size
        self.d_model = d_model
        self.context_length = context_length
        self.dropout = dropout

        # 多个单头注意力模块
        self.heads = nn.ModuleList([Attention(head_size=self.head_size) for _ in range(self.num_heads)])
        # 输出线性投影
        self.projection_layer = nn.Linear(in_features=self.d_model, out_features=self.d_model)
        self.dropout_layer = nn.Dropout(dropout)

    def forward(self, x):
        # 连接多个头的输出，并通过线性投影得到最终输出
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.projection_layer(out)
        out = self.dropout_layer(out)
        return out


class TransformerBlock(nn.Module):
    def __init__(self, num_heads: int):
        """
        TransformerBlock: 实现了一个 Transformer 的基本单元，包括多头注意力机制和前馈神经网络。
        :param num_heads: 多头注意力的头数。
        """
        super().__init__()
        self.d_model = d_model  # 模型的隐藏层维度
        self.context_length = context_length  # 上下文窗口长度
        self.head_size = d_model // num_heads  # 每个注意力头的尺寸
        self.num_heads = num_heads  # 多头注意力的头数
        self.dropout = dropout  # Dropout 概率

        # 多头注意力层
        self.multi_head_attention_layer = MultiHeadAttention(head_size=self.head_size)
        # 前馈神经网络层
        self.feed_forward_layer = FeedForward()
        # LayerNorm 层，用于归一化输入
        self.layer_norm_1 = nn.LayerNorm(normalized_shape=self.d_model)
        self.layer_norm_2 = nn.LayerNorm(normalized_shape=self.d_model)

    def forward(self, x):
        """
        前向传播逻辑。
        :param x: 输入张量，形状为 (B, T, d_model)
        :return: 输出张量，形状为 (B, T, d_model)
        """
        # 第一步：通过 LayerNorm 再进入多头注意力机制，最后加入残差连接
        x = x + self.multi_head_attention_layer(self.layer_norm_1(x))
        # 第二步：通过 LayerNorm 再进入前馈神经网络，最后加入残差连接
        x = x + self.feed_forward_layer(self.layer_norm_2(x))
        return x


class TransformerLanguageModel(nn.Module):
    def __init__(self):
        """
        TransformerLanguageModel: 使用多个 TransformerBlock 构建一个完整的语言模型。
        """
        super().__init__()
        self.d_model = d_model  # 模型的隐藏层维度
        self.context_length = context_length  # 上下文窗口长度
        self.num_heads = num_heads  # 多头注意力的头数
        self.num_blocks = num_blocks  # TransformerBlock 的数量
        self.dropout = dropout  # Dropout 概率
        self.max_token_value = max_token_value  # 词汇表的最大索引值

        # 嵌入层：将离散的词索引转换为稠密向量
        self.token_embedding_lookup_table = nn.Embedding(
            num_embeddings=self.max_token_value + 1,
            embedding_dim=self.d_model
        )

        # Transformer 块：由多个 TransformerBlock 组成，并在最后加一个 LayerNorm
        self.transformer_blocks = nn.Sequential(*(
                [TransformerBlock(num_heads=self.num_heads) for _ in range(self.num_blocks)] +
                [nn.LayerNorm(self.d_model)]  # 最后加一个 LayerNorm
        ))

        # 输出线性层：将隐藏状态映射到词汇表大小的维度上
        self.language_model_out_linear_layer = nn.Linear(
            in_features=self.d_model,
            out_features=self.max_token_value
        )

    def forward(self, idx, targets=None):
        """
        前向传播逻辑。
        :param idx: 输入词索引，形状为 (B, T)
        :param targets: 目标词索引，形状为 (B, T)
        :return: 模型输出 logits 和 loss（如果提供 targets）
        """
        B, T = idx.shape  # 获取 batch 大小和上下文窗口长度

        # 位置嵌入：生成位置编码表，用于为每个时间步添加位置信息
        position_encoding_lookup_table = torch.zeros(self.context_length, self.d_model)
        position = torch.arange(0, self.context_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.d_model, 2).float() * (-math.log(10000.0) / self.d_model))
        position_encoding_lookup_table[:, 0::2] = torch.sin(position * div_term)  # 偶数位置使用 sin
        position_encoding_lookup_table[:, 1::2] = torch.cos(position * div_term)  # 奇数位置使用 cos

        # 限制位置编码表大小并迁移到设备上
        position_embedding = position_encoding_lookup_table[:T, :].to(device)
        # 嵌入输入词并加上位置编码
        x = self.token_embedding_lookup_table(idx) + position_embedding

        # 通过所有 Transformer 块
        x = self.transformer_blocks(x)

        # 输出 logits
        logits = self.language_model_out_linear_layer(x)

        if targets is not None:
            # 计算交叉熵损失
            B, T, C = logits.shape
            logits_reshaped = logits.view(B * T, C)
            targets_reshaped = targets.view(B * T)
            loss = F.cross_entropy(input=logits_reshaped, target=targets_reshaped)
        else:
            loss = None
        return logits, loss

    def generate(self, idx, max_new_tokens):
        """
        基于给定上下文生成新的 token。
        :param idx: 初始输入词索引，形状为 (B, T)
        :param max_new_tokens: 要生成的新 token 的最大数量
        :return: 包含新生成 token 的序列，形状为 (B, T + max_new_tokens)
        """
        for _ in range(max_new_tokens):
            # 限制上下文长度
            idx_crop = idx[:, -self.context_length:]
            # 获取预测值
            logits, loss = self(idx_crop)
            # 获取最后一个时间步的预测
            logits_last_timestep = logits[:, -1, :]
            # 转为概率分布
            probs = F.softmax(input=logits_last_timestep, dim=-1)
            # 从概率分布中采样一个新的 token
            idx_next = torch.multinomial(input=probs, num_samples=1)
            # 将新 token 加入到序列中
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


# 初始化 Transformer 模型实例
model = TransformerLanguageModel()
# 将模型转移到指定设备（如 GPU 或 CPU）
model = model.to(device)


# 获取输入数据的批次（用于训练或验证）
def get_batch(split: str):
    """
    从数据集中随机采样一个批次数据，分为输入 x 和目标 y。
    - split: 指定数据集类型，'train' 表示训练集，'valid' 表示验证集。
    """
    # 根据 split 参数选择训练或验证数据
    data = train_data if split == 'train' else val_data
    # 随机生成索引，用于从数据中提取连续的 context_length 长度的序列
    # 随机生成索引，用于从数据中提取连续的 context_length 长度的序列
    idxs = torch.randint(low=0, high=len(data) - context_length, size=(batch_size,))
    # 使用生成的索引创建输入数据 x
    # 使用生成的索引创建输入数据和目标数据
    x = torch.stack([data[idx:idx + context_length] for idx in idxs]).to(device)
    # 使用生成的索引创建目标数据 y，y 是从 x 的下一个单词开始
    y = torch.stack([data[idx + 1:idx + context_length + 1] for idx in idxs]).to(device)
    # 返回输入和目标数据
    return x, y


# 估算训练和验证集的平均损失
@torch.no_grad()  # 禁用梯度计算以提高效率
def estimate_loss():
    """
    评估模型在训练集和验证集上的平均损失，用于跟踪训练过程中的性能。
    """
    out = {}  # 存储训练集和验证集的损失
    model.eval()  # 将模型设置为评估模式，禁用 dropout 等操作
    for split in ['train', 'valid']:
        losses = torch.zeros(eval_iters)  # 用于存储每次迭代的损失
        for k in range(eval_iters):
            # 获取当前批次数据
            x_batch, y_batch = get_batch(split)
            # 计算模型输出和损失
            logits, loss = model(x_batch, y_batch)
            losses[k] = loss.item()
        # 计算平均损失并存储
        out[split] = losses.mean()
    model.train()  # 恢复模型为训练模式
    return out


# 定义 AdamW 优化器
optimizer = torch.optim.AdamW(params=model.parameters(), lr=learning_rate)

# 用于跟踪每次评估的损失
tracked_losses = list()

# 开始训练过程
for step in range(max_iters):
    # 每 eval_iters 次或训练最后一步，评估当前模型性能
    if step % eval_iters == 0 or step == max_iters - 1:
        losses = estimate_loss()  # 获取评估损失
        tracked_losses.append(losses)  # 保存评估结果
        print('Step:', step, 'Training Loss:', round(losses['train'].item(), 3),
              'Validation Loss:', round(losses['valid'].item(), 3))

    # 获取训练批次数据
    xb, yb = get_batch('train')
    # 计算模型输出和损失
    logits, loss = model(xb, yb)
    # 清空优化器的梯度
    optimizer.zero_grad(set_to_none=True)
    # 反向传播计算梯度
    loss.backward()
    # 执行优化器的步长更新
    optimizer.step()

# 保存训练完成后的模型参数
torch.save(model.state_dict(), 'model-ckpt.pt')

# 使用模型进行文本生成
model.eval()  # 设置模型为评估模式
start = 'The salesperson'  # 初始输入文本
# 将文本编码为 token 索引
start_ids = encoding.encode(start)
# 转换为张量并扩展维度以适配模型输入
x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])
# 调用模型的生成函数，生成 max_new_tokens 个新 token
y = model.generate(x, max_new_tokens=100)
# 解码生成的 token 为文本并输出
print('---------------')
print(encoding.decode(y[0].tolist()))
print('---------------')
