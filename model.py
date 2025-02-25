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
        
        Q（查询）、K（键）、V（值）在初始化时是随机的，理论上它们之间可能没有明显的关系。不过，在训练过程中，模型会通过反向传播不断调整这些权重，优化它们之间的关系，以便学习到有用的特征和模式。

        1. 初始化时的随机性
        在网络的初始化阶段，q = self.query_layer(x)、k = self.key_layer(x) 和 v = self.value_layer(x) 都是通过独立的线性变换生成的，这些线性变换是由模型的参数（即权重矩阵）决定的。在初始化时，这些权重矩阵的值通常是随机的。初始化的随机性意味着，刚开始时 q、k 和 v 之间并没有必然的相似关系。也就是说，刚开始的时候，它们可能并不直接对应于有用的模式或语义信息。
        
        2. 训练中的优化
        虽然在初始化时，q、k 和 v 之间没有明显的关系，但随着训练的进行，模型通过反向传播和优化算法（比如 Adam）调整这些权重，逐渐学习到如何使得查询向量 Q 与键向量 K 之间的相似性能够更好地反映词与词之间的关系。
        
        通过计算注意力权重 weights = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))，模型实际上是在计算 Q 和 K 之间的相似度。这个相似度会通过 softmax 函数归一化，最终形成每个词对的注意力权重。这些注意力权重决定了模型如何根据当前的查询（Q）来加权不同的键（K）和值（V）。所以，训练过程中，Q 和 K 的相似性会不断优化，以便模型能学习到不同词之间的依赖关系。
        
        3. Q 和 K、V 之间的关系
        Q 和 K 的相似度：查询向量 Q 和键向量 K 的点积是计算注意力权重的关键，表示了当前输入与其他输入的相关性。在训练过程中，Q 和 K 之间的相似性会逐渐学习到有意义的模式，比如在同一语境中，某些词会更加相关。
        
        Q 和 V 的关系：Q 和 V 之间的关系通过计算注意力权重来间接建立。V 是值向量，最终通过注意力权重进行加权平均，生成模型的输出。Q 和 V 之间没有直接的点积关系，而是通过 K 之间的关系间接影响。
        
        4. 总结
        在初始阶段，Q、K 和 V 的关系是随机的，可能没有太多显著的语义或上下文相关性。
        在训练过程中，Q、K 和 V 会通过梯度下降进行调整，学习到更符合数据特征和任务需求的关系。
        最终，Q 和 K 之间的相似性会反映出上下文中的依赖关系，而 V 会基于这些注意力权重来生成最终的输出。
        这就是为什么虽然一开始 Q 和 K、V 没有太多显著的关系，但通过训练，模型能够学到如何根据 Q 和 K 的相似度来加权 V，从而捕捉到复杂的上下文信息。
        
        
        Q（查询）确实可以对应多个相似的 K（键）和 V（值），这正是多头注意力机制的核心思想之一。每个注意力头通过独立的线性变换来学习不同的 Q、K、V，从而捕捉不同的语义或上下文关系。通过这种方式，模型能够在一个序列中同时学习到多个不同的模式或依赖关系。

        1. Q 对应多个 K 和 V
        在多头注意力机制中，Q 可以与多个 K 进行对比，来计算多个不同的注意力权重。每个注意力头有自己独立的权重矩阵，这使得每个头能够关注不同的部分。因此，对于一个给定的 Q，它可能会对多个 K 和 V 进行加权计算，捕捉不同的上下文信息。
        
        举个例子，假设输入的句子是：
        
        复制
        编辑
        我是张三，他是李四，他们是乐队组合。
        如果我们关注的是 "是" 这个词，那么：
        
        头1（可能关注近距离的依赖关系）可能会重点关注 "是" 和 "我"、"张" 之间的关系，认为它们在语法和句法上紧密相关。
        头2（可能关注长距离依赖）可能会关注 "是" 和 "他们"、"乐队" 之间的关系，学习到 "是" 在长距离依赖中的作用。
        头3（可能关注语义关系）可能会捕捉 "是" 在不同上下文中的不同含义，如它在不同语境中的作用，可能会将 "是" 和 "张" 或其他词关联。
        2. 多头注意力的意义
        每个注意力头通过独立的权重矩阵来学习自己感兴趣的上下文信息。具体地：
        
        每个头的权重矩阵（查询、键、值）初始化不同，因此每个头可以学习到不同类型的模式或上下文。
        在每个注意力头内部，计算出的权重是 Q 和 K 的相似度，决定了模型如何对不同的 V 进行加权。
        多个头的计算结果被 拼接（concatenate）起来，然后通过一个 线性变换 将它们合并为最终的输出。
        3. 多头的优势
        不同上下文的学习：每个头专注于学习不同类型的上下文依赖（如短程依赖、长程依赖、语法关系、语义关系等）。这种方式使得模型能够更好地理解复杂的语言特性。
        增强表达能力：通过多个头并行处理不同的注意力信息，模型能够在同一个步骤中综合多种不同的信息，提升对文本的理解和生成能力。
        4. 总结
        Q 可能对应多个 K 和 V，这使得模型能够通过计算注意力权重来选择哪些 K 和 V 对于给定的 Q 更重要。
        通过多头注意力，模型能够在同一时间并行学习不同的上下文依赖，从而提升理解和生成的能力。
        每个头通过独立的 Q、K、V 学习不同的信息，使得模型能够捕捉到更丰富的语义和上下文。
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

        '''
        LLM（大语言模型）使用多头注意力机制（比如 8 个注意力头）主要是为了增强模型对输入数据的理解能力和学习不同的上下文模式。选择使用 8 个注意力头的原因涉及多个方面：

        1. 并行处理多种关系
        每个注意力头有自己的查询、键和值的线性变换，因此每个头能够捕捉输入数据中的不同模式或依赖关系。通过多个头并行处理，模型能够在同一时刻学习到：
        
        短期依赖：例如，邻近词语之间的关系（如语法结构、词性标注等）。
        长期依赖：例如，跨句子或段落的上下文关系。
        语义关系：例如，理解词语的多义性或上下文中的特定意义。
        每个头专注于不同类型的依赖关系，这种并行化的设计可以帮助模型更全面地理解输入文本。
        
        2. 增强表示能力
        使用多个注意力头可以增强模型的表示能力。在单个注意力头下，模型只能捕捉到输入中某种特定类型的关系（例如，词语之间的直接依赖）。而通过多头注意力，模型能够在一个层次上并行地学习到不同的上下文信息（例如，语法、语义和长短期依赖）。
        
        例如，在理解一句话时：
        
        头1可能关注句法结构，如词汇顺序。
        头2可能关注词汇之间的远程关系。
        头3可能关注某些特定的词语的语义。
        这种多维度的学习方式可以使模型更强大、更具适应性。
        
        3. 计算效率与平衡
        选择 8 个头是一个折中的选择，它能够提供足够的表达能力，同时保持计算效率的平衡。更多的注意力头可以捕捉更多的关系，但也会增加计算和内存开销。8 个头通常足以在保持计算效率的前提下，获得良好的性能。
        
        计算开销：每增加一个头，计算量和内存开销都会增加。8 个头是一个相对较常见的配置，它在许多任务中能有效地发挥作用。
        表现能力：8 个头可以同时学习到足够多的不同依赖关系。增加更多的头数可能不会显著提升模型的表现，但会增加训练时间和内存消耗。
        4. 训练和优化的便利性
        多头注意力的设计和训练是高度优化的，尤其是在使用 8 个头时。对于许多预训练语言模型（如 GPT-3、BERT 等），8 个头的配置通常足够覆盖大多数语言任务的需求，同时确保训练过程中的稳定性和收敛速度。
        
        5. 经验法则和先前的成功
        许多研究和工程实践表明，8 个注意力头在多个自然语言处理任务上提供了良好的效果。例如，许多大型预训练模型，如 Transformer、BERT、GPT 等，都采用了 8 或 16 个注意力头作为基础配置。8 个头被认为是一个经过多次实验验证的良好配置，既能平衡性能和计算资源，又能提高模型的表达能力。
        
        6. 如何选择头的数量
        虽然 8 个头是常见的选择，但并不是固定的标准。在实际应用中，头的数量可以根据具体任务和计算资源进行调整：
        
        较小的模型：可能会使用 4 或 8 个头，以减少计算资源的消耗。
        较大的模型：可能会使用更多的头（如 16 或 32 个头），以进一步提升表达能力。
        总结
        使用 8 个头是为了：
        
        在计算资源和表现之间取得平衡；
        增强模型的表达能力，让其能够同时捕捉多种不同类型的依赖关系；
        保证计算效率和训练稳定性。
        通过这种设计，模型能够从不同的角度理解输入文本的语义和结构，增强模型的泛化能力。
        '''
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
