# Milti-Head Attention, Core mechanism of Transformer
#
# Input: [batch_size, seq_len, embed_dim]
# 
# in GPT3, embed_dim=12288(128*96)

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int,
                 dropout: float=0.0, bias: bool=True):
        super().__init__()
        # think why ?
        assert embed_dim % num_heads == 0,\
              "embed_dim must be divisiable by num_heads"
        self.emded_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim  = embed_dim // num_heads
        self.register_buffer("scale", torch.tensor(1 / math.sqrt(self.head_dim)))

        # 这里为什么要这么写, 输入输出的dim一样 ?
        # 实际上后面还会 .view, 这里只是把 "多头" 合并到一起了
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.dropout = nn.Dropout(dropout)

    '''
        query, key, value: (b, l_q, c), key: query by default(self-attention)
        attn_mask        :  
        key_padding_mask :  
        need_weights     :
    '''
    def forward(self,
            query: torch.Tensor,
            key  : torch.Tensor,
            value: torch.Tensor,
            attn_mask: torch.Tensor,
            key_padding_mask: torch.Tensor, # Bool, which place to add mask
            need_weights: bool=False,
    ):
        if key is None:
            key = value = query
        if value is None:
            value = key
        
        B, Lq, _ = query.shape
        Lk = key.shape[1]

        q = self.q_proj(query)  # W_q
        k = self.k_proj(key)    # W_k
        v = self.v_proj(value)  # W_v

        def _shape(x):
            return x.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        q, k, v = map(_shape, (q, k, v))
        # q:[B, num_heads, l_q, head_dim]

        attn_score = (q @ k.transpose(-2, -1) * self.scale)
        
        if attn_mask is not None:
            attn_score += attn_mask.unsqueeze(0).unsqueeze(0)
        if key_padding_mask is not None:
            attn_score = attn_score.masked_fill(
                key_padding_mask[:, None, None, :], float("-inf")
            )

        attn_probs = F.softmax(attn_score, dim=-1)
        attn_probs = self.dropout(attn_probs)

        context = attn_probs @ v

        context = (
            context.transpose(1, 2).contiguous()  # 确保内存连续, .view 可以正常操作
            .view(B, Lq, self.embed_dim)
        )

        out = self.out_proj(context)

        if need_weights:
            return out, attn_probs.mean(dim=1)
        return out


# How to use nn.MultiheadAttention ?

mha = nn.MultiheadAttention(embed_dim=512, num_heads=8, batch_first=True)
query = torch.randn(32, 10, 512)  #[B, len, embed_dim]
key = torch.randn(32, 20, 512)
value = key
output, attn_weights = mha(query, key, value)

class CrossAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, tgt, memory, key_padding_mask=None):
        attn_output, _ = self.mha(tgt, memory, memory, key_padding_mask=key_padding_mask)
        return self.norm(tgt + attn_output)
    
# 实际上，在pytorch官方源码中，qkv矩阵会被合并到一起运算
'''
    self.in_proj_weight = nn.Parameter(torch.empty(3 * self.d, self.d))
    self.in_proj_bias   = nn.Parameter(torch.empty(3 * self.d))
    self.out_proj       = nn.Linear(self.d, self.d, bias=True)

    def _qkv_linear(self, x: torch.Tensor):
        """
            一次 GEMM -> (B, L, 3·d) -> chunk
        """
        qkv = F.linear(x, self.in_proj_weight, self.in_proj_bias)      
        return qkv.chunk(3, dim=-1)      
'''
# 然后后面再 .view, 进行多头注意力计算
# 问题：对于自注意力和交叉注意力，下面代码的实现有什么区别？
'''
    def _qkv_linear(self, x: torch.Tensor):
        """
        *自注意力* 快捷路径：一次 GEMM -> (B, L, 3·d) -> chunk
        """
        qkv = F.linear(x, self.in_proj_weight, self.in_proj_bias)      # ① 只做一次
        return qkv.chunk(3, dim=-1)                                    # ② 切三段

    def _proj_slice(self, x: torch.Tensor, idx: int):
        """
        通用路径：给定 idx=0/1/2，切出 Wq/Wk/Wv 各自的 (d , d) 行块再做 GEMM
        """
        start = idx * self.d
        end   = start + self.d
        return F.linear(x,
                        self.in_proj_weight[start:end],
                        self.in_proj_bias[start:end])

    # -------------------------------------------------------
    def forward(self, query, key=None, value=None, need_weights=False, is_causal=False):
        if key is None:   # 自注意力：Q=K=V
            key = value = query
            q, k, v = self._qkv_linear(query)          # ◀▶ 一次 GEMM
        else:              # 交叉注意力：仍需两/三次 GEMM
            q = self._proj_slice(query, 0)              # Wq
            k = self._proj_slice(key,   1)              # Wk
            v = self._proj_slice(value, 2)              # Wv
'''