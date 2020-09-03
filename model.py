import math
import logging

import tensorflow as tf

layers = tf.keras.layers

logger = logging.getLogger(__name__) 

class GPTConfig:
    """ base GPT config, params common to all GPT versions """
    embd_pdrop = 0.1
    resid_pdrop = 0.1
    attn_pdrop = 0.1

    def __init__(self, vocab_size, block_size, **kwargs):
        self.vocab_size = vocab_size
        self.block_size = block_size
        for k,v in kwargs.items():
            setattr(self, k, v)

class GPT1Config(GPTConfig):
    """ GPT-1 like network roughly 125M params """
    n_layer = 12
    n_head = 12
    n_embd = 768


class CausalSelfAttention(tf.keras.Model):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    It is possible to use torch.nn.MultiheadAttention here but I am including an
    explicit implementation here to show that there is nothing too scary here.
    """

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads
        self.key   = layers.Dense(config.n_embd)
        self.query = layers.Dense(config.n_embd)
        self.value = layers.Dense(config.n_embd)
        # regularization
        self.attn_drop   = layers.Dropout(config.attn_pdrop)
        self.resid_drop  = layers.Dropout(config.resid_pdrop)
        # output projection
        self.proj = layers.Dense(config.n_embd)
        # causal mask to ensure that attention is only applied to the left in the input sequence


        self.mask = tf.linalg.LinearOperatorLowerTriangular(tf.ones((config.block_size, config.block_size))).to_dense()

        self.mask = tf.reshape(self.mask,(1, 1, config.block_size, config.block_size))

        self.n_head = config.n_head

    def call(self, x, layer_past=None):
        B, T, C = x.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = tf.reshape(self.key(x),(B, T, self.n_head, C // self.n_head)).transpose(1, 2) # (B, nh, T, hs)
        q = tf.reshape(self.query(x),(B, T, self.n_head, C // self.n_head)).transpose(1, 2) # (B, nh, T, hs)
        v = tf.reshape(self.value(x),(B, T, self.n_head, C // self.n_head)).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.shape(-1)))
        att = att.masked_fill(self.mask[:,:,:T,:T] == 0, float('-inf'))
        att = tf.nn.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).reshape(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))
        return y


config = GPT1Config(256,256)

causal = CausalSelfAttention(config)



# class Block(nn.Module):
#     """ an unassuming Transformer block """

#     def __init__(self, config):
#         super().__init__()
#         self.ln1 = nn.LayerNorm(config.n_embd)
#         self.ln2 = nn.LayerNorm(config.n_embd)
#         self.attn = CausalSelfAttention(config)
#         self.mlp = nn.Sequential(
#             nn.Linear(config.n_embd, 4 * config.n_embd),
#             nn.GELU(),
#             nn.Linear(4 * config.n_embd, config.n_embd),
#             nn.Dropout(config.resid_pdrop),
#         )

#     def forward(self, x):
#         x = x + self.attn(self.ln1(x))
#         x = x + self.mlp(self.ln2(x))
#         return x

