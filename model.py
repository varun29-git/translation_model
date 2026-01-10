# Dependencies
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

#---------------------------------------------------------------------------------------------------------------------------
# Input Embedding Class

class InputEmbeddings(nn.Module):

    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        # Embedding lookup table: maps each token ID to a learnable vector representation
        self.embedding = nn.Embedding(vocab_size, d_model)
    
    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)

#---------------------------------------------------------------------------------------------------------------------------
# Positional Embedding Class

class PositionalEmbedding(nn.Module):

    def __init__(self, d_model, seq_len, dropout):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)

        # seq_len defines how many tokens are processed together as one input sequence
        pe = torch.zeros(seq_len, d_model)
        # Creating a vector of shape (seq_len, 1)
        position = torch.arange(0, seq_len, dtype=float).unsqueeze(1) 
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -math.log(10000.0)/ d_model)
        # Apply the sin to even positions
        pe[:, 0::2] = torch.sin(position * div_term)
        # Apply the cosing to odd positions
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0) # (1, seq_l, d_model)

        self.register_buffer("pe", pe)
    
    def forward(self, x):
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False)
        return self.dropout(x)
    
#---------------------------------------------------------------------------------------------------------------------------
# Layer Normalisation Class

class LayerNorm(nn.Module):

    def __init__(self, d_model, epsilon = 10 ** -6):
        super().__init__()
        self.epsilon = epsilon
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.bias = nn.Parameter(torch.zeros(d_model))
    
    def forward(self, x):
        mean = x.mean(dim = -1, keepdim = True)
        std = x.std(dim = -1, keepdim = True)
        return self.gamma * (x - mean)/(std + self.epsilon) + self.bias

#---------------------------------------------------------------------------------------------------------------------------
# Feed Forward Class

class FeedForwardBlock(nn.Module):
    # FFN(x) = max(0, W1x + b1)W2 + b2  
    def __init__(self, d_model, dff, dropout):
        super().__init__()
        self.Linear_1 = nn.Linear(d_model, dff) # W1 and b1
        self.dropout = nn.Dropout(dropout)
        self.Linear_2 = nn.Linear(dff, d_model) # W2 and b2
    
    def forward(self, x):
        # (Batch, Seq_len, D_model) -> (Batch, Seq_len, dff) -> (Batch, Seq_len, D_model)
        return self.Linear_2(self.dropout(torch.relu(self.Linear_1(x))))
    
#---------------------------------------------------------------------------------------------------------------------------
# Multi Head Attention Class

class MultiHeadAttention(nn.Module):

    def __init__(self, d_model: int, h: int, dropout: float):
        super().__init__()
        self.d_model = d_model
        self.h = h
        assert d_model % h == 0, "d_model is not divisible by h"
        self.d_k = d_model // h

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
   
    @staticmethod
    def Attention(query, key, values, mask, Dropout: nn.Dropout):
        d_k = query.shape[-1]
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)

        if mask is not None:
            attention_scores.masked_fill_(mask == 0, -1e20)
        attention_scores = attention_scores.softmax(dim = -1) # ((Q @ K.T)/ d_model  ** 0.5)

        if Dropout is not None:
            attention_scores = Dropout(attention_scores)
        
        return (attention_scores @ values), attention_scores

    def forward(self, q, k, v, mask):
        query = self.w_q(q) # (Batch, Seq_len, D_model) -> (Batch, Seq_len, D_model)
        key = self.w_k(k) # (Batch, Seq_len, D_model) -> (Batch, Seq_len, D_model)
        value = self.w_v(v) # (Batch, Seq_len, D_model) -> (Batch, Seq_len, D_model)

        # Split into smaller matrixes
        # (Batch, Seq_len, D_model) -> (Batch, Seq_len, h, d_k) -> (Batch, h, Seq_len, d_k)
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1, 2)

        x, self.attention_scores = MultiHeadAttention.Attention(query, key, value, mask, self.dropout)
        x = x.transpose(1,2).contiguous().view(x.shape[0], -1, self.h * self.d_k)

        return self.w_o(x)

#---------------------------------------------------------------------------------------------------------------------------
# Residual Connection Class

class ResidualConnection(nn.Module):

    def __init__(self, d_model, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNorm(d_model)
    
    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))
    
#---------------------------------------------------------------------------------------------------------------------------
# Encoder Block Class

class EncoderBlock(nn.Module):

    def __init__(self, d_model, self_attention_block: MultiHeadAttention, feed_forward_block: FeedForwardBlock, dropout):
        super().__init__()
        self.self_attention = self_attention_block
        self.feed_forward = feed_forward_block
        self.dropout = nn.Dropout(dropout)
        self.residual_connections = nn.ModuleList([ResidualConnection(d_model, dropout) for _ in range(2)])

    def forward(self, x, src_mask):
        x = self.residual_connections[0](x, lambda x: self.self_attention(x, x, x, src_mask))
        x = self.residual_connections[1](x, self.feed_forward)
        return x
    
#---------------------------------------------------------------------------------------------------------------------------
# Encoder Class

class Encoder(nn.Module):

    def __init__(self, d_model, layers: nn.ModuleList):
        super().__init__()
        self.layers = layers
        self.norm = LayerNorm(d_model)
    
    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

#---------------------------------------------------------------------------------------------------------------------------
# Decoder Block Class

class DecoderBlock(nn.Module):

    def __init__(self, d_model, self_attention : MultiHeadAttention, cross_attention : MultiHeadAttention, feed_forward : FeedForwardBlock, dropout):
        super().__init__()
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.residual_connections = nn.ModuleList([ResidualConnection(d_model, dropout) for _ in range(3)])
        self.feed_forward = feed_forward
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, encoder_output, src_mask, tgt_mask):
        x = self.residual_connections[0](x, lambda x : self.self_attention(x, x, x, tgt_mask))
        x = self.residual_connections[1](x, lambda x : self.cross_attention(x, encoder_output, encoder_output, src_mask))
        x = self.residual_connections[2](x, self.feed_forward)
        return x
#---------------------------------------------------------------------------------------------------------------------------
# Decoder Class

class Decoder(nn.Module):

    def __init__(self, d_model, layers: nn.ModuleList):
        super().__init__()
        self.layers = layers
        self.norm = LayerNorm(d_model)
    
    def forward(self, x, encoder_output, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return self.norm(x)

#---------------------------------------------------------------------------------------------------------------------------
# Projection Class

class ProjectionLayer(nn.Module):

    def __init__(self, d_model, vocab_size):
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        return torch.log_softmax(self.proj(x), dim=-1)

#---------------------------------------------------------------------------------------------------------------------------
# Transformer Class

class Transformer(nn.Module):

    def __init__(self, encoder: Encoder, decoder: Decoder, src_embed: InputEmbeddings, tar_embed: InputEmbeddings, src_pos: PositionalEmbedding, tar_pos: PositionalEmbedding, projection: ProjectionLayer):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tar_embed = tar_embed
        self.src_pos = src_pos
        self.tar_pos = tar_pos
        self.projection = projection
    
    def encode(self, src, src_mask):
        src = self.src_embed(src)
        src = self.src_pos(src)
        return self.encoder(src, src_mask)
    
    def decode(self, encoder_output, src_mask, tar, tar_mask):
        tar = self.tar_embed(tar)
        tar = self.tar_pos(tar)
        return self.decoder(tar, encoder_output, src_mask,tar_mask)
    
    def project(self, x):
        return self.projection(x)
    
#---------------------------------------------------------------------------------------------------------------------------
# Function to build a Transformer

def build_transformer(src_vocab_size, tar_vocab_size, src_seq_len, tar_seq_len, d_model=512, N=6, h=8, dropout=0.1, d_ff=2048):

    src_embed = InputEmbeddings(src_vocab_size, d_model)
    tar_embed = InputEmbeddings(tar_vocab_size, d_model)
    src_pos = PositionalEmbedding(d_model, src_seq_len, dropout)
    tar_pos = PositionalEmbedding(d_model, tar_seq_len, dropout)

    encoder_blocks = []
    for _ in range(N):
        encoder_self_attention_block =  MultiHeadAttention(d_model, h, dropout)
        encoder_feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        encoder_block = EncoderBlock(d_model, encoder_self_attention_block, encoder_feed_forward_block, dropout)
        encoder_blocks.append(encoder_block)
    
    decoder_blocks = []
    for _ in range(N):
        decoder_self_attention_block = MultiHeadAttention(d_model, h, dropout)
        decoder_cross_attention_block = MultiHeadAttention(d_model, h, dropout)
        decoder_feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        decoder_block = DecoderBlock(d_model, decoder_self_attention_block, decoder_cross_attention_block, decoder_feed_forward_block, dropout)
        decoder_blocks.append(decoder_block)
    
    encoder = Encoder(d_model, nn.ModuleList(encoder_blocks))
    decoder = Decoder(d_model, nn.ModuleList(decoder_blocks))
    projection_layer = ProjectionLayer(d_model, tar_vocab_size)

    # Transformer
    transformer = Transformer(encoder, decoder, src_embed, tar_embed, src_pos, tar_pos, projection_layer)

    # Initialize Parameters
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    
    return transformer

#---------------------------------------------------------------------------------------------------------------------------





