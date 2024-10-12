import sys
import os

import math

import numpy as np

import torch
import torch.nn as nn
import torch.functional as F

import copy

class MultiHeadedAttention(nn.Module):
    def __init__(self, seq_length):
        super(MultiHeadedAttention, self).__init__()
        
    def forward(self):
        pass
    

class MultiVarInputEncoding(nn.Module):
    def __init__(self, n_unique_tokens, n_embed_dims, n_features, seq_len, padding_token = None, dropout = 0.2):
        """
        Token and positional embedding module for multivariate time series, preparation for the 
        tranformer decoder layer.

        Args:
            n_unique_tokens (int): the number of unique tokens, or vocabulary size
            n_embed_dims (int): the dimensionality of the embedding vector
            seq_len (int): the standard length of the input sequences
            padding_token (int | float, optional): the value of the padding token. Defaults to None.
            dropout (float, optional): the ratio of dropout. Defaults to 0.2.
        """
        super(MultiVarInputEncoding, self).__init__()
        
        self.n_unique_tokes = n_unique_tokens
        self.n_embed_dims = n_embed_dims
        self.seq_length = seq_len
        self.n_features = n_features
        
        # embedding layer
        self.token_embed_table = nn.Embedding(n_unique_tokens, n_embed_dims, padding_idx = padding_token)
        
        # feature embedding (NOTE: NOT USED RIGHT NOW)
        self.feature_embedding = nn.Linear(n_embed_dims * n_features, n_embed_dims)
        
        # positional embedding
        self.pos_encoding = torch.zeros(seq_len, n_embed_dims * n_features)
        pos = torch.arange(0, seq_len, dtype = torch.float32).unsqueeze(1)
        denom = torch.exp(torch.arange(0, n_embed_dims * n_features, 2) * (-np.log(10000.0) / (n_embed_dims * n_features)))
        # apply sin to even, cos to uneven
        self.pos_encoding[:, 0::2] = torch.sin(pos * denom)
        self.pos_encoding[:, 1::2] = torch.cos(pos * denom)
        # add batch dim
        self.pos_encoding = self.pos_encoding.unsqueeze(0)
        self.pos_encoding.requires_grad = False
        
        # final dropout
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, src, tgt):
        # token embedding: (b, t, f) -> (b, t, f, e) 
        src_tok_embeds = self.token_embed_table(src)
        tgt_tok_embeds = self.token_embed_table(tgt)
        # get shape (batch, timesteps, features, embed_dims)
        b, t, f, e = src_tok_embeds.shape
        # reshape: -> (b, t, f * e)
        src_tok_embeds = src_tok_embeds.view((b, t, f * e))
        tgt_tok_embeds = tgt_tok_embeds.view((b, t, f * e))
        # feature embedding: -> (b, t, e) NOTE NOT USED
        # src_tok_embeds = self.feature_embedding(src_tok_embeds)
        
        # multiplication of square root for stability (from og paper)
        src_scaled_embeds = src_tok_embeds * np.sqrt(f * e)
        tgt_scaled_embeds = tgt_tok_embeds * np.sqrt(f * e)
        # add postional encoding
        src_encoded_embeds = src_scaled_embeds + self.pos_encoding[:, : t, :]
        tgt_encoded_embeds = tgt_scaled_embeds + self.pos_encoding[:, : t, :]
        # through dropout layer, helps prevent overfitting to singular tokens
        src_encoded_input = self.dropout(src_encoded_embeds)
                
        return src_encoded_input, tgt_encoded_embeds


class MultiVarTransformer(nn.Module):
    def __init__(self, n_unique_tokens, n_features, seq_len, n_embed_dims = 64, n_heads = 8, n_layers = 4, ff_dim = 1024, padding_token = None):
        super(MultiVarTransformer, self).__init__()
        
        self.n_unique_tokens = n_unique_tokens
        self.padding_token = padding_token
        
        # input embedding encoding 
        self.input_encoder = MultiVarInputEncoding(n_unique_tokens, n_embed_dims, n_features, seq_len, padding_token)
        # transformer layer
        self.transformer = nn.Transformer(n_embed_dims * n_features, n_heads, n_layers, n_layers, ff_dim, batch_first = True)
        # output layer
        self.output_linear = nn.Linear(n_embed_dims * n_features, n_unique_tokens * n_features)
        
    def get_padding_masks(self, src, tgt):
        # if there is a padding token, we create padding masks
        if self.padding_token is not None:
            src_pad_mask = (src == self.padding_token).all(dim=2).float()
            tgt_pad_mask = (tgt == self.padding_token).all(dim=2).float()
            src_pad_mask = src_pad_mask.masked_fill(src_pad_mask == 1., float('-inf')) 
            tgt_pad_mask = tgt_pad_mask.masked_fill(tgt_pad_mask == 1., float('-inf')) 
        else:
            src_pad_mask = None
            tgt_pad_mask = None
            
        return src_pad_mask, tgt_pad_mask
        
    def forward(self, src, tgt):
        # get shape (batch, timesteps, features)
        b, t, f = src.shape
        
        # apply input encoding: (b, t, f) -> (b, t, f * e)
        encoded_src, encoded_tgt = self.input_encoder(src, tgt)
        
        # create non-lookahead mask
        non_lookahead_mask = self.transformer.generate_square_subsequent_mask(t)
        # get padding masks
        src_pad_mask, tgt_pad_mask = self.get_padding_masks(src, tgt)
            
        # apply transformer: -> (b, t, e)
        output = self.transformer(encoded_src, 
                                  encoded_tgt, 
                                  tgt_mask = non_lookahead_mask,
                                  src_key_padding_mask = src_pad_mask,
                                  tgt_key_padding_mask = tgt_pad_mask)
        # output layer: -> (b, t, n_unique * f)
        output = self.output_linear(output)
        # rescale: -> (b, n_unique, t, f), this shape for CrossEntr.Loss
        output = output.view((b, self.n_unique_tokens, t, f))
        
        return output

# class MultiVarTransformerEncoder(nn.Module):
#     def __init__(self, n_unique_tokens, n_embed_dims, n_features, seq_len):
#         super(MultiVarTransformerEncoder, self).__init__()
        
#         self.n_features = n_features
#         self.n_unique_tokens = n_unique_tokens
        
#         # transformer layer
#         self.transformer = nn.Transformer(n_embed_dims, 8, 4, 4, 1024)
#         # output layer
#         self.output_linear = nn.Linear(n_embed_dims, n_unique_tokens * n_features)
        
#     def forward(self, src, tgt):
#         # apply transformer: (b, t, f * e) -> (b, t, e)
#         output = self.transformer(src, tgt)
#         # output layer: -> (b, t, n_unique * f)
#         output = self.output_linear(output)
#         # rescale: -> (b, t, f, n_unique)
#         b, t, _ = output.shape
#         output = output.view((b, t, self.n_features, self.n_unique_tokens))
        
#         return output
        
        
            
# class MultiVarTransformerDecoder(nn.Module):
#     def __init__(self):
#         super(MultiVarTransformerDecoder, self).__init__()

#     def forward(self):
#         pass        
        