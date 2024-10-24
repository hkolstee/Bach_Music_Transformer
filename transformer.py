import sys
import os

import math

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiVarInputEncoding(nn.Module):
    def __init__(self, n_unique_tokens, n_embed_dims, n_features, seq_len, padding_token = None, dropout = 0.2):
        super(MultiVarInputEncoding, self).__init__()
        
        # we need device for pos_encoding 
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        self.n_unique_tokes = n_unique_tokens
        self.n_embed_dims = n_embed_dims
        self.seq_length = seq_len
        self.n_features = n_features
        
        # embedding layer
        self.token_embed_table = nn.Embedding(n_unique_tokens, n_embed_dims, padding_idx = padding_token)
        
        # feature embedding (NOTE: NOT USED RIGHT NOW)
        self.feature_embedding = nn.Linear(n_embed_dims * n_features, n_embed_dims)
        
        # positional embedding
        self.pos_encoding = torch.zeros(seq_len, n_embed_dims * n_features).to(device)
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


class MultiVarTransformerEncoder(nn.Module):
    def __init__(self, input_dim, n_head, ff_dim, attn_dropout, ff_dropout, batch_first = True):
        super(MultiVarTransformerEncoder, self).__init__()
        # attention layer
        self.self_attention = nn.MultiheadAttention(input_dim, n_head, dropout = attn_dropout, batch_first = batch_first)
        # layer_norm
        self.layer_norm_1 = nn.LayerNorm(input_dim)
        
        # ff layers
        self.linear_1 = nn.Linear(input_dim, ff_dim)
        self.linear_2 = nn.Linear(ff_dim, input_dim)
        # feedforward dropout
        self.ff_dropout = nn.Dropout(ff_dropout)
        # layer norm
        self.layer_norm_2 = nn.LayerNorm(input_dim)
        
    def forward(self, src, future_mask, pad_mask):
        # attention output
        self_attention, _ = self.self_attention(src, src, src, attn_mask = future_mask, key_padding_mask = pad_mask)
        # add attn info to source sequence
        src = src + self_attention
        # layer norm
        src = self.layer_norm_1(src)
        
        # feedforward layers
        ff_out = self.linear_2(self.ff_dropout(F.relu(self.linear_1(src))))
        # add residual connection
        src = src + ff_out
        # layer norm
        src = self.layer_norm_2(src)
        
        return src
    
    
class MultiVarTransformerDecoder(nn.Module):
    def __init__(self, input_dim, n_head, ff_dim, attn_dropout, ff_dropout, batch_first = True):
        super(MultiVarTransformerDecoder, self).__init__()
        # attention layer
        self.self_attention = nn.MultiheadAttention(input_dim, n_head, dropout = attn_dropout, batch_first = batch_first)
        # layer norm
        self.layer_norm_1 = nn.LayerNorm(input_dim)
        # cross attention with encoded input
        self.cross_attention = nn.MultiheadAttention(input_dim, n_head, dropout = attn_dropout, batch_first = batch_first)
        # layer norm
        self.layer_norm_2 = nn.LayerNorm(input_dim)
        
        # ff layers
        self.linear_1 = nn.Linear(input_dim, ff_dim)
        self.linear_2 = nn.Linear(ff_dim, input_dim)
        # feedforward dropout
        self.ff_dropout = nn.Dropout(ff_dropout)
        # layer norm
        self.layer_norm_3 = nn.LayerNorm(input_dim)
        
    def forward(self, tgt, encoded_src, future_mask, src_pad_mask, tgt_pad_mask):
        # self attention
        self_attention, _ = self.self_attention(tgt, tgt, tgt, attn_mask = future_mask, key_padding_mask = tgt_pad_mask)
        # add attention info to target sequence
        tgt = tgt + self_attention
        # layer norm
        tgt = self.layer_norm_1(tgt)
        
        # cross attention
        cross_attention, _ = self.cross_attention(tgt, encoded_src, encoded_src, attn_mask = future_mask, key_padding_mask = src_pad_mask)
        # add cross attention to target sequence
        tgt = tgt + cross_attention
        # layer norm
        tgt = self.layer_norm_2(tgt)
        
        # feedforward layers
        ff_out = self.linear_2(self.ff_dropout(F.relu(self.linear_1(tgt))))
        # residual connection
        tgt = tgt + ff_out
        # layer norm
        tgt = self.layer_norm_3(tgt)
        
        return tgt
    

class MultiVarTransformer(nn.Module):
    def __init__(self, 
                 n_unique_tokens, 
                 n_features, 
                 seq_len, 
                 n_embed_dims = 64, 
                 n_heads = 8, 
                 n_layers = 4, 
                 ff_dim = 1024, 
                 embed_dropout = 0.1,
                 attn_dropout = 0.2,
                 ff_dropout = 0.3,
                 padding_token = None):
        """A transformer for discrete multi-variate sequence to sequence tasks.

        Args:
            n_unique_tokens (int): vocabulary size / nr of classes
            n_features (int): the number of features, meaning of concurrent sequences
            seq_len (int): length of the input/output sequences
            n_embed_dims (int, optional): Encoder embedding demensions. Defaults to 64.
            n_heads (int, optional): number of heads for multi-attention. Defaults to 8.
            n_layers (int, optional): number of encoder/decoder layers. Defaults to 4.
            ff_dim (int, optional): dimension of the feedforward layer. Defaults to 1024.
            embed_dropout (float, optional): ratio of dropout applied after input embedding. 
                Defaults to 0.1.
            attn_dropout (float, optional): ratio of dropout applied after attention mechanism. 
                Defaults to 0.2.
            ff_dropout (float, optional): ratio of dropout applied after the feedforward layers. 
                Defaults to 0.3.
            padding_token (int, optional): the value of the padding token. Defaults to None.
        """
        super(MultiVarTransformer, self).__init__()
        
        self.n_unique_tokens = n_unique_tokens
        self.padding_token = padding_token
        self.d_model = n_embed_dims * n_features
        
        # we need device for pos_encoding 
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        # input embedding encoding 
        self.input_encoder = MultiVarInputEncoding(n_unique_tokens, n_embed_dims, n_features, seq_len, 
                                                   padding_token, dropout = embed_dropout)
        
        # Transformer encoder layers
        self.transformer_encoder = nn.ModuleList([
            MultiVarTransformerEncoder(n_embed_dims * n_features, 
                                       n_heads, 
                                       ff_dim, 
                                       attn_dropout, 
                                       ff_dropout)
            for _ in range(n_layers)
        ])
        
        # Transformer decoder layers
        self.transformer_decoder = nn.ModuleList([
            MultiVarTransformerDecoder(n_embed_dims * n_features, 
                                       n_heads, 
                                       ff_dim, 
                                       attn_dropout, 
                                       ff_dropout)
            for _ in range(n_layers)
        ])
        
        # output layer
        self.output_linear = nn.Linear(n_embed_dims * n_features, n_unique_tokens * n_features)
        
    def get_padding_masks(self, src, tgt):
        # if there is a padding token, we create padding masks
        if self.padding_token is not None:
            src_pad_mask = (src == self.padding_token).all(dim=2).float()
            tgt_pad_mask = (tgt == self.padding_token).all(dim=2).float()
            src_pad_mask = src_pad_mask.masked_fill((tgt_pad_mask == 1.), float('-inf')).to(self.device)
            tgt_pad_mask = tgt_pad_mask.masked_fill((tgt_pad_mask == 1.), float('-inf')).to(self.device)
        else:
            src_pad_mask = None
            tgt_pad_mask = None
            
        return src_pad_mask, tgt_pad_mask
    
    def apply_layers(self, layers, x, *args):
        # sequentially forward x through layers, taking variable args
        for layer in layers:
            x = layer(x, *args)
        return x
        
    def forward(self, src, tgt):
        # get shape (batch, timesteps, features)
        b, t, f = src.shape
        
        # apply input embedding + pos encoding: (b, t, f) -> (b, t, f * e)
        encoded_input_src, encoded_input_tgt = self.input_encoder(src, tgt)
        
        # create non-lookahead mask
        non_lookahead_mask = nn.Transformer.generate_square_subsequent_mask(t).to(self.device)
        # get padding masks
        src_pad_mask, tgt_pad_mask = self.get_padding_masks(src, tgt)
        
        # Pass through all encoder layers
        encoder_output = self.apply_layers(self.transformer_encoder, encoded_input_src, non_lookahead_mask, src_pad_mask)

        # Pass through all decoder layers (-> b, t, e * f)
        decoder_output = self.apply_layers(self.transformer_decoder, encoded_input_tgt, encoder_output, non_lookahead_mask, src_pad_mask, tgt_pad_mask)
    
        # output layer: -> (b, t, n_unique * f)
        output = self.output_linear(decoder_output)
        # rescale: -> (b, n_unique, t, f), this shape for CrossEntr.Loss
        output = output.view((b, self.n_unique_tokens, t, f))
        
        return output
