import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional

class attention(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=64, dropout=0.3, activation="relu"):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = nn.ReLU(activation)
        
    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(self, tgt: Tensor, template: Tensor, pos_enc1: Optional[Tensor] = None,
                     pos_dec1: Optional[Tensor] = None, pos_enc2: Optional[Tensor] = None) -> Tensor:

        tgt2 = self.multihead_attn(self.with_pos_embed(tgt, pos_dec1), self.with_pos_embed(template, pos_enc1), template)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        return tgt

class CrossAttention(nn.Module):
    def __init__(self, device, channel=8):
        super().__init__()
        self.device = device
        self.layer=attention(channel,8)
        self.row_embed = nn.Embedding(500, channel//2).to(self.device)
        self.col_embed = nn.Embedding(500, channel//2).to(self.device)
    def pos_embedding(self,x):
        h, w = x.shape[-2:]
        # print(h,w)
        i = torch.arange(w).to(self.device)
        j = torch.arange(h).to(self.device)
        # i = torch.arange(w)
        # j = torch.arange(h)
        x_emb = self.col_embed(i)
        y_emb = self.row_embed(j)
        pos = torch.cat([
            x_emb.unsqueeze(0).repeat(h, 1, 1),
            y_emb.unsqueeze(1).repeat(1, w, 1),
        ], dim=-1).permute(2, 0, 1).unsqueeze(0).repeat(x.shape[0], 1, 1, 1)
        pos=pos.flatten(2).permute(2,0,1)
        # print(pos.shape)
        return pos

    def forward(self,search,template):
        B, C, H, W = search.shape
        pos_1=self.pos_embedding(search)
        pos_2=self.pos_embedding(template)
        search=search.flatten(2).permute(2,0,1)
        template=template.flatten(2).permute(2,0,1)
        # print(search.shape)
        # print(template.shape)
        x=self.layer(tgt=search,template=template, pos_dec1=pos_1, 
                        pos_enc1=pos_2)
        x = x.permute(1,2,0).view(B, C, H, W)
        return x