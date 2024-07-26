import torch
import torch.nn as nn

class PositionalEmbedding(nn.Module):
    def __init__(self, demb):
        super(PositionalEmbedding, self).__init__()

        self.demb = demb

        inv_freq = 1 / (10000 ** (torch.arange(0.0, demb, 2.0) / demb))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, pos_seq, bsz=None):
        #print("pos seq", pos_seq.shape)
        sinusoid_inp = torch.ger(pos_seq, self.inv_freq)
        pos_emb = torch.cat([sinusoid_inp.sin(), sinusoid_inp.cos()], dim=-1)

        if bsz is not None:
            #print("pos case 1", pos_emb[:,None,:].expand(-1, bsz, -1).shape)
            return pos_emb[:,None,:].expand(-1, bsz, -1)
        else:
            #print("pos case 2", pos_emb[:, None, :].shape)
            return pos_emb[:,None,:]