import torch
import torch.nn as nn
import torch.nn.functional as F

class RelMultiHeadAttn(nn.Module):
    def __init__(self, n_head, d_model, d_head, dropout, qkw_norm, dropatt=0,
                 tgt_len=None, ext_len=None, mem_len=None, pre_lnorm=False):
        super(RelMultiHeadAttn, self).__init__()

        self.n_head = n_head
        self.d_model = d_model
        self.d_head = d_head
        self.dropout = dropout
 
        self.qkv_net = nn.Linear(d_model, 3 * n_head * d_head, bias=False)

        self.drop = nn.Dropout(dropout)
        self.dropatt = nn.Dropout(dropatt)
        self.o_net = nn.Linear(n_head * d_head, d_model, bias=False)

        self.layer_norm = nn.LayerNorm(d_model)

        self.scale = 1 / (d_head ** 0.5)

        self.pre_lnorm = pre_lnorm
        self.qkw_norm = qkw_norm

    def _parallelogram_mask(self, h, w, left=False):
        mask = torch.ones((h, w)).byte()
        m = min(h, w)
        mask[:m,:m] = torch.triu(mask[:m,:m])
        mask[-m:,-m:] = torch.tril(mask[-m:,-m:])

        if left:
            return mask
        else:
            return mask.flip(0)

    def _shift(self, x, qlen, klen, mask, left=False):
        if qlen > 1:
            zero_pad = torch.zeros((x.size(0), qlen-1, x.size(2), x.size(3)),
                                    device=x.device, dtype=x.dtype)
        else:
            zero_pad = torch.zeros(0, device=x.device, dtype=x.dtype)

        if left:
            mask = mask.flip(1)
            x_padded = torch.cat([zero_pad, x], dim=1).expand(qlen, -1, -1, -1)
        else:
            x_padded = torch.cat([x, zero_pad], dim=1).expand(qlen, -1, -1, -1)

        x = x_padded.masked_select(mask[:,:,None,None]) \
                    .view(qlen, klen, x.size(2), x.size(3))

        return x

    def _rel_shift(self, x, zero_triu=False):
        zero_pad = torch.zeros((x.size(0), 1, *x.size()[2:]),
                               device=x.device, dtype=x.dtype)
        x_padded = torch.cat([zero_pad, x], dim=1)

        x_padded = x_padded.view(x.size(1) + 1, x.size(0), *x.size()[2:])

        x = x_padded[1:].view_as(x)

        if zero_triu:
            ones = torch.ones((x.size(0), x.size(1)))
            x = x * torch.tril(ones, x.size(1) - x.size(0))[:,:,None,None]

        return x

    def forward(self, w, r, attn_mask=None, mems=None):
        raise NotImplementedError

# # * DEFAULT
# class RelPartialLearnableMultiHeadAttn(RelMultiHeadAttn):
#     def __init__(self, *args, **kwargs):
#         super(RelPartialLearnableMultiHeadAttn, self).__init__(*args, **kwargs)

#         self.r_net = nn.Linear(self.d_model, self.n_head * self.d_head, bias=False)

#     def forward(self, w, r, r_w_bias, r_r_bias, attn_mask=None, mems=None):
#         qlen, rlen, bsz = w.size(0), r.size(0), w.size(1)
#         #print(qlen, rlen, bsz)

#         if mems is not None:
#             cat = torch.cat([mems, w], 0)
#             if self.pre_lnorm:
#                 w_heads = self.qkv_net(self.layer_norm(cat))
#             else:
#                 w_heads = self.qkv_net(cat)
#             r_head_k = self.r_net(r)

#             w_head_q, w_head_k, w_head_v = torch.chunk(w_heads, 3, dim=-1)
#             w_head_q = w_head_q[-qlen:]
#         else:
#             if self.pre_lnorm:
#                 w_heads = self.qkv_net(self.layer_norm(w))
#             else:
#                 w_heads = self.qkv_net(w)
#             r_head_k = self.r_net(r)

#             w_head_q, w_head_k, w_head_v = torch.chunk(w_heads, 3, dim=-1)

#         klen = w_head_k.size(0)
#         #print("11111111", klen)

#         w_head_q = w_head_q.view(qlen, bsz, self.n_head, self.d_head)           # qlen x bsz x n_head x d_head
#         w_head_k = w_head_k.view(klen, bsz, self.n_head, self.d_head)           # qlen x bsz x n_head x d_head
#         w_head_v = w_head_v.view(klen, bsz, self.n_head, self.d_head)           # qlen x bsz x n_head x d_head

#         r_head_k = r_head_k.view(rlen, self.n_head, self.d_head)                # qlen x n_head x d_head

#         #### compute attention score
#         rw_head_q = w_head_q + r_w_bias                                         # qlen x bsz x n_head x d_head
#         #print(rw_head_q.shape, w_head_k.shape)
#         AC = torch.einsum('ibnd,jbnd->ijbn', (rw_head_q, w_head_k))             # qlen x klen x bsz x n_head

#         rr_head_q = w_head_q + r_r_bias
#         BD = torch.einsum('ibnd,jnd->ijbn', (rr_head_q, r_head_k))              # qlen x klen x bsz x n_head
#         BD = self._rel_shift(BD)

#         # [qlen x klen x bsz x n_head]
#         #print(AC.shape, BD.shape)
#         attn_score = AC + BD
#         attn_score.mul_(self.scale)

#         #### compute attention probability
        # if attn_mask is not None and attn_mask.any().item():
        #     if attn_mask.dim() == 2:
        #         attn_score = attn_score.float().masked_fill(
        #             attn_mask[None,:,:,None].bool(), -float('inf')).type_as(attn_score)
        #     elif attn_mask.dim() == 3:
        #         attn_score = attn_score.float().masked_fill(
        #             attn_mask[:,:,:,None].bool(), -float('inf')).type_as(attn_score)
        
#         #########################################################
        
        
#         attn_weights = attn_score[:, :, 0, 0]
#         attn_weights = F.softmax(attn_weights, dim=1)
#         attn_weights = attn_weights.detach().cpu().numpy()
#         #self.attn_map = attn_weights
        
#         # import matplotlib.pyplot as plt
#         # fig, ax = plt.subplots(figsize=(5, 5))
#         # im = ax.imshow(attn_weights, cmap="magma")
#         # cax = fig.add_axes([ax.get_position().x1 + 0.01, ax.get_position().y0, 0.02, ax.get_position().height])
#         # cbar = fig.colorbar(im, cax=cax)
#         # plt.show()
#         #########################################################
        
#         #print("attn_score", attn_score.shape) # torch.Size([num_mem_tokens*2+context_length*3, num_mem_tokens*2+context_length*3+MEM_LEN, bs, d_head])

#         # [qlen x klen x bsz x n_head]
#         attn_prob = F.softmax(attn_score, dim=1)
#         #print(attn_prob.shape)
#         attn_prob = self.dropatt(attn_prob)

#         #### compute attention vector
#         attn_vec = torch.einsum('ijbn,jbnd->ibnd', (attn_prob, w_head_v))

#         # [qlen x bsz x n_head x d_head]
#         attn_vec = attn_vec.contiguous().view(
#             attn_vec.size(0), attn_vec.size(1), self.n_head * self.d_head)

#         ##### linear projection
#         attn_out = self.o_net(attn_vec)
#         attn_out = self.drop(attn_out)

#         if self.pre_lnorm:
#             ##### residual connection
#             output = w + attn_out
#         else:
#             ##### residual connection + layer normalization
#             output = self.layer_norm(w + attn_out)

#         return output, attn_weights


class RelPartialLearnableMultiHeadAttn(RelMultiHeadAttn):
    def __init__(self, *args, **kwargs):
        super(RelPartialLearnableMultiHeadAttn, self).__init__(*args, **kwargs)

        self.r_net = nn.Linear(self.d_model, self.n_head * self.d_head, bias=False)

    def forward(self, w, r, r_w_bias, r_r_bias, attn_mask=None, mems=None):
        qlen, rlen, bsz = w.size(0), r.size(0), w.size(1)

        #if using stable version, then want layernorm of memory as well before MHA
        # ! Need to make experiments wuth cat = torch.cat([mems, w, mems], 0) after
        # ! Need to make experiments with layer_norm and withoutw
        if mems is not None:
            cat = torch.cat([mems, w], 0)
            if self.qkw_norm: # if self.pre_lnorm:
                w_heads = self.qkv_net(self.layer_norm(cat))
            else:
                w_heads = self.qkv_net(cat)
            r_head_k = self.r_net(r)

            w_head_q, w_head_k, w_head_v = torch.chunk(w_heads, 3, dim=-1)
            w_head_q = w_head_q[-qlen:]
        else:
            if self.qkw_norm: #if self.pre_lnorm:
                w_heads = self.qkv_net(self.layer_norm(w))
            else:
                w_heads = self.qkv_net(w)
            
            r_head_k = self.r_net(r)

            w_head_q, w_head_k, w_head_v = torch.chunk(w_heads, 3, dim=-1)

        klen = w_head_k.size(0)

        w_head_q = w_head_q.view(qlen, bsz, self.n_head, self.d_head)           # qlen x bsz x n_head x d_head
        w_head_k = w_head_k.view(klen, bsz, self.n_head, self.d_head)           # qlen x bsz x n_head x d_head
        w_head_v = w_head_v.view(klen, bsz, self.n_head, self.d_head)           # qlen x bsz x n_head x d_head

        r_head_k = r_head_k.view(rlen, self.n_head, self.d_head)                # qlen x n_head x d_head

        #### compute attention score
        rw_head_q = w_head_q + r_w_bias                                         # qlen x bsz x n_head x d_head
        AC = torch.einsum('ibnd,jbnd->ijbn', (rw_head_q, w_head_k))             # qlen x klen x bsz x n_head

        rr_head_q = w_head_q + r_r_bias
        BD = torch.einsum('ibnd,jnd->ijbn', (rr_head_q, r_head_k))              # qlen x klen x bsz x n_head
        BD = self._rel_shift(BD)

        # [qlen x klen x bsz x n_head]
        attn_score = AC + BD
        attn_score.mul_(self.scale)

        #### compute attention probability
        if attn_mask is not None and attn_mask.any().item():
            if attn_mask.dim() == 2:
                attn_score = attn_score.float().masked_fill(
                    attn_mask[None,:,:,None], -float('inf')).type_as(attn_score)
            elif attn_mask.dim() == 3: #THIS IS WHAT IS Usually executed
                #print('Attentionscore shape: ',attn_score.shape)
                #print('MASK SHAPE: ', attn_mask[:,:,:,None].shape)
                #print('MASK EL 1: ', attn_mask[:,:,0])
                attn_score = attn_score.float().masked_fill(
                    attn_mask[:,:,:,None].bool(), -float('inf')).type_as(attn_score)
                
        # attn_weights = attn_score[:, :, 0, 0]
        # attn_weights = F.softmax(attn_weights, dim=1)
        # attn_weights = attn_weights.detach().cpu().numpy()
        attn_weights = attn_score.detach().cpu()

        #print('ATTENTION SCORE: ', attn_score)

        # [qlen x klen x bsz x n_head]
        attn_prob = F.softmax(attn_score, dim=1)
        attn_prob = self.dropatt(attn_prob)

        #### compute attention vector
        attn_vec = torch.einsum('ijbn,jbnd->ibnd', (attn_prob, w_head_v))

        # [qlen x bsz x n_head x d_head]
        attn_vec = attn_vec.contiguous().view(
            attn_vec.size(0), attn_vec.size(1), self.n_head * self.d_head)

        ##### linear projection
        attn_out = self.o_net(attn_vec)
        attn_out = self.drop(attn_out)

        # if self.pre_lnorm:
        #     ##### residual connection
        #     output = w + attn_out
        # else:
        #     ##### residual connection + layer normalization
        #     output = self.layer_norm(w + attn_out)

        output = w + attn_out

        return output, attn_weights

class RelLearnableMultiHeadAttn(RelMultiHeadAttn):
    def __init__(self, *args, **kwargs):
        super(RelLearnableMultiHeadAttn, self).__init__(*args, **kwargs)

    def forward(self, w, r_emb, r_w_bias, r_bias, attn_mask=None, mems=None):
        # r_emb: [klen, n_head, d_head], used for term B
        # r_w_bias: [n_head, d_head], used for term C
        # r_bias: [klen, n_head], used for term D

        qlen, bsz = w.size(0), w.size(1)

        if mems is not None:
            cat = torch.cat([mems, w], 0)
            if self.pre_lnorm:
                w_heads = self.qkv_net(self.layer_norm(cat))
            else:
                w_heads = self.qkv_net(cat)
            w_head_q, w_head_k, w_head_v = torch.chunk(w_heads, 3, dim=-1)

            w_head_q = w_head_q[-qlen:]
        else:
            if self.pre_lnorm:
                w_heads = self.qkv_net(self.layer_norm(w))
            else:
                w_heads = self.qkv_net(w)
            w_head_q, w_head_k, w_head_v = torch.chunk(w_heads, 3, dim=-1)

        klen = w_head_k.size(0)

        w_head_q = w_head_q.view(qlen, bsz, self.n_head, self.d_head)
        w_head_k = w_head_k.view(klen, bsz, self.n_head, self.d_head)
        w_head_v = w_head_v.view(klen, bsz, self.n_head, self.d_head)

        if klen > r_emb.size(0):
            r_emb_pad = r_emb[0:1].expand(klen-r_emb.size(0), -1, -1)
            r_emb = torch.cat([r_emb_pad, r_emb], 0)
            r_bias_pad = r_bias[0:1].expand(klen-r_bias.size(0), -1)
            r_bias = torch.cat([r_bias_pad, r_bias], 0)
        else:
            r_emb = r_emb[-klen:]
            r_bias = r_bias[-klen:]

        #### compute attention score
        rw_head_q = w_head_q + r_w_bias[None]                                   # qlen x bsz x n_head x d_head

        AC = torch.einsum('ibnd,jbnd->ijbn', (rw_head_q, w_head_k))             # qlen x klen x bsz x n_head
        B_ = torch.einsum('ibnd,jnd->ijbn', (w_head_q, r_emb))                  # qlen x klen x bsz x n_head
        D_ = r_bias[None, :, None]                                              # 1    x klen x 1   x n_head
        BD = self._rel_shift(B_ + D_)

        # [qlen x klen x bsz x n_head]
        attn_score = AC + BD
        attn_score.mul_(self.scale)

        #### compute attention probability
        if attn_mask is not None and attn_mask.any().item():
            if attn_mask.dim() == 2:
                attn_score.masked_fill_(attn_mask[None,:,:,None].bool(), -float('inf'))
            elif attn_mask.dim() == 3:
                attn_score.masked_fill_(attn_mask[:,:,:,None].bool(), -float('inf'))

        # [qlen x klen x bsz x n_head]
        attn_prob = F.softmax(attn_score, dim=1)
        attn_prob = self.dropatt(attn_prob)

        #### compute attention vector
        attn_vec = torch.einsum('ijbn,jbnd->ibnd', (attn_prob, w_head_v))

        # [qlen x bsz x n_head x d_head]
        attn_vec = attn_vec.contiguous().view(
            attn_vec.size(0), attn_vec.size(1), self.n_head * self.d_head)

        ##### linear projection
        attn_out = self.o_net(attn_vec)
        attn_out = self.drop(attn_out)

        if self.pre_lnorm:
            ##### residual connection
            output = w + attn_out
        else:
            ##### residual connection + layer normalization
            output = self.layer_norm(w + attn_out)

        return output