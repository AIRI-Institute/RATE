import torch
import torch.nn as nn
from torch.nn.modules.normalization import LayerNorm
import torch.nn.functional as F

from RATE_GTrXL.blocks.Attention import RelPartialLearnableMultiHeadAttn#, RelLearnableMultiHeadAttn #MultiHeadAttn
from RATE_GTrXL.blocks.PositionwiseFF import PositionwiseFF

############################# GATED #####################
'''
GRU gating layer used in Stabilizing transformers in RL.
Note that all variable names follow the notation from section: "Gated-Recurrent-Unit-type gating" 
in https://arxiv.org/pdf/1910.06764.pdf
''' 

# * OLD GATING
class GRUGate(nn.Module):

    def __init__(self,d_model):
        #d_model is dimension of embedding for each token as input to layer (want to maintain this in the gate)
        super(GRUGate, self).__init__()

        # TODO: DEBUG Make sure intitialize bias of linear_w_z to -3
        self.linear_w_r = nn.Linear(d_model,d_model,bias=False)
        self.linear_u_r = nn.Linear(d_model,d_model,bias=False)
        self.linear_w_z = nn.Linear(d_model,d_model)               ### Giving bias to this layer (will count as b_g so can just initialize negative)
        self.linear_u_z = nn.Linear(d_model, d_model,bias=False)
        self.linear_w_g = nn.Linear(d_model, d_model,bias=False)
        self.linear_u_g = nn.Linear(d_model, d_model,bias=False)

        self.init_bias()

    def init_bias(self):
        with torch.no_grad():
            self.linear_w_z.bias.fill_(-2)  # Manually setting this bias to allow starting with markov process
            # Note -2 is the setting used in the paper stable transformers

    def forward(self,x,y):
        ### Here x,y follow from notation in paper
        # TODO: DEBUG MAKE SURE THIS IS APPLIED ON PROPER AXIS
        z = torch.sigmoid(self.linear_w_z(y) + self.linear_u_z(x))  #MAKE SURE THIS IS APPLIED ON PROPER AXIS
        r = torch.sigmoid(self.linear_w_r(y) + self.linear_u_r(x))
        h_hat = torch.tanh(self.linear_w_g(y) + self.linear_u_g(r*x))  #Note elementwise multiplication of r and x
        return (1.-z)*x + z*h_hat

# https://opendilab.github.io/DI-engine/_modules/ding/torch_utils/network/gtrxl.html#GTrXL.forward
class GRUGatingUnit(torch.nn.Module):
    """
    Overview:
        The GRUGatingUnit module implements the GRU gating mechanism used in the GTrXL model.
    Interfaces:
        ``__init__``, ``forward``
    """
    
    def __init__(self, input_dim: int, bg: float = 2.):
        """
        Overview:
            Initialize the GRUGatingUnit module.
        Arguments:
            - input_dim (:obj:`int`): The dimensionality of the input.
            - bg (:obj:`bg`): The gate bias. By setting bg > 0 we can explicitly initialize the gating mechanism to \
                be close to the identity map. This can greatly improve the learning speed and stability since it \
                initializes the agent close to a Markovian policy (ignore attention at the beginning).
        """

        super(GRUGatingUnit, self).__init__()
        self.Wr = torch.nn.Linear(input_dim, input_dim, bias=False)
        self.Ur = torch.nn.Linear(input_dim, input_dim, bias=False)
        self.Wz = torch.nn.Linear(input_dim, input_dim, bias=False)
        self.Uz = torch.nn.Linear(input_dim, input_dim, bias=False)
        self.Wg = torch.nn.Linear(input_dim, input_dim, bias=False)
        self.Ug = torch.nn.Linear(input_dim, input_dim, bias=False)
        self.bg = nn.Parameter(torch.full([input_dim], bg))  # bias
        self.sigmoid = torch.nn.Sigmoid()
        self.tanh = torch.nn.Tanh()
        
    def forward(self, x: torch.Tensor, y: torch.Tensor):
        """
        Overview:
            Compute the output value using the GRU gating mechanism.
        Arguments:
            - x: (:obj:`torch.Tensor`): The first input tensor.
            - y: (:obj:`torch.Tensor`): The second input tensor. \
                x and y should have the same shape and their last dimension should match the input_dim.
        Returns:
            - g: (:obj:`torch.Tensor`): The output of the GRU gating mechanism. \
                The shape of g matches the shapes of x and y.
        """
        #print("I'M GATING LAYER AND I'M WORKING")

        r = self.sigmoid(self.Wr(y) + self.Ur(x))
        z = self.sigmoid(self.Wz(y) + self.Uz(x) - self.bg)
        h = self.tanh(self.Wg(y) + self.Ug(torch.mul(r, x)))  # element wise multiplication
        g = torch.mul(1 - z, x) + torch.mul(z, h)
        return g  # x.shape == y.shape == g.shape
    
class RelPartialLearnableDecoderLayer(nn.Module):
    def __init__(self, n_head, d_model, d_head, d_inner, dropout, use_gate, use_stable_version, qkw_norm,
                 **kwargs):
        super(RelPartialLearnableDecoderLayer, self).__init__()

        self.use_gate = use_gate
        self.use_stable_version = use_stable_version

        if self.use_gate:
            self.gate_mha = GRUGatingUnit(d_model) # GRUGate(d_model)
            self.gate_mlp = GRUGatingUnit(d_model) # GRUGate(d_model)

        # self.norm1 = LayerNorm(d_model)
        # self.norm2 = LayerNorm(d_model)


        self.dec_attn = RelPartialLearnableMultiHeadAttn(n_head, d_model,
                            d_head, dropout, qkw_norm, **kwargs)
        self.pos_ff = PositionwiseFF(d_model, d_inner, dropout, use_gate)
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)

    def forward_orig(self, dec_inp, r, r_w_bias, r_r_bias, dec_attn_mask=None, mems=None):

        # print('1', dec_inp[0:5, 0, 0])

        output, attn_weights = self.dec_attn(dec_inp, r, r_w_bias, r_r_bias,
                               attn_mask=dec_attn_mask,
                               mems=mems)
        # print('2', output[0:5, 0, 0])
        

        output = self.layer_norm1(dec_inp+output)
        output2 = self.pos_ff(output)
        output2 = self.layer_norm2(output+output2)

        return output, attn_weights


    def forward_stable(self, dec_inp, r, r_w_bias, r_r_bias, dec_attn_mask=None, mems=None):

        #Layer norm will be applied at start of MHA module on both dec_inp2 and mems
        dec_inp2 = self.layer_norm1(dec_inp)
        #First Layer norm will be applied within dec_attn

        dec_inp2, attn_weights = self.dec_attn(dec_inp2, r, r_w_bias, r_r_bias,
                                attn_mask=dec_attn_mask,
                                mems=mems)
        

        #NOTE: In stable transformer they apply Relu before the layernorm/gate (in appendix C.3)
        if self.use_gate:
            dec_inp2 = self.gate_mha(dec_inp, F.relu(dec_inp2))
        else:
            dec_inp2 = dec_inp + F.relu(dec_inp2)

        dec_inp3 = self.layer_norm2(dec_inp2)

        dec_inp3 = self.pos_ff(dec_inp3)

        if self.use_gate:
            dec_inp3 = self.gate_mlp(dec_inp2, F.relu(dec_inp3))
        else:
            dec_inp3 = F.relu(dec_inp3) + dec_inp2

        return dec_inp3, attn_weights


    def forward(self,dec_inp, r, r_w_bias, r_r_bias, dec_attn_mask=None, mems=None):

        if self.use_stable_version:
            return self.forward_stable(dec_inp, r, r_w_bias, r_r_bias, dec_attn_mask, mems)

        return self.forward_orig(dec_inp, r, r_w_bias, r_r_bias, dec_attn_mask, mems)