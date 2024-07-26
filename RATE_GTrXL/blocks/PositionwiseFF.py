import torch
import torch.nn as nn

class PositionwiseFF(nn.Module):
    def __init__(self, d_model, d_inner, dropout, use_gate, pre_lnorm=False):
        super(PositionwiseFF, self).__init__()

        self.d_model = d_model
        self.d_inner = d_inner
        self.dropout = dropout
        self.use_gate = use_gate

        self.CoreNet = nn.Sequential(
            nn.Linear(d_model, d_inner), nn.ReLU(inplace=True),
            nn.Dropout(dropout), 
            nn.Linear(d_inner, d_model),
            nn.Dropout(dropout),
        )

        self.layer_norm = nn.LayerNorm(d_model)

        self.pre_lnorm = pre_lnorm

    def forward(self, inp):
        
        output = self.CoreNet(inp)

        return output
        
    # def forward(self, inp):
    #     if not self.use_gate:
    #         if self.pre_lnorm:
    #             ##### layer normalization + positionwise feed-forward
    #             core_out = self.CoreNet(self.layer_norm(inp))

    #             ##### residual connection
    #             output = core_out + inp
    #         else:
    #             ##### positionwise feed-forward
    #             core_out = self.CoreNet(inp)

    #             ##### residual connection + layer normalization
    #             output = self.layer_norm(inp + core_out)
        
    #     else:
    #         output = self.CoreNet(inp)

    #     return output