import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn import init
from torch.autograd import Function


class ConvGRUCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, padding_mode='zeros', batchnorm=True, use_attention=True, timesteps=64):  # Timesteps is funky here... but go ahead and try this until you figure out the exact training length
        " Referenced from https://github.com/happyjin/ConvGRU-pytorch"
        super(ConvGRUCell, self).__init__()
        self.padding = kernel_size // 2
        hidden_size = hidden_dim
        self.batchnorm = batchnorm
        self.timesteps = timesteps
        self.use_attention = use_attention

        if self.use_attention:
            self.a_wu_gate = nn.Conv2d(hidden_size + input_dim, hidden_size, 1, padding=1 // 2)
            # self.a_w_gate = nn.Conv2d(hidden_size, hidden_size, 1, padding=1 // 2)
            # self.a_u_gate = nn.Conv2d(hidden_size, hidden_size, 1, padding=1 // 2)
            init.orthogonal_(self.a_wu_gate.weight)
            # init.orthogonal_(self.a_w_gate.weight)
            # init.orthogonal_(self.a_u_gate.weight)
            init.constant_(self.a_wu_gate.bias, 1.)
            # init.constant_(self.a_w_gate.bias, 1.)  # In future try setting to -1 -- originally set to 1
            # init.constant_(self.a_u_gate.bias, 1.)

        self.ff_cnx = nn.Conv2d(input_dim, hidden_size, 1)
        self.i_w_gate = nn.Conv2d(hidden_size + input_dim, hidden_size, 1)
        self.e_w_gate = nn.Conv2d(hidden_size * 2, hidden_size, 1)

        spatial_h_size = kernel_size
        self.h_padding = spatial_h_size // 2
        self.w_exc = nn.Parameter(torch.empty(hidden_size, hidden_size, spatial_h_size, spatial_h_size))
        self.w_inh = nn.Parameter(torch.empty(hidden_size, hidden_size, spatial_h_size, spatial_h_size))

        self.alpha = nn.Parameter(torch.empty((hidden_size, 1, 1)))
        self.gamma = nn.Parameter(torch.empty((hidden_size, 1, 1)))
        self.kappa = nn.Parameter(torch.empty((hidden_size, 1, 1)))
        self.w = nn.Parameter(torch.empty((hidden_size, 1, 1)))
        self.mu = nn.Parameter(torch.empty((hidden_size, 1, 1)))

        self.bn = nn.ModuleList([nn.BatchNorm2d(hidden_size, eps=1e-03, affine=True, track_running_stats=False) for i in range(2)])

        init.orthogonal_(self.ff_cnx.weight)
        init.orthogonal_(self.w_inh)
        init.orthogonal_(self.w_exc)

        init.orthogonal_(self.i_w_gate.weight)
        init.orthogonal_(self.e_w_gate.weight)

        for bn in self.bn:
            init.constant_(bn.weight, 0.1)

        init.constant_(self.alpha, 1.)
        init.constant_(self.mu, 0.)
        init.constant_(self.gamma, 0.)
        init.constant_(self.w, 1.)
        init.constant_(self.kappa, 1.)

        # Init gate biases
        init.uniform_(self.i_w_gate.bias.data, 1, self.timesteps - 1)
        self.i_w_gate.bias.data.log()
        self.e_w_gate.bias.data = -self.i_w_gate.bias.data

    def forward(self, input, state_cur_inh, state_cur_exc, activ=F.softplus, testmode=False):
        "Run the dales law circuit."""
        inhibition = state_cur_inh
        excitation = state_cur_exc
        if self.use_attention:
            input_state_cur = torch.cat([input, excitation], dim=1)
            att_gate = self.a_wu_gate(input_state_cur)  # Attention Spotlight -- MOST RECENT WORKING
            att_gate = torch.sigmoid(att_gate)

        # Gate E/I with attention immediately
        if self.use_attention:
            gated_input = input  # * att_gate  # In activ range
            gated_excitation = att_gate * excitation
            gated_inhibition = att_gate  # * inhibition
        else:
            gated_input = input

        # Compute inhibition
        inh_intx = self.bn[0](F.conv2d(gated_excitation, self.w_inh, padding=self.h_padding))  # in activ range
        inhibition_hat = activ(activ(self.ff_cnx(input)) - activ(inh_intx * (self.alpha * gated_inhibition + self.mu)))

        # Integrate inhibition
        inh_gate = torch.sigmoid(self.i_w_gate(torch.cat([gated_input, gated_inhibition], dim=1)))
        inhibition = (1 - inh_gate) * inhibition + inh_gate * inhibition_hat  # In activ range

        # Pass to excitatory neurons
        exc_gate = torch.sigmoid(self.e_w_gate(torch.cat([gated_excitation, gated_inhibition], dim=1)))
        exc_intx = self.bn[1](F.conv2d(inhibition, self.w_exc, padding=self.h_padding))  # In activ range
        excitation_hat = activ(exc_intx * (self.kappa * inhibition + self.gamma))  # Skip connection OR add OR add by self-sim
        excitation = (1 - exc_gate) * excitation + exc_gate * excitation_hat
        if testmode:
            return inhibition, excitation, att_gate
        else:
            return excitation, inhibition

