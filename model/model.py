import numpy as np
import torch
import torch.nn as nn
from .helpers import SinusoidalPosEmb


class SelfAttentionLayer(nn.Module):
    def __init__(self, embed_size, num_heads, dropout):
        super(SelfAttentionLayer, self).__init__()
        self.attention = nn.MultiheadAttention(embed_size, num_heads=num_heads, dropout=dropout)

    def forward(self, x):
        return self.attention(x, x, x)


# class AddNorm(nn.Module):
#     def __init__(self, normalized_shape, dropout, **kwargs):
#         super(AddNorm, self).__init__(**kwargs)
#         self.dropout = nn.Dropout(dropout)
#         self.ln = nn.LayerNorm(normalized_shape)
#
#     def forward(self, X, Y):
#         return self.ln(self.dropout(Y) + X)


class AddNorm(nn.Module):
    def __init__(self, normalized_shape, **kwargs):
        super(AddNorm, self).__init__(**kwargs)
        self.ln = nn.LayerNorm(normalized_shape)

    def forward(self, X, Y):
        return self.ln(Y + X)


class Diffusion_Net(nn.Module):
    def __init__(
        self,
        state_dim,
        action_dim,
        hidden_dim=256,
        t_dim=16,
        activation=nn.Mish,
        s_dim=128,
        x_dim=64,
        device=None
    ):
        super(Diffusion_Net, self).__init__()
        self.state_dim = int(np.prod(state_dim))
        self.action_dim = int(np.prod(action_dim))
        self.input_dim = self.state_dim + self.action_dim + t_dim
        self.device = device

        _act = activation
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(t_dim),
            nn.Linear(t_dim, t_dim * 2),
            _act(),
            nn.Linear(t_dim * 2, t_dim)
        )

        self.extractor_x = nn.Sequential(
            nn.Linear(self.action_dim, x_dim),
            _act()
        )

        self.extractor_state = nn.Sequential(
            nn.Linear(self.state_dim, s_dim),
            _act()
        )

        input_dim = s_dim + x_dim + t_dim
        self.mid_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            _act(),
            nn.Linear(hidden_dim, hidden_dim),
            _act(),
            nn.Linear(hidden_dim, action_dim)
        )
        self.final_layer = nn.Tanh()

    def forward(self, x, time, state):
        # if self.device is not None:
        #     state = torch.as_tensor(state, device=self.device, dtype=torch.float32)
        t = self.time_mlp(time)
        state = state.reshape(state.size(0), -1)
        # x = torch.cat([x, t, state], dim=1)
        x = self.extractor_x(x)
        s = self.extractor_state(state)
        inp = torch.cat([x, t, s], dim=1)
        y = self.mid_layer(inp)
        return self.final_layer(y)


class Diffusion_Attention(nn.Module):
    def __init__(
            self,
            state_dim,
            action_dim,
            hidden_dim=256,
            t_dim=16,
            activation=nn.Mish,
            dropout=0.4,
            embed_size=256,
            s_dim=128,
            x_dim=64,
            device=None
    ):
        super(Diffusion_Attention, self).__init__()
        self.state_dim = int(np.prod(state_dim))
        self.action_dim = int(np.prod(action_dim))
        self.device = device

        _act = activation
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(t_dim),
            nn.Linear(t_dim, t_dim * 2),
            _act(),
            nn.Linear(t_dim * 2, t_dim)
        )

        self.extractor_x = nn.Sequential(
            nn.Linear(self.action_dim, x_dim),
            _act(),
            nn.Dropout(dropout)
        )

        self.extractor_state = nn.Sequential(
            nn.Linear(self.state_dim, s_dim),
            _act(),
            nn.Dropout(dropout)
        )

        input_dim = s_dim + x_dim + t_dim
        self.extractor = nn.Sequential(
            nn.Linear(input_dim, embed_size),
            _act(),
            nn.Dropout(dropout)
        )

        self.input_dims = self.state_dim + self.action_dim + t_dim

        self.ex_layer = nn.Sequential(
            nn.Linear(self.input_dims, hidden_dim),
            _act(),
            nn.Linear(hidden_dim, hidden_dim),
            _act()
        )

        self.attention = SelfAttentionLayer(embed_size, 1, dropout=dropout)
        # self.addnorm = AddNorm(embed_size, dropout=dropout)
        self.addnorm = AddNorm(embed_size)

        self.mid_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            # _act(),
            # nn.Linear(hidden_dim, hidden_dim),
            _act(),
            nn.Linear(hidden_dim, self.action_dim)
        )
        self.final_layer = nn.Tanh()

    def forward(self, x, time, state):
        # if self.device is not None:
        #     state = torch.as_tensor(state, device=self.device, dtype=torch.float32)

        t = self.time_mlp(time)
        state = state.reshape(state.size(0), -1)
        ex_x = torch.cat([x, t, state], dim=1)
        ex_y = self.ex_layer(ex_x)
        x = self.extractor_x(x)
        s = self.extractor_state(state)
        inp = torch.cat([x, t, s], dim=1)
        extract = self.extractor(inp)
        att, _ = self.attention(extract)
        y = self.addnorm(extract, att)
        y += ex_y
        y = self.mid_layer(y)
        return self.final_layer(y)


