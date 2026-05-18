import os
import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math


class MultiHeadAttention(nn.Module):
    """Multi-Head Self-Attention for agent interaction modeling."""

    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(embed_dim)

        self._reset_parameters()

    def _reset_parameters(self):
        for p in [self.q_proj, self.k_proj, self.v_proj]:
            nn.init.xavier_uniform_(p.weight)
            nn.init.constant_(p.bias, 0.)
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.constant_(self.out_proj.bias, 0.)

    def forward(self, x):
        """
        x: (batch_size, n_agents, embed_dim)
        Returns: (batch_size, n_agents, embed_dim)
        """
        residual = x
        batch_size, n_agents, _ = x.shape

        q = self.q_proj(x).view(batch_size, n_agents, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, n_agents, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, n_agents, self.num_heads, self.head_dim).transpose(1, 2)

        attn_scores = T.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        attn_output = T.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, n_agents, self.embed_dim)
        output = self.out_proj(attn_output)

        return self.layer_norm(output + residual)


class AttentionCriticNetwork(nn.Module):
    """
    Global Critic with Multi-Head Attention.
    Instead of concatenating all agents' states and actions into a flat vector,
    each agent's (state, action) pair is treated as a token.
    Self-attention learns which agents should coordinate more.
    """

    def __init__(self, beta, input_dims, fc1_dims, fc2_dims, fc3_dims, n_agents, n_actions, name, agent_label,
                 chkpt_dir='tmp/ddpg', num_heads=2, dropout=0.1):
        super(AttentionCriticNetwork, self).__init__()

        self.name = name
        self.n_agents = n_agents
        self.input_dims = input_dims
        self.n_actions = n_actions
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.fc3_dims = fc3_dims

        self.checkpoint_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), chkpt_dir)
        self.checkpoint_file = os.path.join(self.checkpoint_dir, self.name + '_ddpg')

        # Per-agent token embedding: (state + action) -> embed_dim
        self.token_dim = input_dims + n_actions  # 19 + 3 = 22 per agent
        self.embed_dim = 128

        self.token_embedding = nn.Sequential(
            nn.Linear(self.token_dim, self.embed_dim),
            nn.LayerNorm(self.embed_dim),
            nn.ReLU(),
        )

        # Multi-head self-attention block
        self.attention1 = MultiHeadAttention(self.embed_dim, num_heads, dropout)
        self.attention2 = MultiHeadAttention(self.embed_dim, num_heads, dropout)

        # Feed-forward after attention: flatten all agent embeddings -> Q value
        self.ff = nn.Sequential(
            nn.Linear(self.embed_dim * n_agents, fc2_dims),
            nn.LayerNorm(fc2_dims),
            nn.ReLU(),
            nn.Linear(fc2_dims, fc3_dims),
            nn.LayerNorm(fc3_dims),
            nn.ReLU(),
        )

        self.q = nn.Linear(fc3_dims, 1)

        # Weight initialization
        for module in [self.token_embedding, self.ff]:
            for layer in module:
                if isinstance(layer, nn.Linear):
                    f = 1. / np.sqrt(layer.weight.data.size()[0])
                    layer.weight.data.uniform_(-f, f)
                    layer.bias.data.uniform_(-f, f)

        f4 = 0.003
        self.q.weight.data.uniform_(-f4, f4)
        self.q.bias.data.uniform_(-f4, f4)

        self.optimizer = optim.Adam(self.parameters(), lr=beta, weight_decay=0.01)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state, action):
        """
        state:  (batch_size, n_agents * input_dims)  i.e. (batch_size, 95)
        action: (batch_size, n_agents * n_actions)    i.e. (batch_size, 15)
        """
        batch_size = state.shape[0]

        # Reshape into per-agent tokens: (batch, n_agents, input_dims) and (batch, n_agents, n_actions)
        states_per_agent = state.view(batch_size, self.n_agents, self.input_dims)
        actions_per_agent = action.view(batch_size, self.n_agents, self.n_actions)

        # Concatenate state + action per agent: (batch, n_agents, token_dim)
        tokens = T.cat([states_per_agent, actions_per_agent], dim=-1)

        # Embed tokens: (batch, n_agents, embed_dim)
        x = self.token_embedding(tokens)

        # Two layers of self-attention
        x = self.attention1(x)
        x = self.attention2(x)

        # Flatten and feed through FF layers
        x = x.view(batch_size, -1)  # (batch, n_agents * embed_dim)
        x = self.ff(x)
        q_value = self.q(x)

        return q_value

    def save_checkpoint(self):
        print('... saving checkpoint ...')
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        print('... loading checkpoint ...')
        self.load_state_dict(T.load(self.checkpoint_file))

    def save_best(self):
        print('... saving best checkpoint ...')
        checkpoint_file = os.path.join(self.checkpoint_dir, self.name + '_best')
        T.save(self.state_dict(), checkpoint_file)
