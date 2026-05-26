import os
import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class CriticNetwork(nn.Module):
    def __init__(self, beta, input_dims, fc1_dims, fc2_dims, n_agents, n_actions, name, agent_label,
                 chkpt_dir='tmp/ddpg'):
        super(CriticNetwork, self).__init__()

        if name == 'global_critic1' or name == 'global_critic2' or name == 'global_target_critic1' \
                or name == 'global_target_critic2':
            self.input_dims = input_dims * n_agents
            self.n_actions = n_actions * n_agents
            self.name = name
        else:
            self.input_dims = input_dims
            self.n_actions = n_actions
            self.name = name + '_' + str(agent_label)

        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.checkpoint_dir =  os.path.join(os.path.dirname(os.path.realpath(__file__)), chkpt_dir)
        self.checkpoint_file = os.path.join(self.checkpoint_dir, self.name + '_ddpg')

        self.fc1 = nn.Linear(self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)

        self.bn1 = nn.LayerNorm(self.fc1_dims)
        self.bn2 = nn.LayerNorm(self.fc2_dims)

        self.action_value = nn.Linear(self.n_actions, self.fc2_dims)

        self.q = nn.Linear(self.fc2_dims, 1)

        f1 = 1. / np.sqrt(self.fc1.weight.data.size()[0])
        self.fc1.weight.data.uniform_(-f1, f1)
        self.fc1.bias.data.uniform_(-f1, f1)

        f2 = 1. / np.sqrt(self.fc2.weight.data.size()[0])
        self.fc2.weight.data.uniform_(-f2, f2)
        self.fc2.bias.data.uniform_(-f2, f2)

        f4 = 0.003
        self.q.weight.data.uniform_(-f4, f4)
        self.q.bias.data.uniform_(-f4, f4)

        f5 = 1. / np.sqrt(self.action_value.weight.data.size()[0])
        self.action_value.weight.data.uniform_(-f5, f5)
        self.action_value.bias.data.uniform_(-f5, f5)

        self.optimizer = optim.Adam(self.parameters(), lr=beta,
                                    weight_decay=0.01)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

        self.to(self.device)

    def forward(self, state, action):
        state_value = self.fc1(state)
        state_value = self.bn1(state_value)
        state_value = F.relu(state_value)
        state_value = self.fc2(state_value)
        state_value = self.bn2(state_value)
        action_value = self.action_value(action)
        state_action_value = F.relu(T.add(state_value, action_value))
        state_action_value = self.q(state_action_value)

        return state_action_value

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


class ActorNetwork(nn.Module):
    def __init__(self, alpha, input_dims, fc1_dims, fc2_dims, n_agents, n_actions, name, agent_label,
                 chkpt_dir='tmp/ddpg'):
        super(ActorNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.name = name + '_' + str(agent_label)
        self.checkpoint_dir =  os.path.join(os.path.dirname(os.path.realpath(__file__)), chkpt_dir)
        self.checkpoint_file = os.path.join(self.checkpoint_dir, self.name + '_ddpg')

        self.fc1 = nn.Linear(self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)

        self.bn1 = nn.LayerNorm(self.fc1_dims)
        self.bn2 = nn.LayerNorm(self.fc2_dims)

        self.rb_logits = nn.Linear(self.fc2_dims, 3)
        self.mode_logits = nn.Linear(self.fc2_dims, 2)
        self.power_val = nn.Linear(self.fc2_dims, 1)

        f2 = 1. / np.sqrt(self.fc2.weight.data.size()[0])
        self.fc2.weight.data.uniform_(-f2, f2)
        self.fc2.bias.data.uniform_(-f2, f2)

        f1 = 1. / np.sqrt(self.fc1.weight.data.size()[0])
        self.fc1.weight.data.uniform_(-f1, f1)
        self.fc1.bias.data.uniform_(-f1, f1)

        f3 = 0.003
        self.rb_logits.weight.data.uniform_(-f3, f3)
        self.rb_logits.bias.data.uniform_(-f3, f3)
        self.mode_logits.weight.data.uniform_(-f3, f3)
        self.mode_logits.bias.data.uniform_(-f3, f3)
        self.power_val.weight.data.uniform_(-f3, f3)
        self.power_val.bias.data.uniform_(-f3, f3)

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

        self.to(self.device)

    def forward(self, state):
        x = self.fc1(state)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)
        
        rb_action = F.gumbel_softmax(self.rb_logits(x), tau=1.0, hard=True)
        mode_action = F.gumbel_softmax(self.mode_logits(x), tau=1.0, hard=True)
        power_action = T.tanh(self.power_val(x))
        
        return T.cat([rb_action, mode_action, power_action], dim=-1)

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
