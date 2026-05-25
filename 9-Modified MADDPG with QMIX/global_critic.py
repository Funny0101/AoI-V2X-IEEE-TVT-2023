import torch as T
import torch.nn.functional as F
import torch.optim as optim
from Classes.qmix_networks import AgentQNetwork, AgentTargetQNetwork, QMIXMixer, TargetQMIXMixer


class QMIXTrainer():
    def __init__(self, lr, input_dims, tau, n_actions, gamma,
                 q_fc1_dims, q_fc2_dims, batch_size, n_agents,
                 update_actor_interval, noise, mix_embed_dim=32):

        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.lr = lr
        self.number_agents = n_agents
        self.number_actions = n_actions
        self.number_states = input_dims
        self.update_actor_iter = update_actor_interval
        self.learn_step_counter = 0
        self.noise = noise

        state_dim = input_dims * n_agents

        self.q_networks = []
        self.target_q_networks = []
        for i in range(n_agents):
            self.q_networks.append(AgentQNetwork(lr, input_dims, q_fc1_dims, q_fc2_dims, n_actions, i))
            self.target_q_networks.append(AgentTargetQNetwork(lr, input_dims, q_fc1_dims, q_fc2_dims, n_actions, i))

        self.mixer = QMIXMixer(n_agents, state_dim, mix_embed_dim)
        self.target_mixer = TargetQMIXMixer(n_agents, state_dim, mix_embed_dim)

        params = list(self.mixer.parameters())
        for q_net in self.q_networks:
            params += list(q_net.parameters())
        self.optimizer = optim.Adam(params, lr=lr)

        self.update_target_networks(tau=1)

    def save_models(self):
        for i in range(self.number_agents):
            self.q_networks[i].save_checkpoint()
            self.target_q_networks[i].save_checkpoint()
        self.mixer.save_checkpoint()
        self.target_mixer.save_checkpoint()

    def load_models(self):
        for i in range(self.number_agents):
            self.q_networks[i].load_checkpoint()
            self.target_q_networks[i].load_checkpoint()
        self.mixer.load_checkpoint()
        self.target_mixer.load_checkpoint()

    def global_learn(self, agents_nets, state, action, reward_g, reward_t1, reward_t2, state_, terminal):

        states = T.tensor(state, dtype=T.float).to(self.mixer.device)
        states_ = T.tensor(state_, dtype=T.float).to(self.mixer.device)
        actions = T.tensor(action, dtype=T.float).to(self.mixer.device)
        rewards_g = T.tensor(reward_g, dtype=T.float).to(self.mixer.device)
        rewards_t1 = T.tensor(reward_t1, dtype=T.float).to(self.mixer.device)
        rewards_t2 = T.tensor(reward_t2, dtype=T.float).to(self.mixer.device)
        done = T.tensor(terminal).to(self.mixer.device)

        # ---- Critic update ----
        self.optimizer.zero_grad()

        q_values = []
        for i in range(self.number_agents):
            obs_i = states[:, i * self.number_states:(i + 1) * self.number_states]
            act_i = actions[:, i * self.number_actions:(i + 1) * self.number_actions]
            q_values.append(self.q_networks[i](obs_i, act_i))

        q_vals = T.cat(q_values, dim=1)
        q_tot = self.mixer(q_vals, states)

        with T.no_grad():
            target_actions = T.zeros([self.batch_size, self.number_actions * self.number_agents],
                                     device=self.mixer.device)
            for i in range(self.number_agents):
                agents_nets[i].target_actor.eval()
                obs_i_ = states_[:, i * self.number_states:(i + 1) * self.number_states]
                target_act_i = agents_nets[i].target_actor.forward(obs_i_)
                target_act_i = target_act_i + T.clamp(T.randn_like(target_act_i) * 0.2, -0.5, 0.5)
                target_act_i = T.clamp(target_act_i, -0.999, 0.999)
                target_actions[:, i * self.number_actions:(i + 1) * self.number_actions] = target_act_i

            target_q_values = []
            for i in range(self.number_agents):
                obs_i_ = states_[:, i * self.number_states:(i + 1) * self.number_states]
                target_act_i = target_actions[:, i * self.number_actions:(i + 1) * self.number_actions]
                target_q_values.append(self.target_q_networks[i](obs_i_, target_act_i))

            target_q_vals = T.cat(target_q_values, dim=1)
            q_tot_target = self.target_mixer(target_q_vals, states_)

        q_tot_target = q_tot_target.squeeze(-1)
        q_tot_target[done] = 0.0
        target = rewards_g + self.gamma * q_tot_target
        target = target.view(self.batch_size, 1).detach()

        critic_loss = F.mse_loss(q_tot, target)
        critic_loss.backward()
        self.optimizer.step()

        self.update_target_networks()

        self.learn_step_counter += 1

        if self.learn_step_counter % self.update_actor_iter != 0:
            return

        # ---- Actor update (delayed) ----
        current_q_values = []
        for i in range(self.number_agents):
            obs_i = states[:, i * self.number_states:(i + 1) * self.number_states]
            act_i = agents_nets[i].actor.forward(obs_i)
            current_q_values.append(self.q_networks[i](obs_i, act_i))

        current_q_vals = T.cat(current_q_values, dim=1)
        q_tot_actor = self.mixer(current_q_vals, states)

        actor_global_loss = -q_tot_actor.detach()

        for i in range(self.number_agents):
            actor_global_loss_ = actor_global_loss.clone().detach()
            agents_nets[i].local_learn(actor_global_loss_,
                                       states[:, i * self.number_states:(i + 1) * self.number_states],
                                       actions[:, i * self.number_actions:(i + 1) * self.number_actions],
                                       rewards_t1[:, i],
                                       rewards_t2[:, i],
                                       states_[:, i * self.number_states:(i + 1) * self.number_states],
                                       done)

    def update_target_networks(self, tau=None):
        if tau is None:
            tau = self.tau

        for i in range(self.number_agents):
            q_params = dict(self.q_networks[i].named_parameters())
            target_q_params = dict(self.target_q_networks[i].named_parameters())
            for name in q_params:
                q_params[name] = tau * q_params[name].clone() + \
                    (1 - tau) * target_q_params[name].clone()
            self.target_q_networks[i].load_state_dict(q_params)

        mixer_params = dict(self.mixer.named_parameters())
        target_mixer_params = dict(self.target_mixer.named_parameters())
        for name in mixer_params:
            mixer_params[name] = tau * mixer_params[name].clone() + \
                (1 - tau) * target_mixer_params[name].clone()
        self.target_mixer.load_state_dict(mixer_params)
