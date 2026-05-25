import numpy as np

class ReplayBuffer():
    def __init__(self, max_size, input_shape, n_actions, n_agents, alpha=0.6):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.alpha = alpha  # prioritization exponent
        self.epsilon = 1e-4  # small constant to avoid zero priority
        self.max_priority = 1.0

        self.state_memory = np.zeros((self.mem_size, input_shape * n_agents), dtype=np.float16)
        self.action_memory = np.zeros((self.mem_size, n_actions * n_agents), dtype=np.float16)
        self.reward_global_memory = np.zeros(self.mem_size)
        self.reward_task1 = np.zeros((self.mem_size, n_agents), dtype=np.float16)
        self.reward_task2 = np.zeros((self.mem_size, n_agents), dtype=np.float16)
        self.new_state_memory = np.zeros((self.mem_size, input_shape * n_agents), dtype=np.float16)
        self.terminal_memory = np.zeros(self.mem_size, dtype=bool)

        # PER: priorities stored as float64 for numerical stability
        self.priorities = np.zeros(self.mem_size, dtype=np.float64)

    def store_transition(self, state, action, reward_g, reward_t1, reward_t2, state_, done):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.action_memory[index] = action
        self.reward_global_memory[index] = reward_g
        self.reward_task1[index] = reward_t1
        self.reward_task2[index] = reward_t2
        self.new_state_memory[index] = state_
        self.terminal_memory[index] = done

        # New transition gets max priority so it's sampled at least once
        self.priorities[index] = self.max_priority

        self.mem_cntr += 1

    def sample_buffer(self, batch_size, beta=0.4):
        max_mem = min(self.mem_cntr, self.mem_size)

        # Compute sampling probabilities
        priorities = self.priorities[:max_mem]
        probs = priorities ** self.alpha
        probs = probs / (probs.sum() + 1e-8)

        batch = np.random.choice(max_mem, batch_size, p=probs, replace=False)

        # Importance sampling weights
        total = max_mem
        weights = (total * probs[batch]) ** (-beta)
        weights = weights / (weights.max() + 1e-8)  # normalize

        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards_g = self.reward_global_memory[batch]
        rewards_task1 = self.reward_task1[batch]
        rewards_task2 = self.reward_task2[batch]
        states_ = self.new_state_memory[batch]
        dones = self.terminal_memory[batch]

        return states, actions, rewards_g, rewards_task1, rewards_task2, states_, dones, batch, weights

    def update_priorities(self, indices, td_errors):
        """Update priorities based on TD errors after learning."""
        for idx, td_err in zip(indices, td_errors):
            priority = (abs(td_err) + self.epsilon) ** self.alpha
            self.priorities[idx] = priority
            self.max_priority = max(self.max_priority, priority)
