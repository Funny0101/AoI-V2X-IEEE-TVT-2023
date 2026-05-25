import numpy as np
from collections import deque

class ReplayBuffer():
    def __init__(self, max_size, input_shape, n_actions, n_agents, n_step=5, gamma=0.99):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.n_step = n_step
        self.gamma = gamma

        self.state_memory = np.zeros((self.mem_size, input_shape * n_agents), dtype=np.float16)
        self.action_memory = np.zeros((self.mem_size, n_actions * n_agents), dtype=np.float16)
        self.reward_global_memory = np.zeros(self.mem_size)
        self.reward_task1 = np.zeros((self.mem_size, n_agents), dtype=np.float16)
        self.reward_task2 = np.zeros((self.mem_size, n_agents), dtype=np.float16)
        self.new_state_memory = np.zeros((self.mem_size, input_shape * n_agents), dtype=np.float16)
        self.terminal_memory = np.zeros(self.mem_size, dtype=bool)

        # N-step rolling buffer
        self.n_step_buffer = deque(maxlen=n_step)

    def _get_n_step_info(self):
        """Compute n-step returns from the rolling buffer."""
        reward_g = 0.0
        first = self.n_step_buffer[0]
        n_agents = first[3].shape[0]  # first[3] = reward_t1
        reward_t1 = np.zeros(n_agents, dtype=np.float32)
        reward_t2 = np.zeros(n_agents, dtype=np.float32)
        final_state, final_done = None, False

        for idx in range(len(self.n_step_buffer)):
            entry = self.n_step_buffer[idx]
            r_g = entry[2]
            r_t1 = entry[3]
            r_t2 = entry[4]
            state_ = entry[5]
            done = entry[6]

            reward_g += (self.gamma ** idx) * r_g
            reward_t1 += (self.gamma ** idx) * r_t1.astype(np.float32)
            reward_t2 += (self.gamma ** idx) * r_t2.astype(np.float32)

            final_state = state_
            final_done = done
            if done:
                break

        return reward_g, reward_t1, reward_t2, final_state, final_done

    def store_transition(self, state, action, reward_g, reward_t1, reward_t2, state_, done):
        # Store in rolling buffer: (state, action, reward_g, reward_t1, reward_t2, state_, done)
        self.n_step_buffer.append((state, action, reward_g, reward_t1, reward_t2, state_, done))

        # Only flush to main buffer once we have n_step transitions
        if len(self.n_step_buffer) < self.n_step:
            return

        # Origin transition
        origin = self.n_step_buffer[0]
        s, a = origin[0], origin[1]

        # Compute n-step returns
        n_reward_g, n_reward_t1, n_reward_t2, n_state_, n_done = self._get_n_step_info()

        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = s
        self.action_memory[index] = a
        self.reward_global_memory[index] = n_reward_g
        self.reward_task1[index] = n_reward_t1.astype(np.float16)
        self.reward_task2[index] = n_reward_t2.astype(np.float16)
        self.new_state_memory[index] = n_state_
        self.terminal_memory[index] = n_done

        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)

        batch = np.random.choice(max_mem, batch_size)

        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards_g = self.reward_global_memory[batch]
        rewards_task1 = self.reward_task1[batch]
        rewards_task2 = self.reward_task2[batch]
        states_ = self.new_state_memory[batch]
        dones = self.terminal_memory[batch]

        return states, actions, rewards_g, rewards_task1, rewards_task2, states_, dones
