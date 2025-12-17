
class args:
    def __init__(self):
        self.N = 0
        self.obs_dim_n = 0
        self.action_dim_n = 0
        self.max_train_steps = 1e6
        self.episode_limit = 25
        self.evaluate_freq = 200
        self.evaluate_times = 3
        self.max_action = 1.0
        self.buffer_size = 10000
        self.batch_size = 128
        self.hidden_dim = 64
        self.noise_std_init = 0.5
        self.noise_std_min = 0.05
        self.noise_decay_steps = 1000
        self.use_noise_decay = True
        self.lr_a = 5e-4
        self.lr_c = 5e-4
        self.gamma = 0.95
        self.tau = 0.001
        self.use_orthogonal_init = True
        self.use_grad_clip = True
        self.noise_std_decay = (self.noise_std_init - self.noise_std_min) / self.noise_decay_steps

