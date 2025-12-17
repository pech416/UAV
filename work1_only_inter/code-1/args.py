# args.py

class args:
    def __init__(self):
        # ---------------- 基本配置 ----------------
        self.N = None              # 会由 Runner 动态设置 (UAV 数量)
        self.obs_dim_n = None      # 每个 agent 的状态维度
        self.action_dim_n = None   # 每个 agent 的动作维度

        # ---------------- 网络结构 ----------------
        self.hidden_dim = 128      # Actor/Critic 隐藏层大小

        # ---------------- 学习率 ----------------
        self.lr_a = 1e-4           # Actor 学习率
        self.lr_c = 5e-4           # Critic 学习率 (比 actor 稍大，收敛更快)

        # ---------------- 强化学习参数 ----------------
        self.gamma = 0.95          # 折扣因子
        self.tau = 0.005           # 软更新系数 (小一些更稳定)

        # ---------------- 经验回放 ----------------
        self.buffer_size = int(1e6)  # Replay Buffer 大小
        self.batch_size = 256        # 批量大小

        # ---------------- 探索噪声 ----------------
        self.max_action = 1.0
        self.noise_std_init = 0.15   # 初始探索噪声
        self.noise_std_min = 0.03    # 最小探索噪声
        self.noise_std_decay = 0.995 # 每步噪声衰减系数
        self.use_noise_decay = True

        # ---------------- 训练控制 (TD3-style) ----------------
        self.num_critic_updates = 2   # 每步 critic 更新次数
        self.policy_delay = 2         # 每隔多少步更新一次 actor
        self.target_noise_std = 0.05  # target smoothing 噪声标准差
        self.target_noise_clip = 0.2  # target smoothing 噪声截断范围

        # ---------------- 训练细节 ----------------
        self.use_grad_clip = True       # 是否启用梯度裁剪
        self.use_orthogonal_init = True # 网络权重正交初始化
        self.evaluate_times = 5         # 评估次数 (用于测试阶段)

        # ---------------- 日志 ----------------
        self.log_dir = "./logs_maddpg"
