import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


class ActorDNN(nn.Module):
    def __init__(self, state_dim, V):
        super(ActorDNN, self).__init__()
        self.V = V
        # 共享特征提取层
        self.shared_mlp = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )
        # 多头输出：v_t 偏好 (Sigmoid) 和 u_t 决策 (Softmax)
        self.v_head = nn.Sequential(nn.Linear(128, V), nn.Sigmoid())
        self.u_head = nn.Sequential(nn.Linear(128, 3), nn.Softmax(dim=-1))

    def forward(self, x):
        feat = self.shared_mlp(x)
        return self.v_head(feat), self.u_head(feat)


class CollaborativeMemoryDNN:
    def __init__(
        self,
        V,
        state_dim=2,
        lr=0.01,
        batch_size=64,
        memory_size=1024,
        eps_mask=0.05,
        min_active_views=0,
    ):
        self.V = V
        self.state_dim = state_dim
        self.eps_mask = eps_mask
        self.min_active_views = min(max(int(np.ceil(min_active_views)), 0), V)
        self.batch_size = batch_size
        self.model = ActorDNN(state_dim, V)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.criterion_v = nn.BCELoss()
        self.criterion_u = nn.CrossEntropyLoss()

        self.memory_size = memory_size
        self.memory_state = np.zeros((memory_size, state_dim))
        self.memory_v = np.zeros((memory_size, V))
        self.memory_u = np.zeros(memory_size, dtype=np.int64)
        self.memory_counter = 0

    def _enforce_min_active_views(self, v_candidate, w_t):
        if self.min_active_views <= 0:
            return v_candidate

        repaired = np.array(v_candidate, dtype=np.int64, copy=True)
        active_count = int(np.sum(repaired))
        if active_count >= self.min_active_views:
            return repaired

        inactive_indices = np.where(repaired == 0)[0]
        sorted_inactive = inactive_indices[np.argsort(w_t[inactive_indices])[::-1]]
        need = self.min_active_views - active_count
        repaired[sorted_inactive[:need]] = 1
        return repaired

    def encode(self, state, v_opt, u_opt):
        """将最优解 (s_t, v_t*, u_t*) 存入回放缓冲区"""
        idx = self.memory_counter % self.memory_size
        self.memory_state[idx, :] = state
        self.memory_v[idx, :] = v_opt
        self.memory_u[idx] = u_opt
        self.memory_counter += 1

        if self.memory_counter % 10 == 0:  # 异步更新
            self.learn()

    def learn(self):
        """计算混合损失函数并更新网络"""
        max_mem = min(self.memory_counter, self.memory_size)
        sample_indices = np.random.choice(max_mem, self.batch_size)

        s_batch = torch.FloatTensor(self.memory_state[sample_indices])
        v_batch = torch.FloatTensor(self.memory_v[sample_indices])
        u_batch = torch.LongTensor(self.memory_u[sample_indices])

        self.model.train()
        self.optimizer.zero_grad()

        v_pred, u_pred = self.model(s_batch)

        # 多标签交叉熵损失计算
        loss_v = self.criterion_v(v_pred, v_batch)
        loss_u = self.criterion_u(u_pred, u_batch)
        total_loss = loss_v + loss_u

        total_loss.backward()
        self.optimizer.step()

    def decode_and_quantize(self, state, w_t, M_t=10):
        s_tensor = torch.FloatTensor(state).unsqueeze(0)
        self.model.eval()
        with torch.no_grad():
            v_pred, u_pred = self.model(s_tensor)

        v_prob = v_pred.numpy()[0]

        # 【关键修改】：先只生成 v 的候选池
        v_candidates = []
        v_base = 1 * (v_prob > 0.5)
        v_candidates.append(v_base)

        v_abs = np.abs(v_prob - 0.5)
        idx_list = np.argsort(v_abs)
        for i in range(min(M_t // 2 - 1, self.V)):
            v_cand = np.copy(v_base)
            v_cand[idx_list[i]] = 1 - v_cand[idx_list[i]]
            v_candidates.append(v_cand)

        v_noisy = 1 / (1 + np.exp(-(v_prob + np.random.normal(0, 1, self.V))))
        v_base_n = 1 * (v_noisy > 0.5)
        v_candidates.append(v_base_n)
        v_abs_n = np.abs(v_noisy - 0.5)
        idx_list_n = np.argsort(v_abs_n)
        for i in range(min(M_t // 2 - 1, self.V)):
            v_cand = np.copy(v_base_n)
            v_cand[idx_list_n[i]] = 1 - v_cand[idx_list_n[i]]
            v_candidates.append(v_cand)

        # 应用物理先验掩码
        for i_cand in range(len(v_candidates)):
            for i in range(self.V):
                if w_t[i] < self.eps_mask:
                    v_candidates[i_cand][i] = 0
            v_candidates[i_cand] = self._enforce_min_active_views(v_candidates[i_cand], w_t)

        # 去重
        unique_v = list(set([tuple(v) for v in v_candidates]))

        # 【关键修改】：将去重后的 v 强制与所有的 u (0, 1, 2) 进行交叉组合！
        # 让 Critic 拥有完全的选择权，根据当前的带宽压力自行决定 u
        final_candidates = []
        for v in unique_v:
            for u in [0, 1, 2]:
                final_candidates.append((np.array(v), u))

        return final_candidates
