import numpy as np
from actor_memory import CollaborativeMemoryDNN
from critic_water_filling import WaterFilling_Critic

if __name__ == "__main__":
    # --- 系统参数初始化 ---
    T = 1000  # 总时隙数
    V = 4  # 摄像头视角数量
    sys_params = {
        'N': 196,  # 单视角 ViT Token 数量
        'eta': 1e-6,  # 计算能效系数
        'gamma': 10.0,  # 对数效用函数缩放参数
        'k_min': 0.1,  # 最小 Token 保留率
        'beta_0': 0.2,  # 基础精度
        'S_adapter': 1.2,  # adapter 参数下发带宽消耗 (MB)
        'SCL_weights': 50.0,  # 重训练模型权重带宽消耗 (MB)
        'S_query': 5.0  # 关键帧上传带宽消耗 (MB)
    }
    B_avg = 3.0  # 长期平均带宽预算 (MB/时隙)
    V_lya = 20.0  # 李雅普诺夫权衡参数 (V越大越重视精度，队列越长)

    # 初始化 Actor-Critic 记忆网络
    # State = [E_drift(t), Y_bw(t)] -> 维度为 2
    mem = CollaborativeMemoryDNN(V=V, state_dim=2, eps_mask=0.05)

    Y_bw = np.zeros(T)  # 虚拟带宽队列
    G_his = np.zeros(T)

    print(f"开始模拟: 视角数 V={V}, 总时隙 T={T}, 带宽预算 B_avg={B_avg}")

    for t in range(T):
        # 1. 环境观测 (模拟非平稳环境生成)
        # E_drift: 当前预测不确定性(熵)；w_t: 云端下发的空间权重
        E_drift = np.random.uniform(0, 1.0)
        w_t = np.random.uniform(0.01, 1.0, V)

        # 当前状态输入
        state_t = np.array([E_drift, Y_bw[t] / 100.0])  # 状态归一化

        # 2. Actor 生成离散动作候选集
        candidates = mem.decode_and_quantize(state_t, w_t, M_t=10)

        # 3. Critic 评估所有候选集 (结合闭式注水算法)
        best_G = -np.inf
        best_action = None
        best_details = None

        for (v_cand, u_cand) in candidates:
            G, k_t, c_comm, acc, cost = WaterFilling_Critic(
                v_cand, u_cand, w_t, Y_bw[t], V_lya, sys_params
            )

            if G > best_G:
                best_G = G
                best_action = (v_cand, u_cand, k_t)
                best_details = c_comm

        v_opt, u_opt, k_opt = best_action
        c_comm_opt = best_details
        G_his[t] = best_G

        # 4. 队列演进模块 (Queueing Module)
        if t < T - 1:
            # Y_bw(t+1) = max(Y_bw(t) + C_comm - B_avg, 0)
            Y_bw[t + 1] = max(Y_bw[t] + c_comm_opt - B_avg, 0)

        # 5. 策略更新模块 (将最优标签压入回放区并训练)
        mem.encode(state_t, v_opt, u_opt)


        print(f"时隙 {t}: 最优动作 v={v_opt}, u={u_opt}, 带宽队列 Y={Y_bw[t]:.2f}")

    print("模拟结束。队列最终长度:", Y_bw[-1])