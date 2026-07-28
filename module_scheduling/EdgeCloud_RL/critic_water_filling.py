import numpy as np


def WaterFilling_Critic(v_t, u_t, w_t, E_drift, struct_drift, Y_bw, V_lya, sys_params):
    N = sys_params['N']
    eta = sys_params['eta']
    gamma = sys_params['gamma']
    k_min = sys_params['k_min']
    beta_0 = sys_params['beta_0']
    alpha_env = sys_params.get('alpha_env', 0.4)
    alpha_struct = sys_params.get('alpha_struct', 0.3)
    retrain_bonus = sys_params.get('retrain_bonus', 0.2)
    tau_retrain = sys_params.get('tau_retrain', 0.5)

    # 1. 通信开销
    c_comm = 0
    if u_t == 1:
        c_comm = sys_params['S_prompt'] + sys_params['S_query']
    elif u_t == 2:
        c_comm = sys_params['SCL_weights'] + sys_params['S_query']

    # u=0: 本地推理，同时受到环境漂移和结构性漂移惩罚
    # u=1: 轻量 Prompt 更新，主要修复环境漂移，但无法解决结构性漂移
    # u=2: 重训练/持续学习更新，对严重结构性漂移提供额外收益
    if u_t == 0:
        beta_actual = beta_0 - alpha_env * E_drift - alpha_struct * struct_drift
    elif u_t == 1:
        beta_actual = beta_0 - alpha_struct * struct_drift
    else:
        beta_actual = beta_0 + retrain_bonus * max(struct_drift - tau_retrain, 0.0)
    beta_actual = max(0.01, beta_actual)

    V_act = np.where(v_t == 1)[0]
    k_t = np.zeros(len(v_t))

    if len(V_act) == 0:
        G = V_lya * beta_actual - Y_bw * c_comm
        return G, k_t, c_comm, beta_actual, 0

    # 2. 注水算法 (逻辑不变)
    U = list(V_act)
    K_fixed = 0
    while True:
        if len(U) == 0: break
        sum_w = sum([w_t[i] for i in U])
        b = len(U) / gamma - K_fixed
        c_val = sum_w / (2 * eta * N ** 2)

        K_sum = (-b + np.sqrt(b ** 2 + 2 * sum_w / (eta * N ** 2))) / 2

        out_of_bounds = []
        for i in U:
            k_temp = w_t[i] / (2 * eta * N ** 2 * K_sum) - 1 / gamma
            if k_temp < k_min:
                out_of_bounds.append((i, k_min))
            elif k_temp > 1.0:
                out_of_bounds.append((i, 1.0))

        if not out_of_bounds:
            for i in U: k_t[i] = w_t[i] / (2 * eta * N ** 2 * K_sum) - 1 / gamma
            break
        else:
            for idx, val in out_of_bounds:
                k_t[idx] = val
                K_fixed += val
                U.remove(idx)

    # 3. 计算系统单步综合效用 U_t
    # 注意这里使用的是受到漂移影响的 beta_actual
    acc = beta_actual + sum([w_t[i] * np.log(1 + gamma * k_t[i]) for i in V_act])
    cost = eta * (sum([k_t[i] * N for i in V_act])) ** 2
    U_t = acc - cost

    G = V_lya * U_t - Y_bw * c_comm

    return G, k_t, c_comm, acc, cost
