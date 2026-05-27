import math
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ========================================================
# 🔋 严格能耗与无线充电物理模型
# ========================================================
class AdvancedEnergyModel:
    def __init__(self, P_HAP=1200.0, E_max=45000.0):
        # 空气动力学飞行功耗参数
        self.P0 = 79.856
        self.Pi = 88.6
        self.U_tip = 120.0
        self.v0 = 9.2
        self.d0 = 0.012
        self.rho = 1.225
        self.s = 0.05
        self.A = 0.503
        # 通信功耗参数 (每个关联的用户会增加基站发射功耗)
        self.p_tx = 0.15  # 150mW 每用户
        self.eta = 0.8    # WPT 转换效率
        self.P_HAP = float(P_HAP)
        self.E_max = float(E_max)
        
        # 优化控制阈值
        self.return_threshold = 0.25      # 25% 电量触发返航充电
        self.recovery_threshold = 0.90    # 充到 90% 电量视为满血复航

    def flying_power(self, velocity: np.ndarray) -> float:
        v = np.linalg.norm(velocity)
        if v < 1e-6: v = 1e-6
        P0_term = self.P0 * (1.0 + 3.0 * (v**2) / (self.U_tip**2))
        inner = math.sqrt(1.0 + (v**4) / (4.0 * (self.v0**4)))
        subtraction = inner - (v**2) / (2.0 * (self.v0**2))
        Pi_term = self.Pi * math.sqrt(max(subtraction, 1e-10))
        drag_term = 0.5 * self.d0 * self.rho * self.s * self.A * (v**3)
        return float(P0_term + Pi_term + drag_term)

    def flying_energy(self, velocity: np.ndarray, delta_t: float) -> float:
        return self.flying_power(velocity) * delta_t

    def tx_energy(self, num_covered_ue: int, delta_t: float) -> float:
        # 优化项：动态计算通信发射能耗
        return self.p_tx * num_covered_ue * delta_t

    def update_battery(self, current_energy: float, E_fly: float, E_tx: float, E_charge: float) -> float:
        new_energy = current_energy - E_fly - E_tx + E_charge
        return float(max(0.0, min(new_energy, self.E_max)))


# ========================================================
# 🌍 SAGIN 3D 仿真环境 (3 UAVs & 10 UGVs)
# ========================================================
class SAGINMultiAgentEnv:
    def __init__(self, num_uavs=3, num_ugvs=10, delta_t=0.5):
        self.num_uavs = num_uavs
        self.num_ugvs = num_ugvs
        self.delta_t = delta_t  
        self.map_length = 1000.0   
        self.hap_height = 2000.0   
        
        # 🛸 3架无人机初始位置 [X, Y, Z]，均在低空服务层
        self.uav_positions = np.array([
            [300.0, 300.0, 100.0],
            [500.0, 500.0, 100.0],
            [700.0, 700.0, 100.0]
        ])
        
        # 🏎️ 10个地面用户初始 2D 位置 [X, Y, Z=0]
        self.ugv_positions = np.zeros((num_ugvs, 3))
        for i in range(num_ugvs):
            self.ugv_positions[i] = [np.random.uniform(100, 900), np.random.uniform(100, 900), 0.0]
            
        # 给各车辆分配不同的非线性运动速度与轨迹特征参数
        self.ugv_speeds = np.random.uniform(10.0, 16.0, size=num_ugvs)
        self.ugv_phases = np.random.uniform(0, 2 * math.pi, size=num_ugvs)
        
        self.energy_model = AdvancedEnergyModel()
        # 🎯 遵照要求：初始电量全部 100% 满电
        self.uav_energies = np.array([self.energy_model.E_max] * num_uavs)
        self.time_step = 0

    def step(self, actions, uav_states):
        self.time_step += 1
        info = {}
        
        # ─── 🏎️ 10个地面用户的复杂 2D 非线性轨迹演进 ───
        for i in range(self.num_ugvs):
            if i % 3 == 0:
                # 蛇形波浪运动
                self.ugv_positions[i][0] += self.ugv_speeds[i] * self.delta_t
                self.ugv_positions[i][1] += 20.0 * math.sin(0.02 * self.ugv_positions[i][0] + self.ugv_phases[i])
            elif i % 3 == 1:
                # 对角线切入运动
                self.ugv_positions[i][0] += self.ugv_speeds[i] * self.delta_t * 0.7
                self.ugv_positions[i][1] += self.ugv_speeds[i] * self.delta_t * 0.7
            else:
                # 绕各自锚点的局部环形圆周运动
                theta = (self.ugv_speeds[i] * self.delta_t) / 100.0
                x_curr, y_curr = self.ugv_positions[i][0], self.ugv_positions[i][1]
                self.ugv_positions[i][0] = x_curr + 15.0 * math.cos(theta)
                self.ugv_positions[i][1] = y_curr + 15.0 * math.sin(theta)
            
            # 边界越界环绕保护
            if self.ugv_positions[i][0] > self.map_length or self.ugv_positions[i][1] > self.map_length:
                self.ugv_positions[i][0] = np.random.uniform(10.0, 300.0)
                self.ugv_positions[i][1] = np.random.uniform(10.0, 800.0)

        # ─── 🛰️ 3架空中无人机 3D 位置更新 ───
        uav_speeds = actions['uav_speeds']
        for m in range(self.num_uavs):
            self.uav_positions[m] += uav_speeds[m] * self.delta_t
            self.uav_positions[m][0] = np.clip(self.uav_positions[m][0], 0.0, self.map_length)
            self.uav_positions[m][1] = np.clip(self.uav_positions[m][1], 0.0, self.map_length)
            self.uav_positions[m][2] = np.clip(self.uav_positions[m][2], 100.0, 1950.0)

        # ─── 📶 10个地面用户到 3架无人机的动态最近邻关联 (Association) ───
        # 只有处于低空 SERVICE 状态且高度小于 150m 的无人机可以提供接入中继
        active_service_uavs = [m for m in range(self.num_uavs) if uav_states[m] == "SERVICE" and self.uav_positions[m][2] <= 150.0]
        
        assigned_uav_indices = []
        uav_covered_ue_counts = np.zeros(self.num_uavs, dtype=int)
        
        for n in range(self.num_ugvs):
            if len(active_service_uavs) > 0:
                # 计算到所有当前提供服务的无人机的 3D 距离
                dists = [np.linalg.norm(self.uav_positions[m] - self.ugv_positions[n]) for m in active_service_uavs]
                best_idx = active_service_uavs[np.argmin(dists)]
                assigned_uav_indices.append(best_idx)
                uav_covered_ue_counts[best_idx] += 1
            else:
                assigned_uav_indices.append(-1) # 极端情况：三机全部离场，全网中断
            
        # ─── 🔋 各能耗组件拆解与更新 ───
        uav_charge_powers = np.zeros(self.num_uavs)
        uav_fly_powers = np.zeros(self.num_uavs)
        uav_tx_powers = np.zeros(self.num_uavs)
        
        for m in range(self.num_uavs):
            # 1. 飞行功耗
            P_fly = self.energy_model.flying_power(uav_speeds[m])
            E_fly = P_fly * self.delta_t
            uav_fly_powers[m] = P_fly
            
            # 2. 发射功耗
            E_tx = self.energy_model.tx_energy(uav_covered_ue_counts[m], self.delta_t)
            uav_tx_powers[m] = E_tx / self.delta_t
            
            # 3. 优化高能超级快充功耗（优化充电时间：增大充电速率）
            h_current = self.uav_positions[m][2]
            dist_to_hap = self.hap_height - h_current
            E_charge = 0.0
            
            if uav_states[m] == "CHARGE" and dist_to_hap <= 100.0:
                # 🚀 优化项：将保底快充能力直接拉高至 3000W（大幅缩短高空充电滞留时间）
                E_charge = 3000.0 * self.delta_t 
                
            uav_charge_powers[m] = E_charge / self.delta_t
            
            # 演进电池池
            self.uav_energies[m] = self.energy_model.update_battery(
                self.uav_energies[m], E_fly, E_tx, E_charge
            )

        info['ugv_positions'] = self.ugv_positions.copy()
        info['ugv_modes'] = actions['ugv_modes'].copy()
        info['uav_positions'] = self.uav_positions.copy()
        info['assigned_uavs'] = assigned_uav_indices
        info['uav_batteries_soc'] = self.uav_energies / self.energy_model.E_max
        info['uav_charges'] = uav_charge_powers
        info['uav_fly_powers'] = uav_fly_powers
        info['uav_tx_powers'] = uav_tx_powers
        return info


# ========================================================
# 🏁 控制中心循环 (协同状态机逻辑)
# ========================================================
if __name__ == "__main__":
    # 初始化：3 架无人机，10 个地面用户
    env = SAGINMultiAgentEnv(num_uavs=3, num_ugvs=10, delta_t=0.5)
    total_steps = 1000  
    uav_states = ["SERVICE", "SERVICE", "SERVICE"]  # 初始全部处于服务状态
    
    # 历史记录用于最终分析
    history_uav_pos = [[] for _ in range(env.num_uavs)]
    history_uav_bat = [[] for _ in range(env.num_uavs)]
    history_ugv_pos = [[] for _ in range(env.num_ugvs)]
    steps_axis = []
    
    print("====================================================================================================")
    print("🛸 SAGIN 3-UAV & 10-UGV 全周期能耗监控控制台 (初始满电100% -> 自适应错峰充电优化中)")
    print("====================================================================================================")
    
    for step_idx in range(1, total_steps + 1):
        uav_actions_speed = np.zeros((env.num_uavs, 3))
        
        for m in range(env.num_uavs):
            current_soc = env.uav_energies[m] / env.energy_model.E_max
            current_pos = env.uav_positions[m]
            
            # ─── 🔄 状态机自主寻优跳转（动态产生错峰，杜绝多机同时离场） ───
            if uav_states[m] == "SERVICE":
                # 计算当前低空还有几架飞机在服务
                other_services = [i for i, s in enumerate(uav_states) if s == "SERVICE"]
                # 优化策略：如果电量低于触发现，或者虽然电量稍高(如<35%)但有两名队友在低空撑着，提前离场充电，强制错峰
                if current_soc < env.energy_model.return_threshold:
                    uav_states[m] = "RETREAT"
                elif current_soc < 0.40 and len(other_services) >= 2 and m == max(other_services):
                    uav_states[m] = "RETREAT" # 提前分流
            elif uav_states[m] == "RETREAT" and current_pos[2] >= 1900.0:
                uav_states[m] = "CHARGE"
            elif uav_states[m] == "CHARGE" and current_soc >= env.energy_model.recovery_threshold:
                uav_states[m] = "RETURN"
            elif uav_states[m] == "RETURN" and current_pos[2] <= 100.0:
                uav_states[m] = "SERVICE"
                
            # ─── 🧭 基于当前状态的航迹速度决策 ───
            if uav_states[m] == "SERVICE":
                # 无人机各自负责追踪不同的车辆密集区中心（这里简易指定：UAV 1盯车0, UAV 2盯车4, UAV 3盯车7）
                target_idx = [0, 4, 7][m]
                target_pos = env.ugv_positions[target_idx]
                dir_vector = target_pos - current_pos
                dir_vector[2] = 0  # 保持水平平飞
                norm_v = np.linalg.norm(dir_vector) + 1e-4
                uav_actions_speed[m] = (dir_vector / norm_v) * 14.0 # 伴飞速度
            elif uav_states[m] == "RETREAT":
                uav_actions_speed[m] = [0.0, 0.0, 50.0]   # 50m/s 高速爬升，最小化充电路途耽误时间
            elif uav_states[m] == "CHARGE":
                uav_actions_speed[m] = [0.0, 0.0, 0.0]    # 悬停快充
            elif uav_states[m] == "RETURN":
                uav_actions_speed[m] = [0.0, 0.0, -50.0]  # 高速俯冲复航
                
        # 随机分配 10 个车辆的本地/卸载计算模态
        sim_ugv_modes = np.random.randint(1, 4, size=env.num_ugvs)
        sim_actions = {'uav_speeds': uav_actions_speed, 'ugv_modes': sim_ugv_modes}
        
        state_snap = env.step(sim_actions, uav_states)
        
        # 归档画图数据
        steps_axis.append(step_idx)
        for m in range(env.num_uavs):
            history_uav_pos[m].append(state_snap['uav_positions'][m].copy())
            history_uav_bat[m].append(state_snap['uav_batteries_soc'][m])
        for n in range(env.num_ugvs):
            history_ugv_pos[n].append(state_snap['ugv_positions'][n].copy())

        # ─── 🖥️ 控制台日志：多无人机耗电细分监控 ───
        if step_idx % 100 == 0 or step_idx == 1:
            print(f"\n⏱️ 【Simulation Time Step: {step_idx:3d}】")
            print("  🤖 [UAV 无人机群功耗细分与电量监控]:")
            for m in range(env.num_uavs):
                pos_str = f"[{state_snap['uav_positions'][m][0]:5.1f}, {state_snap['uav_positions'][m][1]:5.1f}, {state_snap['uav_positions'][m][2]:5.1f}]"
                print(f"    -> UAV {m+1}: 状态={uav_states[m]:<7} | 3D坐标={pos_str}m | ✈️飞行功耗={state_snap['uav_fly_powers'][m]:5.1f}W | 📡通信发射功耗={state_snap['uav_tx_powers'][m]:4.1f}W | ⚡充电输入={state_snap['uav_charges'][m]:6.1f}W | 🔋SOC={state_snap['uav_batteries_soc'][m]*100:5.1f}%")
            
            # 统计 10 个地面用户的断网率
            assigned_res = state_snap['assigned_uavs']
            lost_service_count = assigned_res.count(-1)
            print(f"  🏎️ [UGV 地面用户状态摘要]: 10个用户中，已成功连上中继的有 {10 - lost_service_count} 个，暂时因飞机换班断网的有 {lost_service_count} 个。")
            print("-" * 140)

    # ========================================================
    # 🗺️ 3D 多智能体轨迹与电量循环图
    # ========================================================
    print("\n📊 正在生成 3机-10用户 架构下的多维能耗优化分析图...")
    fig = plt.figure(figsize=(15, 6))
    
    # 左图：3D 航迹图（部分展示地面用户，防画面太杂乱）
    ax_3d = fig.add_subplot(121, projection='3d')
    colors = ['royalblue', 'darkturquoise', 'forestgreen']
    for m in range(env.num_uavs):
        p_uav = np.array(history_uav_pos[m])
        ax_3d.plot(p_uav[:, 0], p_uav[:, 1], p_uav[:, 2], label=f'UAV {m+1} Optimized Track', color=colors[m], linewidth=2)
    
    # 抽样打印 3 个典型用户的 2D 复杂扭动航迹
    for n in [0, 4, 9]:
        p_ugv = np.array(history_ugv_pos[n])
        ax_3d.plot(p_ugv[:, 0], p_ugv[:, 1], p_ugv[:, 2], label=f'UGV {n+1} 2D Path', linestyle=':', alpha=0.7)
        
    ax_3d.set_title("3D Trajectories under 3-UAV & 10-UGV Architecture")
    ax_3d.set_zlim(0, 2100); ax_3d.legend(fontsize=8); ax_3d.grid(True)
    
    # 右图：总电量循环曲线（体现由于快充优化带来的电量快速回升）
    ax_bat = fig.add_subplot(122)
    for m in range(env.num_uavs):
        ax_bat.plot(steps_axis, np.array(history_uav_bat[m])*100, label=f'UAV {m+1} SOC', color=colors[m], linewidth=2)
    ax_bat.axhline(y=25, color='red', linestyle=':', label='Low Battery Limit (25%)')
    ax_bat.set_title("UAV Total Battery Cycles (Optimized Fast Charging)")
    ax_bat.set_xlabel("Steps"); ax_bat.set_ylabel("Battery Remaining (%)"); ax_bat.legend(); ax_bat.grid(True)
    
    plt.tight_layout()
    plt.show()