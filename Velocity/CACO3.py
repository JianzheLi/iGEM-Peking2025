import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# ----------------------------- 
# 基本参数设定 
# ----------------------------- 
m_microsphere = 1e-7          # 微球质量 (kg)
rho_CaCO3 = 2.71e3            # 碳酸钙密度 (kg/m^3)
r_microsphere = ((3 * m_microsphere) / (4 * np.pi * 3000)) ** (1/3)  # 假设主体密度
print("Microsphere radius (mm):", r_microsphere) 

pho=1.1
mu = 0.001                # 胃液动力粘度 (Pa·s)
area_half = 4 * np.pi * r_microsphere**2  # 涂层面积
thickness_CaCO3 = 2e-5        # 碳酸钙层厚度 (m)
m_CaCO3 = rho_CaCO3 * area_half * thickness_CaCO3  # 碳酸钙质量
print("Calcium carbonate mass (kg):", m_CaCO3)

# ----------------------------- 
# CO2生成估算（碳酸钙+酸 -> Ca2+ + CO2 + H2O） 
# ----------------------------- 
reaction_time = 60  # 反应持续时间 (s)
M_CaCO3 = 100.09     # CaCO3摩尔质量 (g/mol)
M_CO2 = 44.01        # CO2摩尔质量 (g/mol)
Vm = 22.4e-3         # CO2摩尔体积 (m^3/mol) 标准状况

# 1 mol CaCO3 -> 1 mol CO2
n_CaCO3 = m_CaCO3 / M_CaCO3    # (kg) / (g/mol) = (1000 g/kg)/M_CaCO3
n_CaCO3 = n_CaCO3 * 1000       # 单位换算，kg变g

n_CO2 = n_CaCO3                # 摩尔数相等
m_CO2_total = n_CO2 * M_CO2 / 1000  # 转为kg
dm_CO2_dt = m_CO2_total / reaction_time  # CO2释放速率 (kg/s)

# 计算CO2气体体积（假设标准状态 STP） 
V_CO2_total = n_CO2 * Vm  # m^3
print("Total CO2 mass (kg):", m_CO2_total)
print("CO2 release rate (kg/s):", dm_CO2_dt)
print("Total CO2 volume at STP (mL):", V_CO2_total * 1e6)

# ----------------------------- 
# 化学反应推力 
# ----------------------------- 
v_exhaust = 1  # 可先与Mg一致，需根据实际测量更换
F_thrust = dm_CO2_dt * v_exhaust

# ----------------------------- 
# 微球运动的微分方程（1维简化） 
# ----------------------------- 
def dvdt(t, v): 
    F = F_thrust if t <= reaction_time else 0
    F_drag = 6 * np.pi * pho * mu * r_microsphere * v  # 斯托克斯阻力
    a = (F - F_drag) / m_microsphere
    return [a]

# ----------------------------- 
# 数值求解速度随时间变化 
# ----------------------------- 
t_span = (0, reaction_time)
t_eval = np.linspace(*t_span, 1000)
sol = solve_ivp(dvdt, t_span, [0], t_eval=t_eval)

plt.figure(figsize=(8, 5))
plt.plot(sol.t, sol.y[0]*1000, label='speed (mm/s)', color='blue')
plt.axvline(reaction_time, color='red', linestyle='--', label='stop time (s)')
plt.title('Predicting the speed of a microsphere in gastric fluid (CaCO$_3$)')
plt.xlabel('time (s)')
plt.ylabel('speed (mm/s)')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig('CaCO3_speed_prediction.png', dpi=300)
plt.show()

# ----------------------------- 
# CO2体积流率和累计体积 
# ----------------------------- 
M_CO2 = 44.01e-3  # CO2摩尔质量 (kg/mol)
dVdt = (dm_CO2_dt / M_CO2) * Vm   # m^3/s
V_rate = np.where(t_eval <= reaction_time, dVdt * 1e6, 0)  # mL/s
V_cumulative = np.cumsum(V_rate * (t_eval[1] - t_eval[0]))  # mL

plt.figure(figsize=(8, 5))
plt.plot(t_eval, V_rate, label='CO2 release rate (mL/s)')
plt.plot(t_eval, V_cumulative, label='Cumulative CO2 released (mL)', color='orange')

# 标注最终释放体积
final_volume = V_cumulative[-1]
plt.annotate(f'Total ≈ {final_volume:.2f} mL',
             xy=(t_eval[-1], final_volume),
             xytext=(-100, 30),
             textcoords='offset points',
             arrowprops=dict(arrowstyle='->', color='black'),
             fontsize=10, color='red')

plt.title('CO$_2$ Release Dynamics (Volume)')
plt.xlabel('time (s)')
plt.ylabel('CO$_2$ Volume (mL)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('CO2_release_volume.png', dpi=300)
plt.show()