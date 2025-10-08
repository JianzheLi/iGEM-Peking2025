import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import seaborn as sns

# Set plotting style
sns.set(style="whitegrid", palette="muted", font_scale=1.2)
plt.rcParams['figure.figsize'] = (14, 10)

# 基础参数
params = {
    'N': 10000, 
    # 总人口
    'beta': 0.03,        # 传播率
    'rho': 0.2,          # 治疗后相对传染性
    'sigma': 0.0005,     # 潜伏率
    'gamma_I': 1/2000,   # 未治疗康复率
    'gamma_D0': 1/30,    # 初始治疗康复率
    'delta': 0.01,      # 治疗覆盖率
    'omega': 1/10,       # 免疫丧失率
    'xi': 1/20,          # 药物保护结束率
    'k': 0.001,            # 耐药发展速率
    't_max': 2000,       # 模拟总时长
    're': 0.00001          # 耐药恢复速率
}

# 初始条件
initial_conditions = [
    params['N'] - 100,  # S
    0,                  # E
    100,                # I
    0,                  # D
    0,                  # R
    params['gamma_D0']  # gamma_D
]

# 模型
def seir_sd_model(t, y, params):
    S, E, I, D, R, gamma_D = y
    N = params['N']
    beta = params['beta']
    rho = params['rho']
    sigma = params['sigma']
    gamma_I = params['gamma_I']
    delta = params['delta']
    omega = params['omega']
    xi = params['xi']
    k = params['k']
    gamma_D0 = params['gamma_D0']

    dSdt = omega * R + xi * D - (beta * S * (I + E)) / N
    dEdt = (beta * S * (I +E)) / N - sigma * E-(delta ) * E
    dIdt = sigma * E - (gamma_I + delta) * I
    dDdt = delta * (I+E) - gamma_D * D - xi * D
    dRdt = gamma_I * I + gamma_D * D - omega * R
    dgamma_Ddt = - k*gamma_D/N * (I+E) - k * gamma_D * (D)/N + params['re'] * (gamma_D0 - gamma_D)*(D+E+I)/N

    return [dSdt, dEdt, dIdt, dDdt, dRdt, dgamma_Ddt]

# 分两段参数
params_stage1 = params.copy()
params_stage2 = params.copy()

# 第20000天引入药物后参数变化（示例）
params_stage2['beta'] *= 1     # 传播率
params_stage2['delta'] *= 3      # 提高治疗覆盖率
params_stage2['gamma_D0'] *=  5    
params_stage2['k'] *= 0.5       # 耐药发展速度减半

# 第一阶段
t_split = 1000
t_span1 = (0, t_split)
t_eval1 = np.linspace(0, t_split, 500)

sol1 = solve_ivp(
    seir_sd_model,
    t_span1,
    initial_conditions,
    args=(params_stage1,),
    t_eval=t_eval1,
    method='LSODA',
    rtol=1e-6,
    atol=1e-9
)

# 第二阶段
t_span2 = (t_split, params['t_max'])
t_eval2 = np.linspace(t_split, params['t_max'], 500)
initial_conditions_stage2 = sol1.y[:, -1]
#initial_conditions_stage2[-1]*=10
print(initial_conditions_stage2)
sol2 = solve_ivp(
    seir_sd_model,
    t_span2,
    initial_conditions_stage2,
    args=(params_stage2,),
    t_eval=t_eval2,
    method='LSODA',
    rtol=1e-6,
    atol=1e-9
)

# 拼接
t = np.concatenate([sol1.t, sol2.t])
S = np.concatenate([sol1.y[0], sol2.y[0]])
E = np.concatenate([sol1.y[1], sol2.y[1]])
I = np.concatenate([sol1.y[2], sol2.y[2]])
D = np.concatenate([sol1.y[3], sol2.y[3]])
R = np.concatenate([sol1.y[4], sol2.y[4]])
gamma_D = np.concatenate([sol1.y[5], sol2.y[5]])
cumulative_infected = E + I + D + R

# 绘图
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12))

# 人群动态
ax1.plot(t, S, label='Susceptible (S)', linewidth=2.5)
ax1.plot(t, E, label='Exposed (E)', linewidth=2.5)
ax1.plot(t, I, label='Infectious (I)', linewidth=2.5)
ax1.plot(t, D, label='Treated (D)', linewidth=2.5)
ax1.plot(t, R, label='Recovered (R)', linewidth=2.5)
ax1.plot(t, cumulative_infected, '--', label='Cumulative Infected', linewidth=2.5, color='purple')

# 标记药物引入点
ax1.axvline(t_split, color='black', linestyle='--', alpha=0.8, label='Drug Introduced')

max_infected_idx = np.argmax(I)
ax1.axvline(t[max_infected_idx], color='red', linestyle='--', alpha=0.7)
ax1.text(t[max_infected_idx], I[max_infected_idx]*0.8, 
         f'Peak Infection: {I[max_infected_idx]:.0f}\nDay {t[max_infected_idx]:.0f}', 
         ha='center', fontsize=10, bbox=dict(facecolor='white', alpha=0.8))

ax1.set_title('SEIR-SD Model with Drug Resistance - Population Dynamics', fontsize=16, pad=20)
ax1.set_xlabel('Time (days)', fontsize=14)
ax1.set_ylabel('Population', fontsize=14)
ax1.legend(loc='upper right', fontsize=12)
ax1.grid(True, linestyle='--', alpha=0.7)

# 药物耐药性动态
ax2.plot(t, gamma_D, label='Treatment Recovery Rate (γ_D)', linewidth=2.5, color='darkgreen')
#ax2.plot(t, gamma_D * D, label='Treatment Effect (γ_D × D)', linewidth=2.5, color='darkorange')
ax2.axhline(params['gamma_I'], color='red', linestyle='--', label='Untreated Recovery Rate (γ_I)')

ax2.axvline(t_split, color='black', linestyle='--', alpha=0.8, label='Drug Introduced')

half_life_idx = np.argmax(gamma_D < params['gamma_D0']/2)
if half_life_idx > 0:
    ax2.axvline(t[half_life_idx], color='blue', linestyle='--', alpha=0.7)
    ax2.text(t[half_life_idx], gamma_D[0]*0.5, 
             f'Half-life: Day {t[half_life_idx]:.0f}', 
             ha='center', fontsize=10, bbox=dict(facecolor='white', alpha=0.8))

ineffective_idx = np.argmax(gamma_D < params['gamma_I'])
if ineffective_idx > 0:
    ax2.axvline(t[ineffective_idx], color='purple', linestyle='--', alpha=0.7)
    ax2.text(t[ineffective_idx], gamma_D[0]*0.3, 
             f'Treatment Ineffective: Day {t[ineffective_idx]:.0f}', 
             ha='center', fontsize=10, bbox=dict(facecolor='white', alpha=0.8))

ax2.set_title('Drug Resistance Dynamics and Treatment Effect', fontsize=16, pad=20)
ax2.set_xlabel('Time (days)', fontsize=14)
ax2.set_ylabel('Parameter Value', fontsize=14)
ax2.legend(loc='upper right', fontsize=12)
ax2.grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig('with_drug.png', dpi=300, bbox_inches='tight')
plt.show()

# 结果输出
print(f"Simulation Results ({params['t_max']} days):")
print(f"- Peak infections: {np.max(I):.0f} (on day {t[np.argmax(I)]:.0f})")
print(f"- Final cumulative infections: {cumulative_infected[-1]:.0f} ({cumulative_infected[-1]/params['N']*100:.1f}% of population)")
print(f"- Treatment recovery rate decay: from {params['gamma_D0']:.3f} to {gamma_D[-1]:.5f} ({100*(1-gamma_D[-1]/params['gamma_D0']):.1f}% decay)")
if ineffective_idx > 0:
    print(f"- Treatment becomes ineffective on day: {t[ineffective_idx]:.0f}")
else:
    print("- Treatment remains effective throughout simulation")
