import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

#假设的动力学
def decay_model(t, k):
    return np.exp(-k * t)

# ----------------------------
# 不同pH下，t与未溶解百分比
# 假设的数据，等待湿实验结果

t_arr=np.array([0, 5, 10, 15, 20]) # 时间点，单位分钟
data_by_pH = {
    1.5: {'t': t_arr, 'undissolved': np.array([1.0, 0.95, 0.88, 0.82, 0.77])},
    3.0: {'t': t_arr, 'undissolved': np.array([1.0, 0.88, 0.75, 0.65, 0.5])},
    5.0: {'t': t_arr, 'undissolved': np.array([1.0, 0.4, 0.23, 0.1, 0.0])},
    7.0: {'t': t_arr, 'undissolved': np.array([1.0, 0.1, 0.05, 0.0, 0.0])}
}

# ----------------------------
# 对每个 pH 拟合 k 值
k_list = []
pH_list = []

for pH, data in data_by_pH.items():
    t = data['t']
    y = data['undissolved']
    popt, _ = curve_fit(decay_model, t, y, bounds=(0, np.inf))
    k = popt[0]
    print(f"pH={pH}, k={k:.4f}")
    pH_list.append(pH)
    k_list.append(k)

# ----------------------------
# 拟合 k vs pH 的关系（我们用一个指数函数拟合）
def k_vs_pH(pH, a, b, c):
    return a * np.exp(b * pH) + c

pH_array = np.array(pH_list)
k_array = np.array(k_list)
popt_k, _ = curve_fit(k_vs_pH, pH_array, k_array)


pH_fit = np.linspace(1, 8, 100)
k_fit = k_vs_pH(pH_fit, *popt_k)


plt.figure()
plt.scatter(pH_array, k_array, label='Fitted k from experiments')
plt.plot(pH_fit, k_fit, label='Fitted function k(pH)', linestyle='--')
plt.xlabel('pH')
plt.ylabel('Dissolution rate constant (k)')
plt.title('Dissolution Rate Constant vs. pH')
plt.legend()
plt.grid()
plt.savefig('k_vs_pH.png')
# ----------------------------
#  Predict the dissolution curve at a specific pH
def predict_dissolution_curve(pH_val, time_points):
    k_val = k_vs_pH(pH_val, *popt_k)
    return decay_model(time_points, k_val)

# pH = 2.5
t_pred = np.linspace(0, 30, 100)
ph=2.0
undissolved_pred = predict_dissolution_curve(ph, t_pred)

plt.figure()
plt.plot(t_pred, undissolved_pred, label=f"Predicted curve at pH = {ph}")
plt.xlabel('Time (min)')
plt.ylabel('Undissolved fraction')
plt.title(f"Predicted Drug Release Curve at pH = {ph}")
plt.grid()
plt.legend()
plt.savefig(f"predicted_curve_pH_{ph}.png")
