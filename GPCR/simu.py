import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# 定义模型参数
params = {
    # Part 1: Receptor activation
    'K_hs': 0.1,    # binding rate (μM⁻¹s⁻¹)
    'K_sh': 0.01,   # dissociation rate (s⁻¹)
    
    # Part 2: Scaffold formation
    'K_ss': 0.05,   # Ste5 dimerization (μM⁻¹s⁻¹)
    'K_sg': 0.1,    # Gβγ-Ste5 binding (μM⁻¹s⁻¹)
    'K_gss1': 0.02, # Gβγ + 2Ste5 -> complex (μM⁻²s⁻¹)
    'K_gss2': 0.1,  # Gβγ-Ste5 + Ste5 -> complex (μM⁻¹s⁻¹)
    'K_gssg1': 0.05,# 2(Gβγ-Ste5) -> complex (μM⁻¹s⁻¹)
    'K_gssg2': 0.1, # Gβγ-Ste5-Ste5 + Gβγ -> complex (μM⁻¹s⁻¹)
    
    # Part 3: Cascade reaction
    'K_onSte5_Ste11': 0.5,  # association rate (μM⁻¹s⁻¹)
    'K_offSte5_Ste11': 0.01,# dissociation rate (s⁻¹)
    'K_cat_Ste11': 0.5,     # phosphorylation rate (s⁻¹)
    'K_cat_Ste7': 0.4,      # phosphorylation rate (s⁻¹)
    'K_cat_Fus3': 0.3,      # phosphorylation rate (s⁻¹)
    'K_dephos': 0.05,       # dephosphorylation rate (s⁻¹)
    
    # Part 4: Ste12 activation
    'K_in': 0.1,    # nuclear import rate (s⁻¹)
    'K_out': 0.05,  # nuclear export rate (s⁻¹)
    'K_phos1': 0.2, # Dig1 phosphorylation (μM⁻¹s⁻¹)
    'K_phos2': 0.2, # Dig2 phosphorylation (μM⁻¹s⁻¹)
    'K_dephos1': 0.1, # Dig1 dephosphorylation (μM⁻¹s⁻¹)
    'K_dephos2': 0.1, # Dig2 dephosphorylation (μM⁻¹s⁻¹)
    'K_bind': 0.01, # Ste12-Dig rebinding (μM⁻²s⁻¹)
    
    # Part 5: Transcription and translation
    'K_trans': 0.5, # transcription rate (s⁻¹)
    'K_transl': 0.2, # translation rate (s⁻¹)
    'K_deg_mRNA': 0.05, # mRNA degradation (s⁻¹)
    'K_deg_protein': 0.01, # protein degradation (s⁻¹)
    'K_d': 0.5,     # dissociation constant (μM)
    'n': 2,         # Hill coefficient
    
    # Constant concentrations
    'Ste11_total': 2.0,    # total Ste11 (μM)
    'Ste7_total': 2.0,     # total Ste7 (μM)
    'Msg5': 0.5,           # phosphatase (μM)
    'Ptp2': 0.5            # phosphatase (μM)
}

# 设置模拟时间
t_span = (0, 300)  # 0 to 300 seconds
t_eval = np.linspace(t_span[0], t_span[1], 1000)

# 初始浓度 (μM)
initial_concentrations = {
    'hH3R': 1.0,
    'Ste2': 0.5,
    'hH3R_Ste2': 0.0,
    'Ste5': 1.0,
    'Ste5_Ste5': 0.0,
    'Gβγ_Ste5': 0.0,
    'Gβγ_Ste5_Ste5': 0.0,
    'Gβγ_Ste5_Ste5_Gβγ': 0.0,
    'Ste5offSte11': 1.0,
    'Ste11off': 1.0,
    'Ste5_Ste11': 0.0,
    'Ste11pS': 0.0,
    'Ste11pSpS': 0.0,
    'Ste11pSpSpT': 0.0,
    'Ste7': 1.0,
    'Ste7pS': 0.0,
    'Ste7pSpT': 0.0,
    'Fus3': 1.0,
    'Fus3pY': 0.0,
    'Fus3pYpT': 0.0,
    'Fus3cyt': 0.0,
    'Fus3nuc': 0.0,
    'Ste12_Dig': 1.0,
    'Dig1': 0.0,
    'Dig2': 0.0,
    'Ste12': 0.0,
    'mRNAA': 0.0,
    'ProteinA': 0.0
}

# Part 1: Receptor activation
def part1(t, y, params):
    hH3R, Ste2, hH3R_Ste2 = y
    
    # Reactions
    d_hH3R = -params['K_hs'] * hH3R * Ste2 + params['K_sh'] * hH3R_Ste2
    d_Ste2 = -params['K_hs'] * hH3R * Ste2 + params['K_sh'] * hH3R_Ste2
    d_hH3R_Ste2 = params['K_hs'] * hH3R * Ste2 - params['K_sh'] * hH3R_Ste2
    
    return [d_hH3R, d_Ste2, d_hH3R_Ste2]

# Part 2: Scaffold formation
def part2(t, y, params, Gβγ):
    Ste5, Ste5_Ste5, Gβγ_Ste5, Gβγ_Ste5_Ste5, Gβγ_Ste5_Ste5_Gβγ = y
    
    # Reactions
    d_Ste5 = (-2 * params['K_ss'] * Ste5**2 
              - params['K_sg'] * Gβγ * Ste5 
              - 2 * params['K_gss1'] * Gβγ * Ste5**2 
              - params['K_gss2'] * Gβγ_Ste5 * Ste5)
    
    d_Ste5_Ste5 = params['K_ss'] * Ste5**2 - params['K_gss1'] * Gβγ * Ste5_Ste5
    
    d_Gβγ_Ste5 = (params['K_sg'] * Gβγ * Ste5 
                  - params['K_gss2'] * Gβγ_Ste5 * Ste5 
                  - 2 * params['K_gssg1'] * Gβγ_Ste5**2)
    
    d_Gβγ_Ste5_Ste5 = (params['K_gss1'] * Gβγ * Ste5**2 
                       + params['K_gss2'] * Gβγ_Ste5 * Ste5 
                       - params['K_gssg2'] * Gβγ_Ste5_Ste5 * Gβγ)
    
    d_Gβγ_Ste5_Ste5_Gβγ = (params['K_gssg1'] * Gβγ_Ste5**2 
                           + params['K_gssg2'] * Gβγ_Ste5_Ste5 * Gβγ)
    
    return [d_Ste5, d_Ste5_Ste5, d_Gβγ_Ste5, d_Gβγ_Ste5_Ste5, d_Gβγ_Ste5_Ste5_Gβγ]

# Part 3: Cascade reaction
def part3(t, y, params):
    Ste5offSte11, Ste11off, Ste5_Ste11, Ste11pS, Ste11pSpS, Ste11pSpSpT, Ste7, Ste7pS, Ste7pSpT, Fus3, Fus3pY, Fus3pYpT = y
    
    # Ste5-Ste11 binding
    d_Ste5offSte11 = -params['K_onSte5_Ste11'] * Ste5offSte11 * Ste11off + params['K_offSte5_Ste11'] * Ste5_Ste11
    d_Ste11off = -params['K_onSte5_Ste11'] * Ste5offSte11 * Ste11off + params['K_offSte5_Ste11'] * Ste5_Ste11
    d_Ste5_Ste11 = params['K_onSte5_Ste11'] * Ste5offSte11 * Ste11off - params['K_offSte5_Ste11'] * Ste5_Ste11
    
    # Ste11 phosphorylation cascade
    d_Ste11pS = params['K_cat_Ste11'] * Ste5_Ste11 - params['K_cat_Ste11'] * Ste11pS
    d_Ste11pSpS = params['K_cat_Ste11'] * Ste11pS - params['K_cat_Ste11'] * Ste11pSpS
    d_Ste11pSpSpT = params['K_cat_Ste11'] * Ste11pSpS - params['K_dephos'] * Ste11pSpSpT
    
    # Ste7 phosphorylation
    d_Ste7pS = (params['K_cat_Ste7'] * (Ste11pS + Ste11pSpS + Ste11pSpSpT) * Ste7 
                - params['K_cat_Ste7'] * Ste7pS)
    d_Ste7pSpT = params['K_cat_Ste7'] * Ste7pS - params['K_dephos'] * Ste7pSpT
    d_Ste7 = -params['K_cat_Ste7'] * (Ste11pS + Ste11pSpS + Ste11pSpSpT) * Ste7
    
    # Fus3 phosphorylation
    d_Fus3pY = (params['K_cat_Fus3'] * (Ste7pS + Ste7pSpT) * Fus3 
                - params['K_cat_Fus3'] * Fus3pY)
    d_Fus3pYpT = params['K_cat_Fus3'] * Fus3pY - params['K_dephos'] * Fus3pYpT
    d_Fus3 = -params['K_cat_Fus3'] * (Ste7pS + Ste7pSpT) * Fus3
    
    return [d_Ste5offSte11, d_Ste11off, d_Ste5_Ste11, 
            d_Ste11pS, d_Ste11pSpS, d_Ste11pSpSpT,
            d_Ste7, d_Ste7pS, d_Ste7pSpT,
            d_Fus3, d_Fus3pY, d_Fus3pYpT]

# Part 4: Ste12 activation
def part4(t, y, params, Fus3p_total):
    Fus3cyt, Fus3nuc, Ste12_Dig, Dig1, Dig2, Ste12 = y
    
    # Nuclear transport
    d_Fus3cyt = -params['K_in'] * Fus3cyt + params['K_out'] * Fus3nuc
    d_Fus3nuc = params['K_in'] * Fus3cyt - params['K_out'] * Fus3nuc
    
    # Phosphorylation of Dig proteins
    d_Dig1 = params['K_phos1'] * Fus3nuc * Ste12_Dig - params['K_dephos1'] * params['Msg5'] * Dig1
    d_Dig2 = params['K_phos2'] * Fus3nuc * Ste12_Dig - params['K_dephos2'] * params['Ptp2'] * Dig2
    
    # Ste12 activation
    d_Ste12_Dig = -(params['K_phos1'] + params['K_phos2']) * Fus3nuc * Ste12_Dig
    d_Ste12 = (params['K_phos1'] + params['K_phos2']) * Fus3nuc * Ste12_Dig - params['K_bind'] * Ste12 * Dig1 * Dig2
    
    return [d_Fus3cyt, d_Fus3nuc, d_Ste12_Dig, d_Dig1, d_Dig2, d_Ste12]

# Part 5: Transcription and translation
def part5(t, y, params, Ste12):
    mRNAA, ProteinA = y
    
    # Transcription (Hill function)
    transcription = params['K_trans'] * (Ste12**params['n'] / (Ste12**params['n'] + params['K_d']**params['n']))
    
    d_mRNAA = transcription - params['K_deg_mRNA'] * mRNAA
    d_ProteinA = params['K_transl'] * mRNAA - params['K_deg_protein'] * ProteinA
    
    return [d_mRNAA, d_ProteinA]

# 运行完整模拟
def full_model(t, y, params):
    # 解包所有状态变量
    states = {name: y[i] for i, name in enumerate(initial_concentrations)}
    
    # Part 1: Receptor activation
    y1 = [states['hH3R'], states['Ste2'], states['hH3R_Ste2']]
    dy1 = part1(t, y1, params)
    
    # Part 2: Scaffold formation (Gβγ来自激活的受体)
    Gβγ = states['hH3R_Ste2']  # 假设Gβγ浓度等于激活的受体复合物
    y2 = [states['Ste5'], states['Ste5_Ste5'], states['Gβγ_Ste5'], 
          states['Gβγ_Ste5_Ste5'], states['Gβγ_Ste5_Ste5_Gβγ']]
    dy2 = part2(t, y2, params, Gβγ)
    
    # Part 3: Cascade reaction
    y3 = [states['Ste5offSte11'], states['Ste11off'], states['Ste5_Ste11'],
          states['Ste11pS'], states['Ste11pSpS'], states['Ste11pSpSpT'],
          states['Ste7'], states['Ste7pS'], states['Ste7pSpT'],
          states['Fus3'], states['Fus3pY'], states['Fus3pYpT']]
    dy3 = part3(t, y3, params)
    
    # Part 4: Ste12 activation (使用磷酸化的Fus3总量)
    Fus3p_total = states['Fus3pY'] + states['Fus3pYpT']
    y4 = [states['Fus3cyt'], states['Fus3nuc'], states['Ste12_Dig'],
          states['Dig1'], states['Dig2'], states['Ste12']]
    dy4 = part4(t, y4, params, Fus3p_total)
    
    # Part 5: Transcription and translation
    y5 = [states['mRNAA'], states['ProteinA']]
    dy5 = part5(t, y5, params, states['Ste12'])
    
    # 组合所有导数
    dydt = dy1 + dy2 + dy3 + dy4 + dy5
    return dydt

# 初始状态向量
initial_state = list(initial_concentrations.values())

# 运行模拟
sol = solve_ivp(full_model, t_span, initial_state, args=(params,), 
                t_eval=t_eval, method='BDF', rtol=1e-6, atol=1e-8)

# 提取结果
results = {name: sol.y[i] for i, name in enumerate(initial_concentrations.keys())}

# 可视化关键结果
plt.figure(figsize=(15, 12))

# 受体激活
plt.subplot(3, 2, 1)
plt.plot(sol.t, results['hH3R_Ste2'], 'b-', label='hH3R-Ste2')
plt.plot(sol.t, results['hH3R'], 'r--', label='hH3R')
plt.plot(sol.t, results['Ste2'], 'g-.', label='Ste2')
plt.title('Receptor Activation')
plt.ylabel('Concentration (μM)')
plt.legend()

# 支架形成
plt.subplot(3, 2, 2)
plt.plot(sol.t, results['Ste5'], 'b-', label='Ste5')
plt.plot(sol.t, results['Ste5_Ste5'], 'r--', label='Ste5-Ste5')
plt.plot(sol.t, results['Gβγ_Ste5_Ste5'], 'g-.', label='Gβγ-Ste5-Ste5')
plt.plot(sol.t, results['Gβγ_Ste5_Ste5_Gβγ'], 'm:', label='Gβγ-Ste5-Ste5-Gβγ')
plt.title('Scaffold Formation')
plt.legend()

# 磷酸化级联
plt.subplot(3, 2, 3)
plt.plot(sol.t, results['Ste11pSpSpT'], 'b-', label='Ste11pSpSpT')
plt.plot(sol.t, results['Ste7pSpT'], 'r--', label='Ste7pSpT')
plt.plot(sol.t, results['Fus3pYpT'], 'g-.', label='Fus3pYpT')
plt.title('Phosphorylation Cascade')
plt.ylabel('Concentration (μM)')
plt.legend()

# Ste12激活
plt.subplot(3, 2, 4)
plt.plot(sol.t, results['Ste12'], 'b-', label='Free Ste12')
plt.plot(sol.t, results['Ste12_Dig'], 'r--', label='Ste12-Dig Complex')
plt.title('Transcription Factor Activation')
plt.legend()

# 基因表达
plt.subplot(3, 2, 5)
plt.plot(sol.t, results['mRNAA'], 'b-', label='mRNAA')
plt.plot(sol.t, results['ProteinA'], 'r--', label='ProteinA')
plt.title('Gene Expression')
plt.xlabel('Time (s)')
plt.ylabel('Concentration (μM)')
plt.legend()

# 信号传导动态
plt.subplot(3, 2, 6)
plt.plot(sol.t, results['hH3R_Ste2'], 'b-', label='Receptor')
plt.plot(sol.t, results['Gβγ_Ste5_Ste5_Gβγ'], 'r--', label='Scaffold Complex')
plt.plot(sol.t, results['Fus3pYpT'], 'g-.', label='Fus3pYpT')
plt.plot(sol.t, results['ProteinA'], 'm:', label='ProteinA')
plt.title('Signal Transduction Dynamics')
plt.xlabel('Time (s)')
plt.legend()

plt.tight_layout()
plt.savefig('gpcr_model_results.png', dpi=300)
plt.show()