from logging import config
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pyat.ccf import iccf, iccf_mc
import emcee
import corner
import os
from pathlib import Path
import yaml

CONFIG_PATH = Path("configs/Mrk509.yaml")

def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    return config

config = load_config(CONFIG_PATH)
obj_name = config['obj_name']
# 保存原来的 savefig 函数
_old_savefig = plt.savefig

def _smart_savefig(path, *args, **kwargs):
    """自动创建目录的 savefig 替代函数"""
    directory = os.path.dirname(path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
    _old_savefig(path, *args, **kwargs)

# 替换掉原 savefig
plt.savefig = _smart_savefig

def tau_model(params, wavelength):
    """Lambda-tau模型"""
    tau0 = params[0]
    beta = params[1] if len(params) > 1 else params[0]  # beta可能是固定值
    return tau0 * ((wavelength / lambda0) ** beta - 1.0)

def log_likelihood(params, wavelength, tau_obs, tau_err, beta_fixed=None):
    """对数似然函数"""
    if beta_fixed is not None:
        model_params = [params[0], beta_fixed]
    else:
        model_params = params
    
    tau_pred = tau_model(model_params, wavelength)
    chi2 = np.sum(((tau_obs - tau_pred) / tau_err) ** 2)
    return -0.5 * chi2

def log_prior(params, beta_fixed=None):
    """先验分布"""
    tau0 = params[0]
    if beta_fixed is None:
        if len(params) > 1:
            beta = params[1]
            # tau0: -20 到 20, beta: 0.5 到 3.0
            if -20 < tau0 < 20 and 1 < beta < 3.0:
                return 0.0
        else:
            if -20 < tau0 < 20:
                return 0.0
    else:
        if -20 < tau0 < 20:
            return 0.0
    return -np.inf

def log_probability(params, wavelength, tau_obs, tau_err, beta_fixed=None):
    """后验概率"""
    lp = log_prior(params, beta_fixed)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(params, wavelength, tau_obs, tau_err, beta_fixed)

def fit_lambda_tau_mcmc(wavelength, tau_obs, tau_err, beta_fixed=None, 
                        nwalkers=32, nsteps=5000, burn_in=1000):
    """使用MCMC拟合lambda-tau关系"""
    
    # 初始化参数
    if beta_fixed is not None:
        ndim = 1
        # 初始猜测
        p0 = np.array([np.mean(tau_obs)])
        # 添加随机扰动
        pos = p0 + 1e-2 * np.random.randn(nwalkers, ndim)
    else:
        ndim = 2
        p0 = np.array([np.mean(tau_obs), 1.5])
        pos = p0 + 1e-2 * np.random.randn(nwalkers, ndim)
    
    # 设置sampler
    sampler = emcee.EnsembleSampler(
        nwalkers, ndim, log_probability,
        args=(wavelength, tau_obs, tau_err, beta_fixed)
    )
    
    # 运行MCMC
    print(f"Running MCMC with beta_fixed={beta_fixed}...")
    sampler.run_mcmc(pos, nsteps, progress=True)
    
    # 去除burn-in
    samples = sampler.get_chain(discard=burn_in, flat=True)
    
    # 计算结果
    tau0_mcmc = np.median(samples[:, 0])
    tau0_err = np.std(samples[:, 0])
    
    if beta_fixed is None:
        beta_mcmc = np.median(samples[:, 1])
        beta_err = np.std(samples[:, 1])
        params_best = [tau0_mcmc, beta_mcmc]
    else:
        beta_mcmc = beta_fixed
        beta_err = 0.0
        params_best = [tau0_mcmc, beta_fixed]
    
    # 计算chi^2
    tau_pred = tau_model(params_best, wavelength)
    chi2 = np.sum(((tau_obs - tau_pred) / tau_err) ** 2)
    dof = len(wavelength) - ndim
    
    results = {
        'tau0': tau0_mcmc,
        'tau0_err': tau0_err,
        'beta': beta_mcmc,
        'beta_err': beta_err,
        'chi2': chi2,
        'dof': dof,
        'samples': samples,
        'params_best': params_best
    }
    
    return results

# ============== CCF分析部分 ==============

df = pd.read_csv('data/Mrk509.csv')
filters = df['Filter'].unique()
objects = df['Object'].unique()

# 存储CCF结果
tau_cent_arr = pd.DataFrame(index=objects, columns=filters, dtype=object)

# 存储tau_cent的统计值用于lambda-tau拟合
tau_cent_med = pd.DataFrame(index=objects, columns=filters, dtype=object)
tau_cent_err_minus = pd.DataFrame(index=objects, columns=filters, dtype=object)
tau_cent_err_plus = pd.DataFrame(index=objects, columns=filters, dtype=object)

for obj in objects:
    # 取 W2 作为驱动
    sub_drive = df[(df['Object'] == obj) & (df['Filter'] == 'W2')]
    Mjd_drive = sub_drive['MJD'].to_numpy(float)
    flux_drive = sub_drive['Flux'].to_numpy(float)
    err_drive = sub_drive['Error'].to_numpy(float)
    
    n_flt = len(filters)
    fig, axes = plt.subplots(n_flt, 2, figsize=(12, 2.5*n_flt))
    
    for i, flt in enumerate(filters):
        sub = df[(df['Object'] == obj) & (df['Filter'] == flt)]
        Mjd = sub['MJD'].to_numpy(float)
        flux = sub['Flux'].to_numpy(float)
        err = sub['Error'].to_numpy(float)
        
        # 如果数据太少就跳过
        if len(Mjd) < 2 or len(Mjd_drive) < 2:
            continue
        
        # 左侧子图：光变曲线
        ax_left = axes[i, 0] if n_flt > 1 else axes[0]
        ax_left.errorbar(Mjd, flux, yerr=err, fmt='o', markersize=4, capsize=3)
        ax_left.set_ylabel(f'{flt} Flux', fontsize=10)
        if i == n_flt - 1:
            ax_left.set_xlabel('MJD', fontsize=10)
        ax_left.grid(True, alpha=0.3)
        
        # 调用 iccf 获取CCF曲线
        tau, ccf, _, tau_peak_single, tau_cent_single = iccf(
            Mjd_drive, flux_drive,
            Mjd, flux,
            ntau=100, tau_beg=-15, tau_end=15
        )
        
        # 调用 iccf_mc 获取tau_cent分布
        ccf_arr, tau_peak_arr, tau_cent_arr.loc[obj, flt] = iccf_mc(
            Mjd_drive, flux_drive, err_drive,
            Mjd, flux, err,
            nsim=1000, ntau=100, tau_beg=-15, tau_end=15
        )
        
        # 右侧子图：CCF和tau_cent分布
        ax_right = axes[i, 1] if n_flt > 1 else axes[1]
        
        # 绘制CCF曲线（左y轴）
        ax_right.plot(tau, ccf, 'k-', linewidth=1.5, label='CCF')
        ax_right.set_ylabel('CCF', fontsize=10)
        ax_right.set_xlabel('Time Lag (days)', fontsize=10)
        
        # 创建右y轴绘制tau_cent分布
        ax_right2 = ax_right.twinx()
        
        # 绘制tau_cent直方图
        counts, bins, _ = ax_right2.hist(tau_cent_arr.loc[obj, flt], bins=30, alpha=0.6, 
                                          color='orange' if np.mean(tau_cent_arr.loc[obj, flt]) > 0 else 'blue',
                                          density=True, label='τ_cent distribution')
        
        # 标记tau_cent平均值
        med_tau = np.median(tau_cent_arr.loc[obj, flt])
        tau_cent_med.loc[obj, flt] = med_tau
        lo, hi = np.percentile(tau_cent_arr.loc[obj, flt], [16, 84])
        err_minus = med_tau - lo
        err_plus  = hi - med_tau
        tau_cent_err_minus.loc[obj, flt] = err_minus
        tau_cent_err_plus.loc[obj, flt] = err_plus
        tau_cent_avg = np.mean(tau_cent_arr.loc[obj, flt])
        tau_cent_err = np.std(tau_cent_arr.loc[obj, flt])
        ax_right.axvline(med_tau, color='red', linestyle='--', linewidth=1.5, 
                        label=f'τ_cent = {med_tau:.2f}±{tau_cent_err:.2f}')
        ax_right2.axvline(x=lo,  color='red', linestyle='--', linewidth=1.2, alpha=0.7)       # 下界
        ax_right2.axvline(x=hi,  color='red', linestyle='--', linewidth=1.2, alpha=0.7)       # 上界
        ax_right2.axvspan(lo, hi, color='red', alpha=0.12) 
        
        ax_right2.set_ylabel('Probability Density', fontsize=10)
        ax_right.legend(loc='upper left', fontsize=8)
        ax_right.grid(True, alpha=0.3)
        
        
        print(f'{obj}-{flt}: τ_cent = {med_tau:.2f} -{err_minus:.2f}/+{err_plus:.2f} days (68% 等尾)')
    
    plt.tight_layout()
    plt.savefig(f'fig/timeseries/{obj}_ccf_analysis.png', dpi=300, bbox_inches='tight')


# ============== 新增：Lambda-Tau 拟合部分 ==============

# 定义各滤光片的中心波长（单位：Angstrom）
WAVELENGTHS = {
    'W2': 1928,    # UVW2
    'M2': 2236,    # UVM2
    'W1': 2600,    # UVW1
    'U': 3467,     # Swift,U
    # 'u':3540,     # SDSS,u
    'B': 4392,     # Swift,B
    'g': 4770,     
    'V': 5468,     # Swift,V
    # 'lV': 5383,
    'r': 6215,     
    'i': 7545,     
    'z': 8700      
}

# 定义巡天/空间设备的滤光片
SURVEY_FILTERS = ['U', 
                  'B',
                  'W1', 
                  'W2', 
                  'M2', 
                  'SV']
# Lambda-tau拟合模型：tau = tau0 * [(lambda/lambda0)^beta - 1]
lambda0 = 1928 # 参考波长

# 对每个对象执行lambda-tau拟合
for obj in objects:
    print(f"\n{'='*60}")
    print(f"Fitting lambda-tau relation for {obj}")
    print(f"{'='*60}\n")
    
    # 准备数据
    wavelengths = []
    tau_values = []
    tau_errs = []
    filter_names = []
    is_survey = []
    tau_cent_SV = 1.40
    tau_ref = {'SV': 1.40,
               'r': 1.38}
    err_ref = {'SV': 1.35,
               'r': 0.27}

    for flt in filters:
        if flt in WAVELENGTHS and pd.notna(tau_cent_med.loc[obj, flt]):
            wavelengths.append(WAVELENGTHS[flt])
            # if flt in ['SV', 'r']:
            #     tau_values.append(tau_ref[flt])
            #     tau_errs.append(err_ref[flt])
            # else:
            tau_values.append(tau_cent_med.loc[obj, flt])
            tau_errs.append((tau_cent_err_minus.loc[obj, flt] + tau_cent_err_plus.loc[obj, flt]) / 2)
            filter_names.append(flt)
            is_survey.append(flt in SURVEY_FILTERS)
    
    if len(wavelengths) < 3:
        print(f"Not enough data points for {obj}, skipping...")
        continue
    
    wavelengths = np.array(wavelengths)
    tau_values = np.array(tau_values)
    tau_errs = np.array(tau_errs)
    is_survey = np.array(is_survey)
    
    # 执行三种拟合
    fit_results = {}
    
    # 1. beta = 4/3 (固定)
    print("\n1. Fitting with β = 4/3 (fixed)...")
    fit_results['beta_4_3'] = fit_lambda_tau_mcmc(
        wavelengths, tau_values, tau_errs, 
        beta_fixed=4.0/3.0, nwalkers=32, nsteps=3000, burn_in=500
    )
    
    # 2. beta = 2 (固定)
    print("\n2. Fitting with β = 2 (fixed)...")
    fit_results['beta_2'] = fit_lambda_tau_mcmc(
        wavelengths, tau_values, tau_errs, 
        beta_fixed=2.0, nwalkers=32, nsteps=3000, burn_in=500
    )
    
    # 3. beta作为自由参数
    print("\n3. Fitting with β as free parameter...")
    fit_results['beta_free'] = fit_lambda_tau_mcmc(
        wavelengths, tau_values, tau_errs, 
        beta_fixed=None, nwalkers=32, nsteps=3000, burn_in=500
    )
    
    # 绘制lambda-tau关系图
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fit_types = [
        ('beta_4_3', 'β = 4/3 (fixed)'),
        ('beta_2', 'β = 2 (fixed)'),
        ('beta_free', 'β free')
    ]
    
    for idx, (fit_key, fit_label) in enumerate(fit_types):
        ax = axes[idx]
        result = fit_results[fit_key]
        
        # 绘制数据点
        # 巡天/空间设备用方块
        survey_mask = is_survey
        ax.errorbar(wavelengths[survey_mask], tau_values[survey_mask], 
                   yerr=tau_errs[survey_mask], 
                   fmt='s', color='blue', markersize=8, capsize=3, 
                   label='Survey/Space', alpha=0.7)
        
        # 地面设备用圆圈
        ground_mask = ~is_survey
        ax.errorbar(wavelengths[ground_mask], tau_values[ground_mask], 
                   yerr=tau_errs[ground_mask], 
                   fmt='o', color='orange', markersize=8, capsize=3, 
                   label='Ground-based', alpha=0.7)
        
        # 绘制拟合曲线
        lambda_model = np.linspace(1500, 10000, 200)
        tau_fit = tau_model(result['params_best'], lambda_model)
        ax.plot(lambda_model, tau_fit, 'k-', linewidth=2, label='Best fit')
        
        # 绘制不确定性带（使用MCMC样本）
        if fit_key == 'beta_free':
            tau_samples = np.array([tau_model([s[0], s[1]], lambda_model) 
                                   for s in result['samples'][::100]])
        else:
            tau_samples = np.array([tau_model([s[0], result['beta']], lambda_model) 
                                   for s in result['samples'][::100]])
        
        tau_16 = np.percentile(tau_samples, 16, axis=0)
        tau_84 = np.percentile(tau_samples, 84, axis=0)
        ax.fill_between(lambda_model, tau_16, tau_84, alpha=0.2, color='gray')
        
        # 设置标签和标题
        ax.set_xlabel('Rest Wavelength (Å)', fontsize=12)
        ax.set_ylabel('Time Lag (days)', fontsize=12)
        ax.set_title(fit_label, fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)
        
        # 添加拟合结果文本
        if result['beta_err'] > 0:
            text = f"τ₀ = {result['tau0']:.2f} ± {result['tau0_err']:.2f} days\n"
            text += f"β = {result['beta']:.3f} ± {result['beta_err']:.3f}\n"
        else:
            text = f"τ₀ = {result['tau0']:.2f} ± {result['tau0_err']:.2f} days\n"
            text += f"β = {result['beta']:.3f} (fixed)\n"
        text += f"χ² = {result['chi2']:.2f}\n"
        text += f"dof = {result['dof']}\n"
        text += f"χ²/dof = {result['chi2']/result['dof']:.2f}"
        
        ax.text(0.05, 0.95, text, transform=ax.transAxes, 
               fontsize=10, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle(f'{obj}: λ-τ Relationship', fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'fig/timeseries/{obj}_lambda_tau_fit.png', dpi=300, bbox_inches='tight')
    
    # 打印结果表格
    print(f"\n{'='*80}")
    print(f"Fitting Results for {obj}")
    print(f"{'='*80}")
    print(f"{'Model':<20} {'τ₀ (days)':<25} {'β':<25} {'χ²/dof':<15}")
    print(f"{'-'*80}")
    
    for fit_key, fit_label in fit_types:
        result = fit_results[fit_key]
        if result['beta_err'] > 0:
            beta_str = f"{result['beta']:.3f} ± {result['beta_err']:.3f}"
        else:
            beta_str = f"{result['beta']:.3f} (fixed)"
        
        print(f"{fit_label:<20} "
              f"{result['tau0']:>6.2f} ± {result['tau0_err']:<6.2f}      "
              f"{beta_str:<25} "
              f"{result['chi2']/result['dof']:>6.2f}")
    
    print(f"{'='*80}\n")
    
    # 可选：绘制corner图展示MCMC后验分布
    for fit_key, fit_label in fit_types:
        result = fit_results[fit_key]
        samples = result['samples']
        
        if fit_key == 'beta_free':
            labels = ['τ₀', 'β']
            fig = corner.corner(samples, labels=labels, 
                              quantiles=[0.16, 0.5, 0.84],
                              show_titles=True, title_fmt='.3f')
        else:
            labels = ['τ₀']
            fig = corner.corner(samples, labels=labels, 
                              quantiles=[0.16, 0.5, 0.84],
                              show_titles=True, title_fmt='.3f')
        
        fig.suptitle(f'{obj}: {fit_label} - MCMC Posterior', 
                    fontsize=13, fontweight='bold')
        plt.savefig(f'fig/timeseries/{obj}_{fit_key}_corner.png', dpi=200, bbox_inches='tight')

    tau_corr = tau_values + result['tau0']
    tau_corr_err = tau_errs + result['tau0_err']
    # 储存结果
    d1 = {'wave':wavelengths,
         'tau':tau_values,
         'tau_err':tau_errs,
         'tau0':result['tau0'],
         'tau0_err':result['tau0_err'],
         'tau_corr':tau_corr,
         'tau_corr_err':tau_corr_err
    }

    df = pd.DataFrame(data=d1)
    df.to_csv(obj_name + '_timeseries.csv', index=False)
print("\n" + "="*80)
print("All analysis completed!")
print("="*80)

