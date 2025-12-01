import numpy as np
import pandas as pd
import emcee
from astropy.cosmology import FlatLambdaCDM
import yaml
import os
import matplotlib.pyplot as plt
import corner
from pathlib import Path

# 光速常数 (km/s)
c = 3e5

# 配置加载
CONFIG_PATH = "configs/Mrk142.yaml"
def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    return config

config = load_config(CONFIG_PATH)
z = config['redshift']

# 读取数据
d1 = pd.read_csv("Mrk142_fluxflux.csv")
d2 = pd.read_csv("Mrk142_timeseries.csv")

# 数据提取
tau = d2['tau_corr'].values
lam = d1['wave'].values
f_nu = d1['unred_agn_b_Jy'].values
cos_i = np.sqrt(2)/2
epsl = d1['epsilon'].values

# 计算光度距离
def luminosity_distance(tau, lam, f_nu, cos_i, epsl):
    D = tau * (lam / 1e4)**(-3/2) * (f_nu / cos_i)**(-1/2) * ((1 - epsl) / (1 - epsl**(3/2)))**2 * 6.3
    return D

D = luminosity_distance(tau, lam, f_nu, cos_i, epsl)

# 对数似然函数
def log_likelihood(theta, z, D, D_err):
    H0, f0 = theta
    cosmo = FlatLambdaCDM(H0=H0, Om0=0.3)
    DL_model = cosmo.luminosity_distance(z).value
    total_var = D_err**2 + f0**2 * (D)**2
    chi2_term = np.sum((D - DL_model)**2 / total_var)
    log_term = np.sum(np.log(total_var))
    return -0.5 * (chi2_term + log_term + len(D) * np.log(2 * np.pi))  # 修改这里

# 对数先验函数
def log_prior(theta, H_low, H_up, f0_low, f0_up):
    H0, f0 = theta
    if H_low < H0 < H_up and f0_low < f0 < f0_up:
        return 0.0
    return -np.inf

# 总的对数概率函数
def log_probability(theta, z, D, D_err, H_low, H_up, f0_low, f0_up):
    lp = log_prior(theta, H_low, H_up, f0_low, f0_up)
    return lp + log_likelihood(theta, z, D, D_err) if np.isfinite(lp) else -np.inf

# MCMC 拟合函数
def mcmc_fit(z, D, D_err, config):
    # 获取配置中的先验范围
    H_low, H_up, f0_low, f0_up = config['prior_lim']

    # 设置 MCMC 参数
    ndim = 2  # H0 和 f0 的维度
    nwalkers = config['nwalkers']
    nsteps = config['n_steps']

    # 初始化 walkers 的位置
    init_pos = np.column_stack([
        np.random.uniform(H_low, H_up, nwalkers),
        np.random.uniform(f0_low, f0_up, nwalkers)
    ])

    # 定义 MCMC 采样器
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, args=(z, D, D_err, H_low, H_up, f0_low, f0_up))
    sampler.run_mcmc(init_pos, nsteps, progress=True)

    # 丢弃 burnin 部分并且进行薄化
    burnin = config.get('burnin', 500)
    thin = config.get('thin', 15)
    samples = sampler.get_chain(discard=burnin, thin=thin, flat=True)

    # 输出拟合结果的百分位数
    H0_mcmc = np.percentile(samples[:, 0], [16, 50, 84])
    f0_mcmc = np.percentile(samples[:, 1], [16, 50, 84])

    print(f'H0 = {H0_mcmc}')
    print(f'f0 = {f0_mcmc}')
    return sampler

# 配置参数
config = {
    'prior_lim': [20, 100, 0.1, 1.0],  # H0 范围和 f0 范围
    'nwalkers': 32,  # MCMC walkers 数量
    'n_steps': 10000,  # MCMC 步数
    'burnin': 500,  # burnin 步数
    'thin': 15  # thin 步数
}

# 创建文件夹路径
fig_dir = Path('fig/Hubble')
fig_dir.mkdir(parents=True, exist_ok=True)

# 使用示例
if __name__ == "__main__":
    # 对 z, D, D_err 进行 MCMC 拟合
    D_err = np.ones_like(D) * 0.1  # 假设误差为常数
    sampler = mcmc_fit(z, D, D_err, config)

    # 绘制链图
    plt.figure(figsize=(10, 7))
    plt.plot(sampler.get_chain(flat=True))
    plt.xlabel("MCMC Steps")
    plt.ylabel("Parameter values")
    plt.title("MCMC Chain")
    plt.savefig(fig_dir / "chain_plot.png")
    plt.close()

    # 绘制收敛性图
    plt.figure(figsize=(10, 7))
    plt.plot(sampler.get_autocorr_time())
    plt.xlabel("MCMC Steps")
    plt.ylabel("Autocorrelation time")
    plt.title("Autocorrelation Time")
    plt.savefig(fig_dir / "autocorr_time.png")
    plt.close()

    # 绘制角度图 (Corner plot)
    flat_samples = sampler.get_chain(discard=config['burnin'], thin=config['thin'], flat=True)
    corner.corner(flat_samples, labels=["$H_0$", "$f_0$"], show_titles=True, title_kwargs={"fontsize": 12})
    plt.savefig(fig_dir / "corner_plot.png")
    plt.close()

    # def integrand(x):
    #     return 1 / np.sqrt(0.3 * x**3 + 0.7)
    # def cal_H_single(D_L, z):
    #     result, _ = quad(integrand, 1, 1 + z)
    #     H_0 = c*result / (D_L * (1 + z))
    #     return H_0
    # def calculate_H(z, D):
    #     for i, d in enumerate(D):
    #         print(f'{i}:{d}')
    #         H_0 = cal_H_single(d, z)
    #         print(f'{i}:H0 = {H_0}')
    # calculate_H(z, D)