import yaml
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import PyROA
from PyROA.Utils import (
    FluxFlux,
    Lightcurves,
    LagSpectrum,
    Chains,
    CornerPlot,
    Convergence,
)

# ===================== 用户配置 =====================

# 1) 配置文件路径（请确保 config.yaml 存在）
CONFIG_PATH = Path("configs/Mrk142.yaml")  # 单个 AGN 配置文件
OUTPUTDIR = Path("pyroa_output")  # 输出文件夹，PyROA 结果存到这里（目前 Utils 用的是当前目录）
OUTPUTDIR.mkdir(parents=True, exist_ok=True)
DATADIR = Path("data/")

# 2) 从 YAML 读取 AGN 配置
def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    return config

# 3) 获取配置的 AGN 信息
config = load_config(CONFIG_PATH)

# =================== 从 CSV 写出 PyROA 数据 ====================

def prepare_dat_files(csv_path: Path, datadir: Path, filters: list, objname: str):
    """
    从 CSV 写出 PyROA 所需 .dat；返回：
      filters: 实际可用的滤光片列表（与 wavemap 取交集、按 wavemap 次序排序）
      wavelengths: 与 filters 对应的观测中心波长（Å）
    """
    df = pd.read_csv(csv_path)
    need_cols = {"MJD", "Filter", "Flux", "Error"}
    missing = need_cols - set(df.columns)
    if missing:
        raise ValueError(f"CSV 缺少列：{missing}")

    # 只保留 wavemap 中存在的滤光片，保证顺序一致
    filters = [f for f in filters if f in set(df["Filter"])]
    if config["delay_ref"] not in filters:
        raise ValueError(f"DELAY_REF={config['delay_ref']} 不在 CSV 的滤光片中：{filters}")

    datadir.mkdir(parents=True, exist_ok=True)
    for flt in filters:
        sub = df[df["Filter"] == flt].copy()
        sub = sub.sort_values("MJD")
        # 写三列：MJD, Flux, Error （不做任何去消光/本征化）
        np.savetxt(
            datadir / f"{objname}_{flt}.dat",
            sub[["MJD", "Flux", "Error"]].to_numpy()
        )
    
    # 返回滤光片和观测中心波长
    wavelengths = config["wavelengths"]  # 从配置文件读取波长列表
    return filters, wavelengths


# ===================== PyROA 拟合 =====================

def run_pyroa_fit(datadir: Path, objname: str, filters, priors, init_tau=None):
    """运行 PyROA 拟合，返回 fit 对象。"""
    fit = PyROA.Fit(
        str(datadir),
        objname,
        filters,
        priors,
        add_var=True,                # 为每条曲线拟合额外噪声项 σ
        delay_ref=config["delay_ref"],
        init_tau=init_tau,           # 若提供，用作初值帮助收敛
        Nsamples=config["Nsamples"], # 从配置读取采样数量
        Nburnin=config["Nburnin"],   # 从配置读取 burnin 数量
    )
    return fit


# ===================== Flux–Flux 分析 =====================

def do_fluxflux(objname: str, filters, delay_ref: str, gal_ref: str, band_colors,
                wavelengths, ebv: float, redshift: float, burnin: int):
    """
    调用 Utils.FluxFlux：生成 <figname>_fluxflux.pdf / _SED.pdf 和 CSV。
    注意：这里统一做【银河去消光 + 本征波长】（若 CSV_ALREADY_DEREDDENED 则不做去消光）。
    """
    ebv_for_fluxflux = ebv  # 如果 CSV 已经去消光，这里可以改成 0.

    FluxFlux(
        objName=objname,
        filters=filters,
        delay_ref=delay_ref,
        gal_ref=gal_ref,
        wavelengths=wavelengths,               # 观测波长；函数内会用 redshift 作本征化
        input_units='flam',
        output_units='flam',
        burnin=burnin,
        band_colors=band_colors,
        redshift=redshift,
        ebv=ebv_for_fluxflux,
        figname=f"{objname}",
    )


# ===================== 主程序 =====================

def main():
    print("== 1) 写 PyROA 输入 .dat ==")
    # 注意：这里 dat 文件写在当前目录 ./，和 Utils 默认 datadir='./' 一致
    filters, wavelengths = prepare_dat_files(
        Path(config["csv"]),
        Path('./'),
        config["filters"],
        config["obj_name"]
    )
    print("filters:", filters)
    print("wavelengths (Å):", wavelengths)

    # 统一配一套颜色，在后面所有 utils 的绘图里用
    band_colors = [
        'royalblue', 'darkcyan', 'olivedrab', '#ff6f00', '#ef0000', '#610000',
        'brown', 'gold', 'purple', 'green'
    ]
    if len(band_colors) < len(filters):
        band_colors = (band_colors * (len(filters) // len(band_colors) + 1))[:len(filters)]

    print("\n== 2) 运行 PyROA 拟合 ==")
    # # 直接把 ICCF_TAU 作为 init_tau
    # init_tau = config.get("ICCF_TAU", None)
    # fit = run_pyroa_fit(DATADIR, config["obj_name"], filters, config["priors"], init_tau=init_tau)

    # ================== 使用 Utils 里的所有可用绘图 ==================

    print("\n== 2a) Lightcurves：光变 + 残差 + lag 后验 ==")
    Lightcurves(
        objName=config["obj_name"],
        filters=filters,
        delay_ref=config["delay_ref"],
        datadir=str(DATADIR) + '/',          # .dat 写在当前目录
        outputdir=str(Path('./')),        # PyROA 默认也在当前目录输出 .obj
        burnin=config["Nburnin"],
        band_colors=band_colors,
        grid=True,
        grid_step=5.0,
        show_delay_ref=True,              # 把参考波段也画出来
        figname=f"{config['obj_name']}_lightcurves",
    )

    print("== 2b) LagSpectrum：时滞谱 ==")
    LagSpectrum(
        filters=filters,
        delay_ref=config["delay_ref"],
        wavelengths=wavelengths,
        burnin=config["Nburnin"],
        samples_file='samples_flat.obj',
        outputdir=str(Path('./')),
        band_colors=band_colors,
        redshift=config["redshift"],
        figname=f"{config['obj_name']}_lagspectrum",
    )

    print("== 2c) Chains：MCMC 链图（以 tau 为例） ==")
    Chains(
        nparam='tau',                      # 只画 tau 参数在各个滤光片上的链
        filters=filters,
        delay_ref=config["delay_ref"],
        burnin=config["Nburnin"],
        samples_file='samples_flat.obj',
        outputdir=str(Path('./')),
        figname=f"{config['obj_name']}_chains_tau.pdf",
    )

    print("== 2d) CornerPlot：后验角图（以 tau 为例） ==")
    CornerPlot(
        nparam='tau',                      # 只画 tau 的 corner
        filters=filters,
        delay_ref=config["delay_ref"],
        burnin=config["Nburnin"],
        samples_file='samples_flat.obj',
        outputdir=str(Path('./')),
        figname=f"{config['obj_name']}_corner_tau.pdf",
    )

    print("== 2e) Convergence：自相关收敛性诊断 ==")
    Convergence(
        outputdir=str(Path('./')),
        samples_file='samples_flat.obj',
        burnin=config["Nburnin"],
        init_chain_length=100,
        savefig=True,                      # 输出 'pyroa_convergence.pdf'
    )

    print("\n== 3) Flux–Flux & SED ==")
    do_fluxflux(
        objname=config["obj_name"],
        filters=filters,
        delay_ref=config["delay_ref"],
        gal_ref=config["gal_ref"],
        wavelengths=wavelengths,
        ebv=config["ebv"],
        band_colors=band_colors,
        redshift=config["redshift"],
        burnin=config["Nburnin"],          # 与 Fit 的 burn-in 一致
    )

    print("\n完成！")


if __name__ == "__main__":
    main()
