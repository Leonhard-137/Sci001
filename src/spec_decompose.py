import yaml
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import PyROA
from PyROA.Utils import FluxFlux  # 修正了导入路径

# ===================== 用户配置 =====================

# 1) 配置文件路径（请确保 config.yaml 存在）
CONFIG_PATH = Path("configs/Mrk142.yaml")  # 单个 AGN 配置文件
OUTPUTDIR = Path("pyroa_output")  # 输出文件夹，PyROA 结果存到这里
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
        np.savetxt(datadir / f"{objname}_{flt}.dat",
                   sub[["MJD", "Flux", "Error"]].to_numpy())
    
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
        add_var=True,              # 为每条曲线拟合额外噪声项 σ
        delay_ref=config["delay_ref"],
        init_tau=init_tau,         # 若提供，用作初值帮助收敛
        Nsamples=config["Nsamples"],  # 从配置读取采样数量
        Nburnin=config["Nburnin"],    # 从配置读取 burnin 数量
    )
    return fit


# ===================== Flux–Flux 分析 =====================

def do_fluxflux(objname: str, filters, delay_ref: str, gal_ref: str, band_colors,
                wavelengths, ebv: float, redshift: float, burnin: int):
    """
    调用 Utils.FluxFlux：生成 <figname>_fluxflux.pdf / _SED.pdf 和 CSV。
    注意：这里统一做【银河去消光 + 本征波长】（若 CSV_ALREADY_DEREDDENED 则不做去消光）。
    """
    # 如果 CSV 已经去消光，避免重复校正
    ebv_for_fluxflux = ebv

    FluxFlux(
        objName=objname,
        filters=filters,
        delay_ref=delay_ref,
        gal_ref=gal_ref,
        wavelengths=wavelengths,               # 观测波长；函数内会用 redshift 作本征化
        input_units='flam',output_units='flam',
        burnin=burnin,
        band_colors = band_colors,
        redshift=redshift,
        ebv=ebv_for_fluxflux,
        figname=f"{objname}_ff",
    )


# ===================== 主程序 =====================

def main():
    print("== 1) 写 PyROA 输入 .dat ==")
    filters, wavelengths = prepare_dat_files(Path(config["csv"]), Path('./'), config["filters"], config["obj_name"])
    print("filters:", filters)
    print("wavelengths (Å):", wavelengths)

    print("\n== 2) 运行 PyROA 拟合 ==")
    # 直接把 ICCF_TAU 作为 init_tau
    # init_tau = config.get("ICCF_TAU", None)  # 如果配置了 ICCF_TAU，就作为 init_tau 传递
    # fit = run_pyroa_fit(DATADIR, config["obj_name"], filters, config["priors"], init_tau=init_tau)
    # PyROA.Plot(fit)


    print("\n== 3) Flux–Flux & SED ==")
    band_colors = [
        'royalblue', 'darkcyan', 'olivedrab', '#ff6f00', '#ef0000', '#610000',
        'brown', 'gold', 'purple', 'green'
    ]
        
    if len(band_colors) < len(filters):
        band_colors = (band_colors * (len(filters) // len(band_colors) + 1))[:len(filters)]

    do_fluxflux(
        objname=config["obj_name"],
        filters=filters,
        delay_ref=config["delay_ref"],
        gal_ref=config["gal_ref"],
        wavelengths=wavelengths,
        ebv=config["ebv"],
        band_colors = band_colors,
        redshift=config["redshift"],
        burnin=config["Nburnin"],   # 与 Fit 的 burn-in 一致
    )

    print("\n完成！")


if __name__ == "__main__":
    main()


