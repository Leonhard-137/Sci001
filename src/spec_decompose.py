import pandas as pd
from dust_extinction.parameter_averages import CCM89
import astropy.units as u
import os
import numpy as np
import PyROA
from pathlib import Path
import matplotlib.pyplot as plt

def main():
    # 设置所有输出路径
    base_dir = Path("pyroa_analysis_results")
    data_dir = base_dir / "input_data"
    output_dir = base_dir / "pyroa_output"
    plots_dir = base_dir / "plots"
    
    # 创建所有必要的目录
    for directory in [base_dir, data_dir, output_dir, plots_dir]:
        directory.mkdir(exist_ok=True)
    
    print(f"工作目录: {base_dir.absolute()}")
    print(f"输入数据目录: {data_dir}")
    print(f"PyROA输出目录: {output_dir}")
    print(f"图表输出目录: {plots_dir}")
    
    # 读取数据
    df = pd.read_csv('data/Mrk142.csv')

    # 消光参数设置
    Rv = 3.1
    Ebv = 0.0136
    ext_model = CCM89(Rv=Rv)
    z = 0.0446  # 红移

    # 滤波器波长定义（单位：埃）
    WAVELENGTHS = {
        'W2': 1928, 
        'M2': 2246, 
        'W1': 2600, 
        'U': 3465,
        'B': 4392, 
        'SV': 5468, 
        'u': 3543, 
        'g': 4770,
        'r': 6215, 
        'i': 7545, 
        'z': 8700
    }

    # 计算各波长的消光修正
    A_v = Rv * Ebv
    A_lambda_dict = {}

    for flt, wlen in WAVELENGTHS.items():
        # 根据红移修正波长，计算消光量
        wlen_val = wlen * u.AA / (1 + z)
        Alam_over_Av = ext_model(wlen_val)
        A_lam = Alam_over_Av * A_v
        A_lambda_dict[flt] = A_lam

    # 应用消光修正
    df['A_lambda'] = df['Filter'].map(A_lambda_dict)
    df['Flux_ext_rest'] = df['Flux'] * (10 ** (0.4 * df['A_lambda']))

    # 为每个滤波器保存数据文件到指定目录
    for flt in WAVELENGTHS:
        flt_data = df[df['Filter'] == flt]
        if len(flt_data) != 0:
            output_data = flt_data[['MJD', 'Flux_ext_rest', 'Error']].to_numpy()
            np.savetxt(data_dir / f"Mrk142_{flt}.dat", output_data)
            print(f"已保存: {data_dir / f'Mrk142_{flt}.dat'}")

    # PyROA 参数设置
    filters = list(WAVELENGTHS.keys())
    priors = [
        [0.1, 2],    # RMS尺度先验
        [0.1, 2],    # 平均值先验
        [-25, 25],   # 时间延迟先验
        [0.01, 10.0], # delta 参数范围
        [0, 5]       # 额外sigma范围
    ]

    print("\nPyROA 开始拟合...")
    
    # 构建PyROA拟合模型，明确指定输出目录
    fit = PyROA.Fit(
        str(data_dir),      # 输入数据目录
        "Mrk142", 
        filters, 
        priors,
        add_var=True,
        Nsamples=1000,
        Nburnin=500
        # output_dir=str(output_dir)  # 明确指定输出目录
    )
    
    # 运行MCMC采样
    print("运行MCMC采样...")
    fit.run_mcmc()
    
    # 显示结果摘要
    print("\n拟合结果摘要:")
    summary = fit.summary()
    print(summary)
    
    # 保存摘要到文件
    summary_file = output_dir / "fitting_summary.txt"
    with open(summary_file, 'w') as f:
        f.write(str(summary))
    print(f"结果摘要已保存: {summary_file}")
    
    # 生成并保存诊断图
    try:
        # 链图
        chain_plot = plots_dir / "mcmc_chains.png"
        fit.plot_chains()
        plt.savefig(chain_plot, dpi=300, bbox_inches='tight')
        print(f"链图已保存: {chain_plot}")
        
        # 后验分布图
        corner_plot = plots_dir / "corner_plot.png"
        fit.plot_corner()
        plt.savefig(corner_plot, dpi=300, bbox_inches='tight')
        print(f"后验分布图已保存: {corner_plot}")
        
        # 光变曲线拟合图
        lightcurve_plot = plots_dir / "lightcurve_fit.png"
        fit.plot_lightcurves()
        plt.savefig(lightcurve_plot, dpi=300, bbox_inches='tight')
        print(f"光变曲线图已保存: {lightcurve_plot}")
        
    except Exception as e:
        print(f"生成图表时出错: {e}")
    
    # 列出所有生成的文件
    print(f"\n=== 所有生成的文件列表 ===")
    print(f"输入数据目录 ({data_dir}):")
    for file in data_dir.glob("*.dat"):
        print(f"  - {file.name}")
    
    print(f"\nPyROA输出目录 ({output_dir}):")
    for file in output_dir.glob("*"):
        print(f"  - {file.name}")
    
    print(f"\n图表目录 ({plots_dir}):")
    for file in plots_dir.glob("*.png"):
        print(f"  - {file.name}")
    
    return fit

if __name__ == '__main__':
    # # 设置环境变量，防止多进程问题
    # os.environ["MKL_NUM_THREADS"] = "1"
    # os.environ["NUMEXPR_NUM_THREADS"] = "1" 
    # os.environ["OMP_NUM_THREADS"] = "1"
    # 运行主程序
    fit_result = main()
