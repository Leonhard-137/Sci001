from genericpath import exists
from tarfile import data_filter
import PyROA
import pandas as pd
from dust_extinction.parameter_averages import CCM89
import astropy.units as u
import os
import numpy as np
def main():
    df = pd.read_csv('data/Mrk142.csv')

    Rv = 3.1
    Ebv = 0.0136
    ext_model = CCM89(Rv = Rv)
    z = 0.0446

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

    A_v = Rv*Ebv
    A_lambda_dict = {}

    for flt, wlen in WAVELENGTHS.items():
        '''
        根据设置的Rv和E_bv，计算得到A_v，这是V波长的消光量，再根据设置的模型计算出每一个波长的A_lambda/A_v，
        由此可以得到A_lambda。
        '''
        wlen_val = wlen*u.AA/(1+z)
        Alam_over_Av = ext_model(wlen_val)
        A_lam = Alam_over_Av * A_v
        A_lambda_dict[flt] = A_lam

    df['A_lambda'] = df['Filter'].map(A_lambda_dict)
    df['Flux_ext_rest'] = df['Flux']*(10**(0.4*df['A_lambda']))

    roadir = 'pyroa_data'
    os.makedirs(roadir, exist_ok = True)

    for flt in WAVELENGTHS:
        flt_data = df[df['Filter'] == flt]
        if len(flt_data != 0):
            output_data = flt_data[['MJD', 'Flux_ext_rest', 'Error']].to_numpy()
            np.savetxt(f"{roadir}/Mrk142_{flt}.dat", output_data)

    # Pyroa
    filters = list(WAVELENGTHS.keys())
    priors = [
        [0.1, 2], # RMS尺度先验
        [0.1, 2], # 平均值先验
        [-25, 25], # 时间延迟先验
        [0.01, 10.0], # delta 参数范围
        [0, 5] # 额外sigma范围
    ]

    print("PyROA Started!...")
    # 构建拟合模型
    fit = PyROA.Fit(
        roadir, "Mrk142", filters, priors,
        add_var=True,
        Nsamples=1000,
        Nburnin=500
    )

if __name__ == '__main__':
    # # 设置环境变量，防止多进程问题
    # os.environ["MKL_NUM_THREADS"] = "1"
    # os.environ["NUMEXPR_NUM_THREADS"] = "1" 
    # os.environ["OMP_NUM_THREADS"] = "1"
    
    # 运行主程序
    fit_result = main()


        

