from re import X
import pandas as pd
import numpy as np
import astropy.units as u
from dust_extinction.parameter_averages import CCM89
from pyat.mean_rms_spectra import get_mean_rms

df = pd.read_csv('data/Mrk142.csv')

Rv = 3.1
Ebv = 0.0136
ext_model = CCM89(Rv=Rv)
z = 0.0446

WAVELENGTHS = {
    'W2': 1928, 'M2': 2246, 'W1': 2600, 'U': 3465,
    'B': 4392, 'SV': 5468, 'lV': 5383, 'u': 3543, 'g': 4770,
    'r': 6215, 'i': 7545, 'z': 8700
}

# 计算 A_lambda (mag)
A_V = Rv * Ebv
A_lambda_dict = {}

for flt, lam in WAVELENGTHS.items():
    lam_val = lam * u.AA/(1 + z)  # 计算观测波长对应的源波长
    A_over_Av = ext_model(lam_val)  # A_lambda / A_V
    A_lambda = A_over_Av * A_V      # 实际 A_lambda
    A_lambda_dict[flt] = A_lambda  # 存数值

df['A_lambda'] = df['Filter'].map(A_lambda_dict)
df['Flux_ext_rest'] = df['Flux'] * (10 ** (0.4 * df['A_lambda']))

rms_mean = []

for flt, group in df.groupby("Filter"):
    prof = group["Flux_ext_rest"].to_numpy()
    err  = group["Error"].to_numpy()

    # 计算加权平均和 RMS
    mean, rms = get_mean_rms(prof, err, axis=0, weight="error")

    rms_mean.append({
        "Filter": flt,
        "Mean_Flux_ext_rest": mean,
        "RMS_Flux_ext_rest": rms
    })

rms_mean = pd.DataFrame(rms_mean).set_index("Filter")

X_lambda_t = {}

for flt in WAVELENGTHS:
    flt_data = df.loc[df['Filter'] == flt, 'Flux_ext_rest']
    rms_value = rms_mean.loc[flt, 'RMS_Flux_ext_rest']  # 通过索引访问
    X_lambda_t_temp = (flt_data - flt_data.mean()) / rms_value
    X_lambda_t[flt] = X_lambda_t_temp

print(X_lambda_t)
