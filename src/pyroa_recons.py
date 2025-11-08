import os, pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

objName   = "Mrk142"
datadir   = r"E:\Phd\Program\Sci001\pyroa_input"     # 观测数据目录
outputdir = r"E:\Phd\Program\Sci001"   # 拟合输出目录（含 X_t.obj）
filters   = ["W2","M2","W1","U","B",2246,,"u","g","r","i","z"]

# 1) 读取驱动光变 X(t)
with open(os.path.join(outputdir, "X_t.obj"), "rb") as f:
    X_obj = pickle.load(f)

# 兼容两种可能结构：字典或二元组
if isinstance(X_obj, dict):
    tX = np.asarray(X_obj.get("t", X_obj.get("time")))
    X  = np.asarray(X_obj.get("X", X_obj.get("flux")))
else:
    tX, X = map(np.asarray, X_obj)

# 构造插值器（线性；超出范围可选外推或掐掉）
interp = interp1d(tX, X, kind="linear", bounds_error=False, fill_value=np.nan)

# 2) 读取各滤光片原始数据，并与 X(t_obs) 做散点
ncols = 4
nrows = int(np.ceil(len(filters)/ncols))
fig, axes = plt.subplots(nrows, ncols, figsize=(3.6*ncols, 3.2*nrows), squeeze=False)

for i, flt in enumerate(filters):
    r, c = divmod(i, ncols)
    ax = axes[r][c]
    fpath = os.path.join(datadir, f"{objName}_{flt}.dat")
    t, f, fe = np.loadtxt(fpath, unpack=True, usecols=(0,1,2))

    X_obs = interp(t)
    m = np.isfinite(X_obs) & np.isfinite(f)
    ax.errorbar(X_obs[m], f[m], yerr=fe[m], fmt='.', alpha=0.8)
    ax.set_xlabel("X(t_obs)")
    ax.set_ylabel(f"Flux ({flt})")
    ax.set_title(flt)

# 去掉空子图
for j in range(i+1, nrows*ncols):
    r, c = divmod(j, ncols)
    fig.delaxes(axes[r][c])

plt.tight_layout()
plt.show()
