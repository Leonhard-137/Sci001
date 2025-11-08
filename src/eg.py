# Enable inline plotting in notebook
%matplotlib inline
# Populate namespace with numerical python function library and matplotlib plotting library.
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec#
from scipy import interpolate
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# ---- 全局绘图风格（更稳的写法）----
plt.rcParams.update({
    "font.family": "sans-serif",           # 用 sans-serif
    "font.sans-serif": ["DejaVu Sans"],    # 指定 DejaVu Sans
    "figure.figsize": [40, 15],            # 你原来的大图尺寸
    "font.size": 40
})

# ---- 数据与路径 ----
filters = ["1", "2", "3"]
datadir = Path("MockData") / "HighSN"      # 相对路径，不要以 '/' 开头
data = []

# （可选）调试：看看当前工作目录和是否找到文件夹
# print("CWD =", Path.cwd())
# print("数据目录存在？", datadir.exists())
# print("示例文件：", list(datadir.glob("TestObj_*.dat"))[:5])

for i, f in enumerate(filters):
    file = datadir / f"TestObj_{f}.dat"
    if not file.exists():
        print(f"[WARN] 找不到文件：{file.resolve()}")
        continue

    arr = np.loadtxt(file)
    data.append(arr)

    mjd  = arr[:, 0]
    flux = arr[:, 1]
    err  = arr[:, 2]

    plt.figure(i)
    plt.errorbar(
        mjd, flux,
        yerr=err,
        ls="none",
        marker=".",
        ms=20,
        elinewidth=5
    )
    plt.ylabel("Flux")
    plt.xlabel("Time")
    plt.title(f"TestObj_{f}")
    plt.tight_layout()
    # 如果需要保存图片，取消下一行注释：
    # plt.savefig(datadir / f"TestObj_{f}.png", dpi=150)

import PyROA
datadir = "MockData/HighSN/"   # 相对路径
objName="TestObj"
filters=["1","2","3"]
init_tau = [5.0, 10.0]
priors = [[0.5, 2.0],[0.5, 2.0], [0.0, 20.0], [0.05, 5.0], [0.0, 10.0]]

fit = PyROA.Fit(datadir, objName, filters, priors, add_var=True, init_tau = init_tau, Nsamples=10000, Nburnin=5000)

plt.rcParams.update({
    "font.family": "Sans",  
    "font.serif": ["DejaVu"],
"figure.figsize":[40,30],
"font.size": 30})  


import pickle
file = open("samples_flat.obj",'rb')
samples_flat = pickle.load(file)
file = open("Lightcurve_models.obj",'rb')
models = pickle.load(file)


#Split samples into chunks, 4 per lightcurve i.e A, B, tau, sig
chunk_size=4
transpose_samples = np.transpose(samples_flat)
#Insert zero where tau_0 would be 
transpose_samples = np.insert(transpose_samples, [2], np.array([0.0]*len(transpose_samples[1])), axis=0)
samples_chunks = [transpose_samples[i:i + chunk_size] for i in range(0, len(transpose_samples), chunk_size)] 



fig = plt.figure(5)
gs = fig.add_gridspec(len(filters), 2, hspace=0, wspace=0, width_ratios=[5, 1])
axs= gs.subplots(sharex='col')

band_colors=["royalblue", "darkcyan", "olivedrab", "#ff6f00", "#ef0000", "#610000"]

#Loop over lightcurves
filters=["1","2","3"]
datadir = "MockData/HighSN/"
data=[]
for i in range(len(filters)):
    #Read in data
    file = datadir + "TestObj_" + str(filters[i]) + ".dat"
    data.append(np.loadtxt(file))
    mjd = data[i][:,0]
    flux = data[i][:,1]
    err = data[i][:,2]    
    
    #Add extra variance
    sig = np.percentile(samples_chunks[i][-1], 50)
    err = np.sqrt(err**2 + sig**2)
    
    #Plot Data
    axs[i][0].errorbar(mjd, flux , yerr=err, ls='none', marker=".", color=band_colors[i], ms=20, elinewidth=5)
    #Plot Model
    t, m, errs = models[i]
    axs[i][0].plot(t,m, color="black", lw=3)
    axs[i][0].fill_between(t, m+errs, m-errs, alpha=0.5, color="black")
    axs[i][0].set_xlabel("Time")
    axs[i][0].set_ylabel("Flux")

    #Plot Time delay posterior distributions
    tau_samples = samples_chunks[i][2],
    axs[i][1].hist(tau_samples, color=band_colors[i], bins=50)
    axs[i][1].axvline(x = np.percentile(tau_samples, [16, 50, 84])[1], color="black")
    axs[i][1].axvline(x = np.percentile(tau_samples, [16, 50, 84])[0] , color="black", ls="--")
    axs[i][1].axvline(x = np.percentile(tau_samples, [16, 50, 84])[2], color="black",ls="--")
    axs[i][1].axvline(x = 0, color="black",ls="--")    
    axs[i][1].set_xlabel("Time Delay ")
    axs[0][0].set_title("Lightcurves")
    axs[0][1].set_title("Time Delay")
for ax in axs.flat:
    ax.label_outer()    

#Driving Lightcurve
plt.figure(6)
plt.rcParams.update({
    "figure.figsize":[40,15],
"font.size": 40}) 
plt.plot(fit.t, fit.X, lw=3)
plt.fill_between(fit.t, fit.X-fit.X_errs, fit.X+fit.X_errs, alpha=0.5)
plt.xlabel("$t$")
plt.ylabel("$X(t)$")
plt.title("Driving Lightcurve")
plt.show()
PyROA.Plot(fit)

