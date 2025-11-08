## 概述

PyROA 是一个用于建模类星体光变曲线的工具（[Donnan et al. 2021](https://ui.adsabs.harvard.edu/abs/2021arXiv210712318D/abstract)），其中变异性使用运行最优平均（ROA）描述，参数使用马尔可夫链蒙特卡洛（MCMC）技术采样 - 特别是使用 emcee。采用贝叶斯方法，可以在采样参数上使用先验。它还可以对不同望远镜的相同滤波器光变曲线进行相互校准，并相应调整其不确定性（[Donnan et al. 2023](https://ui.adsabs.harvard.edu/abs/2023arXiv230209370D/abstract)）。

目前主要有三个用途：

1. 确定不同波长光变曲线之间的时间延迟
    
2. 对多个望远镜的光变曲线进行相互校准，将它们合并为单一光变曲线
    
3. 确定引力透镜类星体图像之间的时间延迟，同时建模微引力透镜效应
    

PyROA 还包含一个噪声模型，其中每个光变曲线都有一个参数，用于向流量测量添加额外方差，以考虑低估的误差。如果需要可以关闭此功能。

该代码易于使用，提供了示例 Jupyter notebook，演示了三种主要用途。代码通过指定一个目录来运行，该目录包含每个光变曲线的 .dat 文件，列依次为时间、流量、流量误差。还需要指定先验，特别是每个参数的均匀先验范围。

如有疑问，请发送邮件至：fergus.donnan@physics.ox.ac.uk

## 安装

使用 pip 安装：`pip install PyROA`

## 版本更新

### v3.2.2

- 修复了初始化 walker 的问题
    

### v3.2.1

- 将 A 和 B 的先验从 RMS 和均值单位改为 MAD 和中位数。这应该能更好地处理具有大量异常值的光变曲线设置先验时的情况。
    

### v3.2.0

- 当 delay_dist=True 时，有效参数数量现在仅使用驱动光变曲线。这一变化影响较小，但现在参数数量仅取决于驱动光变曲线的平滑度，而不取决于延迟分布的宽度。
    
- A 和 B（尺度和偏移）参数的先验处理方式不同。用户现在指定无量纲的下限和上限，作为每个光变曲线计算的 RMS（A）和均值（B）的因子。这允许每个光变曲线的先验范围不同，特别是在每个光变曲线流量尺度差异很大的情况下提供更合理的限制。
    
- walker 的初始值现在设置为更好地采样参数空间，现在设置为 0.2*初始值。
    
- 修复了 delay_dist=True 不允许负平均延迟的问题。
    
- 修复了最佳拟合延迟的内联输出错误。
    

### v3.1.0

- 添加了新的实用函数（Utils.py）来分析 PyROA 的输出。参见 Utils_Tutorial 了解用法示例。这些包括显示光变曲线（+残差）、收敛图、按参数的 corner 和链图、延迟谱图和流量-流量分析图。
    
- 添加了相互校准文件的更改。现在为每个数据点输出原始望远镜、该点的 ROA 模型+不确定性以及自由度值。
    

### v3.0.0

- 重新实现了延迟分布函数，现在使用数值卷积。这允许对任何类型的传递函数进行建模。
    
- 添加了 emcee 后端的使用，允许保存和恢复进度。
    

### v2.0.2

- 修复了计算慢速分量的错误。
    

### v2.0.1

- 修复了使用延迟分布时驱动光变曲线输出的错误。
    

### v2.0.0

- 为 Fit() 添加了选项，指定哪个滤波器是测量延迟的参考。将滤波器名称指定给参数 delay_ref。
    
- 将延迟分布更改为截断高斯分布，防止延迟小于模糊参考光变曲线延迟的延迟分布贡献。
    
- 如果包含慢速分量，代码现在的运行速度应与不包含该分量时相同。
    

### v1.0.3

PyROA 初始版本

## 使用方法

### 案例 1：测量不同波长光变曲线之间的延迟

要测量不同波长光变曲线之间的时间延迟，我们首先指定一个目录、对象名称和滤波器。在该目录中，每个光变曲线是一个 .dat 文件，包含三列：时间、流量、流量误差，命名为："objName_filter.dat"。使用提供的模拟数据，我们将运行代码：

```
import PyROA

datadir = "/MockData/HighSN/"
objName = "TestObj"
filters = ["1", "2", "3"]

priors = [[0.5, 2.0], [0.5, 2.0], [-50.0, 50.0], [0.01, 10.0], [0.0, 10.0]]

fit = PyROA.Fit(datadir, objName, filters, priors, add_var=True)
```

先验是均匀的，其中限制按以下方式指定：

`priors = [[A_lower, A_upper], [B_lower, B_upper], [tau_lower, tau_upper], [delta_lower, delta_upper], [sig_lower, sig_upper]]`

在上面这些限制范围较大，但以下是它们的含义简要说明：

- 第一个参数是 A，即每个光变曲线的 RMS（对于三个光变曲线有 A1、A2、A3）。用户指定的限制是每个原始光变曲线计算的 RMS 的分数。这允许每个光变曲线的真实限制不同，这在光变曲线之间的尺度差异很大时尤其重要。限制 [0.5, 2.0] 应该适用于所有使用情况。
    
- 下一个参数 B 表示每个光变曲线的均值（对于三个光变曲线有 B1、B2、B3）。这类似地指定为每个光变曲线计算的均值的分数。限制 [0.5, 2.0] 应该适用于所有使用情况。
    
- 接下来，tau 是光变曲线之间的时间延迟（这里只有 tau2、tau3），因此这个先验范围给出了模型探索的延迟范围。
    
- 下一个参数 delta 给出了窗口函数的宽度，必须为正且非零。如果您的概率返回 nan，可能是因为此先验的下限太小。
    
- 最后一个参数是额外误差参数，同样为正。
    

每个光变曲线有 4 个参数：A、B、tau、sig，其中第一个光变曲线的 tau=0。

Delta 控制运行最优平均的灵活性，这是使用所有光变曲线计算的。

使用 Fit 时可以指定更多选项。完整说明如下：

**class Fit(datadir, objName, filters, priors, init_tau=None, init_delta=1.0, delay_dist=False, add_var=True, sig_level=4.0, Nsamples=10000, Nburnin=5000, include_slow_comp=False, slow_comp_delta=30.0, calc_P=False)**

**参数：**

**datadir : string :** 光变曲线目录，格式为 "objName_filter.dat"

**objName : string :** 对象名称，用于查找光变曲线 .dat 文件

**filters : array :** 滤波器名称列表

**priors : array :** 指定参数均匀先验限制的数组。具体格式如上所述。

**init_tau : array :** 初始时间延迟列表。这可以帮助减少预烧期或找到正确解（如果延迟很大且光变曲线重叠很少）。

**init_delta : float :** delta 的初始值。

**delay_dist : bool :** 是否对每个时间延迟包含延迟分布，根据延迟分布的宽度模糊光变曲线。如果设置为 True，每个延迟参数现在表示截断高斯延迟分布的峰值，新参数 tau_rms 表示宽度。这是相对于第一个光变曲线测量的模糊，其中分布的截断点是该光变曲线的延迟，防止小于该波段的延迟贡献到延迟分布。此选项确实需要更长的运行时间，尤其是在较大的数据集上。

**add_var : bool :** 是否包含为每个光变曲线的流量误差添加额外方差的参数。

**sig_level : float :** sigma 剪切的阈值（以 sigma 为单位）。

**Nsamples : int :** 拟合过程的 MCMC 样本数（每个 walker）。此值包括预烧期。

**Nburnin : int :** 要作为预烧期移除的 Nsamples 数量。

**include_slow_comp : bool :** 是否在模型中包含慢变分量，由具有固定宽窗口函数的 ROA 表示，由 slow_comp_delta 指定。

**slow_comp_delta : int :** 如果包含慢分量，则为其窗口函数的宽度。

**calc_P : bool :** 预计算 ROA 参数数量作为 delta 函数的选项，随后插值用于拟合例程。此选项对于大型数据集可能显著增加运行时间。警告：这是近似的，因为它不考虑当前时间延迟或额外方差参数。建议仅在延迟较小且 add_var = False 时使用。

**delay_ref : string :** 滤波器名称，与 filters 数组匹配，延迟将相对于该滤波器测量。默认设置为提供的第一个滤波器。

### 案例 2：多望远镜光变曲线的相互校准

当使用来自多个望远镜的数据时，例如 Las Cumbres Observatory，这些数据可以合并为单个光变曲线，其中运行最优平均提供合并光变曲线的模型。

与之前类似，通过指定包含每个光变曲线为 .dat 文件的目录来运行，该文件包含三列：时间、流量、流量误差。文件必须命名为："objName_filter_scope.dat"。因此，在指定目录后运行代码，提供 objName、合并光变曲线的滤波器以及要合并的望远镜列表：

```
import PyROA

datadir = "/F9_lightcurves/"
objName = "F9"
filter = "B"

# 望远镜名称列表
scopes = ["1m003", "1m004", "1m005", "1m009", "1m010", "1m011", "1m012", "1m013"]
# 先验
priors = [[0.01, 10.0], [0.0, 2.0]]

fit = PyROA.InterCalibrate(datadir, objName, filter, scopes, priors)
```

这里的先验仅针对 delta 和额外误差参数，形式为：`priors = [[delta_lower, delta_upper], [sig_lower, sig_upper]]`。

这将合并的光变曲线输出为 .dat 文件到指定的目录。还会从运行函数的位置创建 corner 图。

InterCalibrate 函数的完整选项列表是：

**class InterCalibrate(datadir, objName, filter, scopes, priors, init_delta=1.0, sig_level=4.0, Nsamples=15000, Nburnin=10000)**

**参数：**

**datadir : string :**光变曲线目录，格式为 "objName_filter_scope.dat"

**objName : string :**对象名称，用于查找光变曲线 .dat 文件

**filter : string :**合并光变曲线的滤波器名称。

**scopes : array :**望远镜名称列表。

**priors : array :**指定参数均匀先验限制的数组。具体格式如上所述。

**init_delta : float :**delta 的初始值。

**sig_level : float :**sigma 剪切的阈值（以 sigma 为单位）。

**Nsamples : int :**拟合过程的 MCMC 样本数（每个 walker）。此值包括预烧期。

**Nburnin : int :**要作为预烧期移除的 Nsamples 数量。

### 案例 3：测量引力透镜类星体图像之间的时间延迟

为了测量引力透镜类星体图像光变曲线之间的时间延迟，我们使用函数 GravLensFit。这与之前类似运行，指定一个包含每个光变曲线 .dat 文件的目录，其中三列为：时间、星等、星等误差。这里亮度以星等表示，函数进行转换：flux = 3.0128e-5 * 10^(-0.4*m)。这转换为任意流量单位，因此可以根据数据更改此因子。

这里我们指定图像而不是滤波器：

```
import PyROA

datadir = "/PG 1115+080/"
objName = "PG 1115+080"
images = ["A", "B", "C"]

priors = [[0.0, 5.0], [0.0, 50.0], [-400.0, 400.0], [2.5, 150.0], [0.0, 2.0], [-50.0, 50.0]]

fit = PyROA.GravLensFit(datadir, objName, images, priors, init_delta=10.0, Nsamples=20000, Nburnin=15000)
```

这里的先验是针对：

`priors = [[A1_lower, A1_upper], [B1_lower, B1_upper], [tau_lower, tau_upper], [delta_lower, delta_upper], [sig_lower, sig_upper], [P_lower, P_upper]]`，其中 A1 和 B1 是第一个光变曲线在任意流量单位中的 RMS 和均值，tau 是图像之间的时间延迟，delta 是 ROA 窗口宽度，sig 是额外方差参数，P 是所有微引力透镜多项式系数的先验范围。

**class GravLensFit(datadir, objName, images, priors, init_tau=None, init_delta=10.0, add_var=True, sig_level=4.0, Nsamples=10000, Nburnin=5000, flux_convert_factor=3.0128e-5)**

**参数：**

**datadir : string :**光变曲线目录，格式为 "objName_image.dat"

**objName : string :**对象名称，用于查找光变曲线 .dat 文件

**images : array :**图像列表。

**priors : array :**指定参数均匀先验限制的数组。具体格式如上所述。

**init_tau : array :**初始时间延迟列表。这可以帮助减少预烧期或找到正确解（如果延迟很大且光变曲线重叠很少）。

**init_delta : float :**delta 的初始值。

**add_var : bool :**是否包含为每个光变曲线的流量误差添加额外方差的参数。

**sig_level : float :**sigma 剪切的阈值（以 sigma 为单位）。

**Nsamples : int :**拟合过程的 MCMC 样本数（每个 walker）。此值包括预烧期。

**Nburnin : int :**要作为预烧期移除的 Nsamples 数量。

**flux_convert_factor : float :**星等转换为流量时使用的因子，其中 flux = flux_convert_factor * 10^(-0.4*m)。

## 引用

如果您使用此代码，请引用（[Donnan et al. 2021](https://ui.adsabs.harvard.edu/abs/2021arXiv210712318D/abstract)）：

```
@ARTICLE{2021MNRAS.508.5449D,
       author = {{Donnan}, Fergus R. and {Horne}, Keith and {Hern{\'a}ndez Santisteban}, Juan V.},
        title = "{Bayesian analysis of quasar light curves with a running optimal average: new time delay measurements of COSMOGRAIL gravitationally lensed quasars}",
      journal = {\mnras},
     keywords = {gravitational lensing: strong, methods: data analysis, quasars: general, Astrophysics - Instrumentation and Methods for Astrophysics, Astrophysics - Astrophysics of Galaxies},
         year = 2021,
        month = dec,
       volume = {508},
       number = {4},
        pages = {5449-5467},
          doi = {10.1093/mnras/stab2832},
archivePrefix = {arXiv},
       eprint = {2107.12318},
 primaryClass = {astro-ph.IM},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2021MNRAS.508.5449D},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}
```

相互校准程序见 [Donnan et al. 2023](https://ui.adsabs.harvard.edu/abs/2023arXiv230209370D/abstract)：

```
@ARTICLE{2023arXiv230209370D,
       author = {{Donnan}, Fergus R. and {Hern{\'a}ndez Santisteban}, Juan V. and {Horne}, Keith and {Hu}, Chen and {Du}, Pu and {Li}, Yan-Rong and {Xiao}, Ming and {Ho}, Luis C. and {Aceituno}, Jes{\'u}s and {Wang}, Jian-Min and {Guo}, Wei-Jian and {Yang}, Sen and {Jiang}, Bo-Wei and {Yao}, Zhu-Heng},
        title = "{Testing Super-Eddington Accretion onto a Supermassive Black Hole: Reverberation Mapping of PG 1119+120}",
      journal = {arXiv e-prints},
     keywords = {Astrophysics - Astrophysics of Galaxies, Astrophysics - High Energy Astrophysical Phenomena},
         year = 2023,
        month = feb,
          eid = {arXiv:2302.09370},
        pages = {arXiv:2302.09370},
          doi = {10.48550/arXiv.2302.09370},
archivePrefix = {arXiv},
       eprint = {2302.09370},
 primaryClass = {astro-ph.GA},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2023arXiv230209370D},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}
```