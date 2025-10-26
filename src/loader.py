# import matplotlib.pyplot as plt
# import pandas as pd
# import numpy as np
# from pyat.ccf import iccf, iccf_mc
# df = pd.read_csv('data/Mrk142.csv')
# filters = df['Filter'].unique()
# objects = df['Object'].unique()
# ccf_peak_mc = pd.DataFrame(index=objects, columns=filters, dtype=object)
# tau_peak_mc = pd.DataFrame(index=objects, columns=filters, dtype=object)
# tau_cent_mc = pd.DataFrame(index=objects, columns=filters, dtype=object)
# tau_beg = -15
# tau_end = 15
# ntau = 500
# nsim = 1000
# for obj in objects:
#     # 取 W2 作为驱动
#     sub_drive = df[(df['Object'] == obj) & (df['Filter'] == 'W2')]
#     Mjd_drive = sub_drive['MJD'].to_numpy(float)
#     flux_drive = sub_drive['Flux'].to_numpy(float)
#     err_drive = sub_drive['Error'].to_numpy(float)
#     n_flt = len(filters)
#     fig, axes = plt.subplots(n_flt, 1, figsize = (8, 2.5*n_flt), sharex=True)
#     for i, flt in enumerate(filters):
#         sub = df[(df['Object'] == obj) & (df['Filter'] == flt)]
#         Mjd = sub['MJD'].to_numpy(float)
#         flux = sub['Flux'].to_numpy(float)
#         err = sub['Error'].to_numpy(float)
#         # 如果数据太少就跳过
#         if len(Mjd) < 2 or len(Mjd_drive) < 2:
#             continue
#         # 调用 iccf_mc
#         tau, ccf, ccf_arr, tau_peak, tau_cent = iccf(
#             Mjd_drive, flux_drive,
#             Mjd, flux,
#             ntau=ntau, tau_beg=tau_beg, tau_end=tau_end
#         )

#         ccf_peak_arr, tau_peak_arr, tau_cent_arr = iccf_mc(
#                     Mjd_drive, flux_drive, err_drive,
#                     Mjd, flux, err,
#                     ntau=ntau, tau_beg=tau_beg, tau_end=tau_end,
#                     nsim=nsim,  # 减少模拟次数以加快速度
#                     threshold=0.8,
#                     mode="multiple",  # 使用single模式
#                     ignore_warning=False
#                 )
#         ax = axes[i]
#         bin = np.linspace(tau_beg, tau_end, ntau+1)

#         ax.hist(
#             tau_cent_arr,
#             bin
#         )
#         # 把数组整体存进去
#         ccf_peak_mc.loc[obj, flt] = ccf_arr
#         tau_peak_mc.loc[obj, flt] = tau_peak
#         tau_cent_mc.loc[obj, flt] = tau_cent
#         print(f'{obj}-{flt}:tau_cent={tau_cent:.2f}')
#         print(f'{obj}-{flt}:tau_peak={tau_peak:.2f}')
    # plt.show()
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib.gridspec import GridSpec
from pyat.ccf import iccf, iccf_mc

# 读取数据
df = pd.read_csv('data/Mrk142.csv')
filters = df['Filter'].unique()
objects = df['Object'].unique()

# 定义颜色方案
color_drive = '#4472C4'  # 蓝色用于驱动光变曲线
color_response = '#ED7D31'  # 橙色用于响应光变曲线

tau_beg = -15
tau_end = 15
ntau = 500
nsim = 1000
for obj in objects:
    # 取 W2 作为驱动光变曲线
    sub_drive = df[(df['Object'] == obj) & (df['Filter'] == 'W2')]
    Mjd_drive = sub_drive['MJD'].to_numpy(float)
    flux_drive = sub_drive['Flux'].to_numpy(float)
    err_drive = sub_drive['Error'].to_numpy(float)
    
    if len(Mjd_drive) < 2:
        continue
    
    # 获取有效的滤光片列表
    valid_filters = []
    for flt in filters:
        sub = df[(df['Object'] == obj) & (df['Filter'] == flt)]
        if len(sub) >= 2:
            valid_filters.append(flt)
    
    n_flt = len(valid_filters)
    if n_flt == 0:
        continue
    
    # 创建图形
    fig = plt.figure(figsize=(10, 1.8*n_flt))
    
    # 使用GridSpec创建子图布局
    gs = GridSpec(n_flt, 2, figure=fig, 
                  width_ratios=[2, 1],  # 左边宽一些
                  height_ratios=[1]*n_flt,
                  hspace=0.1,  # 减小垂直间距
                  wspace=0.2)
    
    for i, flt in enumerate(valid_filters):
        sub = df[(df['Object'] == obj) & (df['Filter'] == flt)]
        Mjd = sub['MJD'].to_numpy(float)
        flux = sub['Flux'].to_numpy(float)
        err = sub['Error'].to_numpy(float)
        
        # ===== 左边：光变曲线 =====
        ax_lc = fig.add_subplot(gs[i, 0])
        
        # 根据滤光片类型决定颜色
        if 'UV' in flt or flt in ['U', 'W1', 'W2', 'M2']:
            # UV波段用蓝色
            lc_color = color_drive
        else:
            # 光学/红外波段用橙色
            lc_color = color_response
        
        # 画光变曲线
        ax_lc.errorbar(Mjd, flux, yerr=err, fmt='o', markersize=2, 
                      color=lc_color, ecolor=lc_color, alpha=0.8,
                      capsize=0, linewidth=1)
        
        # 设置标签
        ax_lc.text(0.02, 0.85, flt, transform=ax_lc.transAxes, 
                  fontsize=9, verticalalignment='top', style='italic')
        
        # 只在最下面的子图显示x轴标签
        if i == n_flt - 1:
            ax_lc.set_xlabel('Modified Julian Date', fontsize=10)
        else:
            ax_lc.set_xticklabels([])
        
        # ax_lc.set_ylabel('Flux (arbitrary units)', fontsize=9)
        ax_lc.tick_params(axis='both', labelsize=8)
        
        # ===== 右边：CCF和tau分布 =====
        ax_ccf = fig.add_subplot(gs[i, 1])
        
        # 计算CCF
        tau, ccf, ccf_peak, tau_peak, tau_cent = iccf(
            Mjd_drive, flux_drive,
            Mjd, flux,
            ntau=ntau, tau_beg=tau_beg, tau_end=tau_end
        )
        
        # 画CCF曲线
        ax_ccf.plot(tau, ccf, 'k-', linewidth=1.2)
        
        # 设置CCF的y轴范围
        ax_ccf.set_ylim(-0.2, 1.05)
        ax_ccf.set_xlim(tau_beg, tau_end)  # 类似原图的范围
        
        # 如果不是自相关，计算并显示MC结果
        if flt != 'W2' and len(Mjd) > 10:
            try:
                # MC模拟
                print(f'Running MC for {obj}-{flt}...')
                ccf_peak_arr, tau_peak_arr, tau_cent_arr = iccf_mc(
                    Mjd_drive, flux_drive, err_drive,
                    Mjd, flux, err,
                    ntau=ntau, tau_beg=tau_beg, tau_end=tau_end,
                    nsim=nsim,  # 减少模拟次数以加快速度
                    threshold=0.8,
                    mode="multiple",  # 使用single模式
                    ignore_warning=False
                )
                
                # 创建第二个y轴用于直方图
                ax_hist = ax_ccf.twinx()
                
                # 画tau_cent的分布
                bin = np.linspace(tau_beg, tau_end, ntau+1)
                
                # 分离正负延迟
                ax_hist.hist(tau_cent_arr, bins=bin, 
                                                   alpha=0.7, color=color_response, 
                                                   density=True)
                
                # 画平均值线
                mean_tau = np.mean(tau_cent_arr)
                ax_ccf.axvline(x=mean_tau, color='red', linestyle='-', 
                             linewidth=1.5, alpha=0.8)
                
                # 设置直方图y轴
                ax_hist.set_ylim(0, None)
                ax_hist.set_ylabel('')
                ax_hist.set_yticks([])
                
                print(f'{obj}-{flt}: τ_cent = {mean_tau:.2f} ± {np.std(tau_cent_arr):.2f} days')
                
            except Exception as e:
                print(f'MC failed for {obj}-{flt}: {e}')
        
        # 设置CCF轴标签
        if i == n_flt - 1:
            ax_ccf.set_xlabel('Lag (days)', fontsize=10)
        else:
            ax_ccf.set_xticklabels([])
        
        if i == 0:
            ax_ccf.set_title('Cross-Correlation Coefficient', fontsize=10)
        
        ax_ccf.set_ylabel('CCF', fontsize=9)
        ax_ccf.tick_params(axis='both', labelsize=8)
        
        # 在第一行添加额外的标签
        if i == 0:
            ax_lc.text(0.5, 1.15, obj, transform=ax_lc.transAxes, 
                      fontsize=12, weight='bold', ha='center')
    
    plt.tight_layout()
    
    # 保存图片
    plt.savefig(f'{obj}_ccf_analysis.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(f'{obj}_ccf_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

print("Analysis completed!")