import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# Read original data
df = pd.read_csv('data/Mrk142.csv')

# PyROA result parameters (example values, read from PyROA output files in practice)
drive_t = np.array([58000, 58050, 58100, 58150, 58200])  # Drive light curve times
drive_x = np.array([10.0, 12.5, 9.8, 11.2, 10.5])       # Drive light curve fluxes

# Parameters for each band: A(scale factor), B(offset), tau(time delay)
params = {
    'W2': {'A': 1.0, 'B': 5.0, 'tau': 0},
    'M2': {'A': 1.1, 'B': 4.8, 'tau': 2},
    'W1': {'A': 1.2, 'B': 4.6, 'tau': 4}
}

# Create interpolation function for drive light curve
drive_interp = interp1d(drive_t, drive_x, kind='linear', fill_value='extrapolate')

# Reconstruct and plot
plt.figure(figsize=(12, 8))

for flt, param in params.items():
    # Get original data for this band
    flt_data = df[df['Filter'] == flt]
    
    if len(flt_data) > 0:
        # Reconstruction formula: F_i(t) = A_i * X(t - tau_i) + B_i
        recon_times = flt_data['MJD'] - param['tau']
        drive_values = drive_interp(recon_times)
        reconstructed_flux = param['A'] * drive_values + param['B']
        
        # Plot original data points
        plt.errorbar(flt_data['MJD'], flt_data['Flux'], 
                    yerr=flt_data['Error'], fmt='o', label=f'{flt} original', alpha=0.7)
        
        # Plot reconstructed curve
        plt.plot(flt_data['MJD'], reconstructed_flux, '-', linewidth=2, 
                label=f'{flt} reconstructed')

plt.xlabel('MJD')
plt.ylabel('Flux')
plt.legend()
plt.title('PyROA Light Curve Reconstruction')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
