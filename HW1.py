#%%
import numpy as np
import matplotlib.pyplot as plt

def calc_WeD(tD,reD):
    if tD < 0.01:
        WeD = 2/np.sqrt(np.pi) * np.sqrt(tD)
        return WeD
    elif tD < 200 and tD > 0.01 and tD < 0.25 * reD ** 2:
        WeD_num = 1.12838*np.sqrt(tD) + 1.19328*tD + 0.269872*tD*np.sqrt(tD) + 0.00855294 * tD**2
        WeD_denom = 1 + 0.616599*np.sqrt(tD) + 0.0413008*tD
        WeD = WeD_num/WeD_denom        
        return WeD
    elif tD > 200 and tD < 0.25 * reD**2:
        WeD = (2.02566*tD - 4.29881)/ np.log(tD)
        return WeD
    elif tD > 0.25 * reD ** 2:
        WeD = (reD**2 - 1)/2 * (1 - np.exp(-2 * tD / ((reD**2 - 1)*(np.log(reD) - 0.75))))
        return WeD
    else:
        print('Whoops, there was an error')
        return None
    
def Vectorized_calc_WeD(tD,reD):
    tD = np.asarray(tD)
    WeD = np.zeros_like(tD, dtype=float)
    
    cond1 = tD < 0.01
    WeD[cond1] = 2 / np.sqrt(np.pi) * np.sqrt(tD[cond1])
    
    cond2 = (0.01 <= tD) & (tD < 200) & (tD < 0.25 * reD ** 2)
    WeD_num = (1.12838 * np.sqrt(tD) + 1.19328 * tD +
               0.269872 * tD * np.sqrt(tD) + 0.00855294 * tD**2)
    WeD_denom = 1 + 0.616599 * np.sqrt(tD) + 0.0413008 * tD
    WeD[cond2] = WeD_num[cond2] / WeD_denom[cond2]
    
    cond3 = (200 <= tD) & (tD < 0.25 * reD ** 2)
    WeD[cond3] = (2.02566 * tD[cond3] - 4.29881) / np.log(tD[cond3])
    
    cond4 = tD >= 0.25 * reD ** 2
    WeD[cond4] = ((reD**2 - 1) / 2) * (1 - np.exp(-2 * tD[cond4] / ((reD**2 - 1) * (np.log(reD) - 0.75))))
    
    return WeD
    
#%%
dimensionless_time = np.linspace(0.01,100,300)
plot1_radius = np.arange(2.5,5.1,0.5)
plot2_radius = np.arange(10,16.1,2)

# %%
fig, axes = plt.subplots(1, 2, figsize=(12, 5),layout="constrained")

# Plot for smaller radii
for r in plot1_radius:
    WeD_values = Vectorized_calc_WeD(dimensionless_time, r)
    axes[0].plot(dimensionless_time, WeD_values, label=f'R = {r}')

axes[0].set_xlabel('Dimensionless Time')
axes[0].set_xscale('log')
axes[0].set_ylabel('WeD')
axes[0].set_title('Plot A: Smaller Radii')
axes[0].legend()
axes[0].grid(True)

# Plot for larger radii
for r in plot2_radius:
    WeD_values = Vectorized_calc_WeD(dimensionless_time, r)
    axes[1].plot(dimensionless_time, WeD_values, label=f'R = {r}')

axes[1].set_xlabel('Dimensionless Time')
axes[1].set_ylabel('WeD')
axes[1].set_xscale('log')

axes[1].set_title('Plot B: Larger Radii')
axes[1].legend()
axes[1].grid(True)

# Show the plots
plt.show()
