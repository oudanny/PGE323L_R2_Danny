# Daniel McAllister-Ou
#%% 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


#%%
def dimensionless_time(k,t,phi,ct,re,mu):
    '''
    Calculate dimensionless time
    k : permeability in mD
    t : time in hours
    phi : porosity
    ct : total compressibility in 1/psi
    re : reservoir radius in feet
    mu : viscosity in cP 
    '''
    return 0.0002637*k*t/(phi*mu*ct*(re**2))

# # Q2.A stuff
# t = np.array([7200,5400,3600,1800])
# re = np.sqrt(3000*43560/(np.pi*(140/360)))
# dt = dimensionless_time(k=400,t=t,phi=.2,ct=7E-6,re=re,mu=1)
# print(dt)

def fetovich_inital_encroachable_water(pi,ct,ra,re,ha,phi,theta):
    '''
    Calculates initial encroachable water for Fetkovich model

    pi : initial pressure in psi
    ct : total compressibility in 1/psi
    ra : aquifer radius in feet
    re : reservoir radius in feet
    ha : aquifer height in feet
    phi : porosity
    theta : aquifer angle in degrees
    '''
    return pi*ct*(ra**2-re**2)*ha*np.pi*theta*phi/(5.614*360)



def fetkovich_productivity_index(k,ha,theta,mu,ra,re):
    '''
    Calculates productivity index (J) for Fetkovich model

    ra : aquifer radius in feet
    re : reservoir radius in feet
    ha : aquifer height in feet
    theta : aquifer angle in degrees
    mu : viscosity in cP 
    k : permeability in mD
    '''
    return 0.007082*k*ha*theta/(mu*360*(np.log(ra/re)-0.75))

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

# Q2.b stuff
re = np.sqrt(3000*43560/(np.pi*(140/360)))
ra = np.sqrt(60000*43560/(np.pi*(140/360)))
Wei = fetovich_inital_encroachable_water(pi=2700,ct=7E-6,ra=ra,re=re,phi=0.2,theta=140,ha=50)
j = fetkovich_productivity_index(ra=ra,re=re,theta=140,ha=50,k=400,mu=1)/24
U = 1.119*0.389*0.2*7E-6*50*10342.3**2

time = np.array([0,75,150,225,300,375,500,750,1000])
pressures = np.array([2700,2650,2627,2568,2540,2485,2440,2371,2316])

df_init = {'time [d]':time,
           'pressure [psi]':pressures}
#%%
df = pd.DataFrame.from_dict(df_init)
df['time [h]'] = df['time [d]']*24

#%%
# Fetkovich Method
fetkovich_df = df.copy(deep='True')
# Does Pn 
fetkovich_df['Pn'] = fetkovich_df['pressure [psi]'].shift(1).fillna(fetkovich_df['pressure [psi]']) + fetkovich_df['pressure [psi]']
fetkovich_df['Pn'] /= 2

# Calculate Δt (Time Differences)
fetkovich_df['Δt'] = fetkovich_df['time [h]'].diff().fillna(fetkovich_df['time [h]'][0])

# Initialize columns
fetkovich_df['Pa'] = np.nan
fetkovich_df['ΔWen'] = np.nan
fetkovich_df['Wen'] = np.nan
fetkovich_df.at[0, 'Pa'] = fetkovich_df.at[0, 'pressure [psi]']
fetkovich_df.at[0, 'ΔWen'] = 0
fetkovich_df.at[0, 'Wen'] = 0

cumulative_Wen = 0  # initialze cumulative sum of ΔWen

for i in range(1, len(fetkovich_df)):
    # ΔWen calculation
    fetkovich_df.at[i, 'ΔWen'] = (Wei / fetkovich_df.at[0, 'pressure [psi]']) * (fetkovich_df.at[i - 1, 'Pa'] - fetkovich_df.at[i, 'Pn']) * (1 - np.exp(-j * fetkovich_df.at[0, 'pressure [psi]'] * fetkovich_df.at[i, 'Δt'] / Wei))
    cumulative_Wen += fetkovich_df.at[i, 'ΔWen']  # Sum all previous ΔWen values
    # Pa,i calc
    fetkovich_df.at[i, 'Pa'] = fetkovich_df.at[0, 'pressure [psi]'] * (1 - (cumulative_Wen / Wei)) 
    fetkovich_df.at[i, 'Wen'] = cumulative_Wen

    # print(f"Step {i}:")
    # print(f"  Wei = {Wei}")
    # print(f"  Pi = {fetkovich_df.at[0, 'pressure [psi]']}")
    # print(f"  Pa(i-1) = {fetkovich_df.at[i - 1, 'Pa']}")
    # print(f"  Pn(i) = {fetkovich_df.at[i, 'Pn']}")
    # print(f"  Pa,n-1 - Pn,n = {fetkovich_df.at[i - 1, 'Pa'] - fetkovich_df.at[i, 'Pn']}")
    # print(f"  j = {j}")
    # print(f"  Δt = {fetkovich_df.at[i, 'Δt']}")
    # print(f"  Wen = {cumulative_Wen}") 
    # print(f"  ΔWen = {fetkovich_df.at[i, 'ΔWen']}\n") 

#%%
# VEH Method
# Initialize
red = ra/re
VEH_df = df.copy(deep='True')
Pi = VEH_df.loc[0, 'pressure [psi]']
VEH_df['Wen'] = np.nan
VEH_df.at[0,'Wen'] = 0

# Find ΔP 
VEH_df.at[0, 'ΔP'] = 0
VEH_df.at[1, 'ΔP'] = (Pi - VEH_df.loc[1, 'pressure [psi]']) / 2
VEH_df.at[2, 'ΔP'] = (Pi - VEH_df.loc[2, 'pressure [psi]']) / 2
VEH_df.loc[3:, 'ΔP'] = (VEH_df['pressure [psi]'].shift(2) - VEH_df['pressure [psi]']) / 2


for i in range(1,len(VEH_df)):
    VEH_copy = VEH_df.loc[:i].copy(deep='True')
    VEH_copy['Δt'] = VEH_copy['time [h]'].iloc[-1] - VEH_copy['time [h]'].shift(1).fillna(0)
    VEH_copy['Δtd'] = dimensionless_time(k=400,t=VEH_copy['Δt'],phi=.2,ct=7E-6,re=re,mu=1)
    VEH_copy['ΔWed'] = Vectorized_calc_WeD(VEH_copy['Δtd'],red)
    VEH_copy['ΔWe'] = VEH_copy['ΔWed'] * VEH_copy['ΔP']
    VEH_df.at[i,'Wen'] = VEH_copy['ΔWe'].sum()

    # print(VEH_copy['Δt'])
VEH_df['Wen'] = VEH_df['Wen']*U

#%%
fig, ax = plt.subplots(1, 2, figsize=(12, 5), layout='constrained')  # (1 row, 2 columns)

# First plot (Normal scale)
sns.lineplot(data=fetkovich_df, x='time [d]', y='Wen', label='Fetkovich', ax=ax[0], linestyle='dashed')
sns.lineplot(data=VEH_df, x='time [d]', y='Wen', label='VEH', ax=ax[0], linestyle='dashed')

ax[0].legend()
ax[0].set_xlabel('Time [Days]')
ax[0].set_ylabel('Water Encroached [RB]')
ax[0].set_title('Normal Scale')

# Second plot (Log scale)
sns.lineplot(data=fetkovich_df, x='time [d]', y='Wen', label='Fetkovich', ax=ax[1], linestyle='dashed')
sns.lineplot(data=VEH_df, x='time [d]', y='Wen', label='VEH', ax=ax[1], linestyle='dashed')

ax[1].legend()
ax[1].set_xlabel('Time [Days]')
ax[1].set_ylabel('Water Encroached [RB]')
ax[1].set_title('Log Scale')
ax[1].set_yscale('log')  # Corrected

# Title for the entire figure
fig.suptitle('Water Encroached [RB] vs Time [Days] for VEH and Fetkovich Models\n')

plt.show()