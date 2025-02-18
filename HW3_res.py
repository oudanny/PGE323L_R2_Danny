# Daniel McAllister-Ou
# 2/15/2025
# HW 3
#%%
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt

#%%
# $k_{rw} = k_{rw}^0 S^{n_w}$
# $k_{ro} = k_{ro}^0 (1-S)^{n_o}$
# $S = \frac{S_w - S_{wr}}{1 - S_{or} - S_{wr}}$

def corey_water_relperm(krwo,S,nw):
    return krwo*S**nw
def corey_oil_relperm(kroo,S,no):
    return kroo*(1-S)**no
def saturation(sw,swr,sor):
    return (sw-swr)/(1-sor-swr)

Sw = np.arange(0.3,0.801,0.01)
S = saturation(Sw,0.3,0.2)

fig,ax = plt.subplots(layout='constrained')
ax.plot(Sw,corey_water_relperm(0.18,S,2.5),label='Water')
ax.plot(Sw,corey_oil_relperm(0.9,S,2.5),label='Oil')
ax.set_xlabel('Water Saturation')
ax.set_ylabel('Relative Permeability')
plt.legend()
plt.suptitle('Part B: Rel Perm Curves')
plt.savefig(r'.\Plots\HW3_part_b_rel_perm.png')

#%%

def fractional_flow(k, u, mu_o, delta_rho, g, alpha, k_ro, mu_w, k_rw):
    numerator = 1 + (k*k_ro / (u * mu_o)) * (-delta_rho * g * np.sin(alpha))
    denominator = 1 + (k_ro * mu_w) / (k_rw * mu_o)


    return numerator / denominator

def endpoint_mobility_ratio(krwo,kroo,mu_o,mu_w):
    return krwo/mu_w / (kroo/mu_o)

def question_d(k,u,mu_o,krwo,nw,kroo,no,swr,sor,mu_w):
    '''
    Res Hw3 Question D

    Parameters
    k : float
        permeability
    u : float
        darcy velocity
    mu_o : list
        oil viscosity   
    krwo : float
        endpoint water relative permeability 
    nw : float
        Corey water exponent
    kroo : float
        endpoint oil relative permeability
    no : float
        Corey oil exponent
    swr : float
        residual water saturation
    sor : float
        residual oil saturation
    mu_w : float
        water viscosity
    
    '''
    Sw = np.arange(swr,1-sor,0.01)
    S = saturation(Sw,swr,sor)
    k_rw = corey_water_relperm(krwo,S,nw)
    k_ro = corey_oil_relperm(kroo,S,no)
    S2 = S + 1E-6
    k_rw2 = corey_water_relperm(krwo,S2,nw)
    k_ro2 = corey_oil_relperm(kroo,S2,no)

    fig,ax = plt.subplots(1,2,layout='constrained',figsize=(3.5*3,3.5*2))
    for i, muo in enumerate(mu_o):
        Mo = endpoint_mobility_ratio(krwo,kroo,muo,mu_w)
        fw = fractional_flow(k,u,muo,0,32,0,k_ro,mu_w,k_rw)
        fw2 = fractional_flow(k,u,muo,0,32,0,k_ro2,mu_w,k_rw2)
        dfw = (fw2 - fw) / (S2 - S)

        ax[0].plot(S,fw,label=f'Mobility Ratio: {Mo:.2f}')
        ax[1].plot(S,dfw,label=f'Mobility Ratio: {Mo:.2f}')
    ax[0].set_xlabel('Water Saturation')
    ax[0].set_ylabel('Fractional Flow')
    ax[1].set_xlabel('Water Saturation')
    ax[1].set_ylabel('Derivative of Fractional Flow')
    
    handles, labels = ax[1].get_lines(), []
    for line in ax[1].get_lines():
        labels.append(line.get_label())

    unique_labels = []
    unique_handles = []
    for handle, label in zip(handles, labels):
        if label not in unique_labels:
            unique_labels.append(label)
            unique_handles.append(handle)
    fig.legend(unique_handles, unique_labels, 
            bbox_to_anchor=(0.25, 0.965),
            loc='upper right')
    plt.suptitle('Part D: Fractional Flow Curves')
    plt.savefig(r'.\Plots\HW3_part_d_frac_flow.png')

question_d(k=500,u=0.01,mu_o=[0.5,5,50],krwo=0.18,nw=2.5,kroo=0.9,no=2.5,swr=0.3,sor=0.2,mu_w=1)

# %%

def gravity_number(delta_rho,u, mu_o,k,kroo):
    # 144 is a conversion factor from lb/ft^3 to psi/ft
    # 5.88 is a conversion for md/cp/ft/d to ft/psi
    return delta_rho/144 * k * kroo / (mu_o * u)

# $f_w = \frac{1 - N_g^0(1-S)^{n_o}\sin\alpha}{1 + \frac{(1-S)^{n_o}}{M^0S^{n_w}}}$
def dimensionless_fractional_flow(k, u, mu_o, delta_rho, alpha, k_ro, mu_w, k_rw, S, nw, no):
    Ng = gravity_number(delta_rho, u, mu_o, k, k_ro)
    Mo = endpoint_mobility_ratio(k_rw, k_ro, mu_o, mu_w)
    numerator = 1 - Ng * (1-S)**no * np.sin(np.deg2rad(alpha))
    denominator = 1 + (1-S)**no / Mo / S**nw
    return numerator / denominator

def question_e(k, u, mu_o, krwo, nw, kroo, no, swr, sor, mu_w, delta_rho, alpha):
    '''
    Res Hw3 Question E

    Parameters
    k : float
        permeability
    u : float
        darcy velocity
    mu_o : float
        oil viscosity   
    krwo : float
        endpoint water relative permeability 
    nw : float
        Corey water exponent
    kroo : float
        endpoint oil relative permeability
    no : float
        Corey oil exponent
    swr : float
        residual water saturation
    sor : float
        residual oil saturation
    mu_w : float
        water viscosity
    delta_rho : float
        density difference
    alpha : list
        angle of inclination
    
    '''
    Sw = np.arange(swr, 1-sor, 0.01)
    S = saturation(Sw, swr, sor)
    k_rw = corey_water_relperm(krwo, S, nw)
    k_ro = corey_oil_relperm(kroo, S, no)
    S2 = S + 1E-6
    k_rw2 = corey_water_relperm(krwo, S2, nw)
    k_ro2 = corey_oil_relperm(kroo, S2, no)

    fig, ax = plt.subplots(1, 2, layout='constrained', figsize=(3.5*3, 3.5*2))
    for i in alpha:
        Ng = gravity_number(delta_rho, u, mu_o, k, k_ro)
        Ngsinalpha = Ng[-1] * np.sin(np.deg2rad(i))
        fw = dimensionless_fractional_flow(k, u, mu_o, delta_rho, i, k_ro, mu_w, k_rw, S, nw, no)
        fw2 = dimensionless_fractional_flow(k, u, mu_o, delta_rho, i, k_ro2, mu_w, k_rw2, S2, nw, no)
        dfw = (fw2 - fw) / (S2 - S)

        ax[0].plot(S, fw, label=f'Gravity Number * sin(alpha): {Ngsinalpha:.2f}')
        ax[1].plot(S, dfw, label=f'Gravity Number * sin(alpha): {Ngsinalpha:.2f}')
    ax[0].set_xlabel('Water Saturation')
    ax[0].set_ylabel('Fractional Flow')
    ax[1].set_xlabel('Water Saturation')
    ax[1].set_ylabel('Derivative of Fractional Flow')

    handles, labels = ax[1].get_lines(), []
    for line in ax[1].get_lines():
        labels.append(line.get_label())

    unique_labels = []
    unique_handles = []
    for handle, label in zip(handles, labels):
        if label not in unique_labels:
            unique_labels.append(label)
            unique_handles.append(handle)
    fig.legend(unique_handles, unique_labels, 
            bbox_to_anchor=(0.25, 0.965),
            loc='upper right')

    plt.suptitle('Part E: Dimensionless Fractional Flow Curves')
    plt.savefig(r'.\Plots\HW3_part_e_dim_frac_flow.png')
    
# %%
question_e(k=500, u=0.01, mu_o=5, krwo=0.18, nw=2.5, kroo=0.9, no=2.5, swr=0.3, sor=0.2, mu_w=1, delta_rho=62.4-35, alpha=[-30, 0, 30])
# %%
