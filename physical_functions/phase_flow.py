import numpy as np
import constants.constantssunx as cons

def calc_multiphase_flow_params(temperature, saturation_water, saturation_hydrate):
    equil_pres = equilibrium_pressure_fun(temperature)
    eff_por = effective_porosity_fun(saturation_hydrate)
    abs_perm = absolute_permeability_w_hydrate_masuda(saturation_hydrate)
    phase_perm_gas = phase_permeability_gas_fun(saturation_water, saturation_hydrate)
    phase_perm_water = phase_permeability_water_fun(saturation_water, saturation_hydrate)

    return equil_pres, eff_por, abs_perm, phase_perm_gas, phase_perm_water

def effective_porosity_fun(saturation_hydrate):
    return (1 - saturation_hydrate) * cons.POROSITY

def absolute_permeability_w_hydrate_masuda(saturation_hydrate):
    # Masuda et al., 1999 - N = 10
    N = 15
    return cons.PERMEABILITY * (1 - saturation_hydrate) ** N
'''
%   Yousif et al., 1991
%     for i = 1:XMeshSize
%         if PorosityEffective(i) < 0.11
%             AbsPermWithHydrate(i) = 1e-15 * 5.51721 * PorosityEffective(i)^0.86;
%         else
%             AbsPermWithHydrate(i) = 1e-15 * 4.84653e8 * PorosityEffective(i)^9.13;
%         end
%     end
'''

def phase_permeability_gas_fun(saturation_water, saturation_hydrate):
    return (((1 - saturation_water - saturation_hydrate) / (1 - saturation_hydrate) - cons.SATURATION_GAS_RESIDUAL) \
        / (1 - cons.SATURATION_GAS_RESIDUAL - cons.SATURATION_WATER_RESIDUAL)) ** 2

def phase_permeability_water_fun(saturation_water, saturation_hydrate):
    return ((saturation_water / (1 - saturation_hydrate) - cons.SATURATION_WATER_RESIDUAL) \
        / (1 - cons.SATURATION_WATER_RESIDUAL - cons.SATURATION_WATER_RESIDUAL)) ** 4

def equilibrium_pressure_fun(temperature):
    return 1.15 * np.exp(cons.A_W + cons.B_W / temperature)

def gas_viscosity_sun_x(temperature, density_gas):
    visc = 2.4504e-3 + 2.8764e-5 * temperature + 3.279e-9 * temperature**2 \
            - 3.7838e-12 * temperature**3 + 2.0891e-5 * density_gas \
            + 2.5127e-7 * density_gas**2 - 5.822e-10 * density_gas**3 \
            + 1.8378e-13 * density_gas**4
    return 0.001 * visc