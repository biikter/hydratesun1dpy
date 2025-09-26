import numpy as np
import constants.constantssunx as cons
import constants.computationconstants as comp

def interface_area_amyx(porosity, permeability, saturation_hydrate, saturation_water):
    # by Amyx et al., 1960, Sun and Mohanty, 2006
    return np.sqrt(porosity**3 / (2 * permeability)) \
        * (saturation_hydrate * saturation_water * (1 - saturation_hydrate - saturation_water)) ** (2/3)


def mass_rates_fun(temperature, interface_area, pressure_equilibrium, pressure, saturation_hydrate):
    mr_gas = cons.DISSOCIATION_CONSTANT * np.exp( - cons.DELTA_E_BY_R / temperature) * interface_area * (pressure_equilibrium - pressure)

    no_hydrate_mask = (mr_gas < 0) | (saturation_hydrate <= 0)
    mr_gas[no_hydrate_mask] = 0

    mr_hydrate = - mr_gas * (cons.HYDRATE_NUMBER *  cons.MOLAR_WEIGHT_WATER + cons.MOLAR_WEIGHT_GAS) / cons.MOLAR_WEIGHT_GAS
    mr_water = mr_gas * cons.HYDRATE_NUMBER * cons.MOLAR_WEIGHT_WATER/ cons.MOLAR_WEIGHT_GAS

    return mr_gas, mr_hydrate, mr_water


def calc_saturation_change(saturation_hydrate, saturation_water, mass_rate_hydrate, mass_rate_water, dV_dx, time_step):
    sat_hyd = saturation_hydrate + mass_rate_hydrate * time_step / (cons.DENSITY_HYDRATE * cons.POROSITY)
    no_hydrate_mask = sat_hyd <= 0
    sat_hyd[no_hydrate_mask] = 0

    sat_wat = saturation_water + (mass_rate_water - dV_dx * cons.DENSITY_WATER) * time_step / (cons.DENSITY_WATER * cons.POROSITY)
    saturation_water_residual_w_hydrate = cons.SATURATION_WATER_RESIDUAL * (1 - saturation_hydrate)
    residual_water_mask = sat_wat < saturation_water_residual_w_hydrate
    sat_wat[residual_water_mask] = saturation_water_residual_w_hydrate[residual_water_mask]

    sat_wat[comp.X_MESH_SIZE - 1] = sat_wat[comp.X_MESH_SIZE - 3]
    sat_wat[comp.X_MESH_SIZE - 2] = sat_wat[comp.X_MESH_SIZE - 3]

    return sat_hyd, sat_wat