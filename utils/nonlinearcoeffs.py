import numpy as np
import utils.generatetimespanmesh as tx
import utils.initializeglobalvariables as gv
import constants.constantssunx as cons

def density_gas_fun(k, x):
    return np.interp(x, tx.x_mesh, gv.density_gas[k, :])

def saturation_hydrate_fun(x):
    return np.interp(x, tx.x_mesh, gv.saturation_hydrate)

def saturation_water_fun(x):
    return np.interp(x, tx.x_mesh, gv.saturation_water)

def mass_rate_gas_fun(x):
    return np.interp(x, tx.x_mesh, gv.mass_rate_gas)

def abs_perm_fun(x):
    return np.interp(x, tx.x_mesh, gv.abs_perm_with_hydrate)

def phase_perm_gas_fun(x):
    return np.interp(x, tx.x_mesh, gv.phase_perm_gas)

def viscosity_gas_fun(x):
    return np.interp(x, tx.x_mesh, gv.viscosity_gas)

def heat_cond_fun(x):
    return np.interp(x, tx.x_mesh, gv.heat_conductivity)

def den_by_heat_cap_fun(x):
    return np.interp(x, tx.x_mesh, gv.density_by_heat_capacity)

def temperature_fun(k, x):
    return np.interp(x, tx.x_mesh, gv.temperature[k, :])

def velocity_gas_fun(x):
    return np.interp(x, tx.x_mesh, gv.velocity_gas)

def velocity_water_fun(x):
    return np.interp(x, tx.x_mesh, gv.velocity_water)

def mass_rate_hydrate_fun(x):
    return np.interp(x, tx.x_esh, gv.mass_rate_hydrate)