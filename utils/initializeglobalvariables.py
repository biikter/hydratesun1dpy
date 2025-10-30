import numpy as np

import constants.constantssunx as cons
import constants.computationconstants as comp
import utils.generatetimespanmesh as tx
import physics.twophaseflow as pf


density_gas = np.full(comp.X_MESH_SIZE, cons.DENSITY_GAS_INITIAL)
temperature = np.full(comp.X_MESH_SIZE, cons.TEMPERATURE_INITIAL)
pressure = np.full(comp.X_MESH_SIZE, cons.PRESSURE_INITIAL)
saturation_hydrate = np.full(comp.X_MESH_SIZE, cons.SATURATION_HYDRATE_INITIAL)
saturation_water = np.full(comp.X_MESH_SIZE, cons.SATURATION_WATER_INITIAL)
mass_rate_hydrate = np.zeros(comp.X_MESH_SIZE)
mass_rate_gas = np.zeros(comp.X_MESH_SIZE)
mass_rate_water = np.zeros(comp.X_MESH_SIZE)
heat_flow_boundary = np.zeros(comp.X_MESH_SIZE)
# SunX
viscosity_gas = np.full(comp.X_MESH_SIZE, pf.gas_viscosity_sun_x(cons.TEMPERATURE_INITIAL, cons.DENSITY_GAS_INITIAL))
pressure_previous = np.full(comp.X_MESH_SIZE, cons.PRESSURE_INITIAL)

pressure_equilibrium = np.zeros(comp.X_MESH_SIZE)

# saturation_water_residual_relative = np.zeros(comp.X_MESH_SIZE)
# pressure_capillary = np.zeros(comp.X_MESH_SIZE)
# pressure_water = np.zeros(comp.X_MESH_SIZE)
velocity_water = np.zeros(comp.X_MESH_SIZE)
velocity_gas = np.zeros(comp.X_MESH_SIZE)
dv_dx = np.zeros(comp.X_MESH_SIZE)
dp_dt = np.zeros(comp.X_MESH_SIZE)
interface_area = np.zeros(comp.X_MESH_SIZE)
abs_perm_with_hydrate = np.full(comp.X_MESH_SIZE, pf.absolute_permeability_w_hydrate_masuda(cons.SATURATION_HYDRATE_INITIAL))
phase_perm_gas = np.full(comp.X_MESH_SIZE, pf.phase_permeability_gas_fun(cons.SATURATION_WATER_INITIAL, cons.SATURATION_HYDRATE_INITIAL))
phase_perm_water = np.full(comp.X_MESH_SIZE, pf.phase_permeability_water_fun(cons.SATURATION_WATER_INITIAL, cons.SATURATION_HYDRATE_INITIAL))
porosity_effective = np.zeros(comp.X_MESH_SIZE)

heat_conductivity = cons.POROSITY * (1 - saturation_hydrate - saturation_water) * cons.HEAT_CONDUCTIVITY_GAS \
    + cons.POROSITY * saturation_water * cons.HEAT_CONDUCTIVITY_WATER \
    + cons.POROSITY * saturation_hydrate * cons.HEAT_CONDUCTIVITY_HYDRATE \
    + (1 - cons.POROSITY) * cons.HEAT_CONDUCTIVITY_ROCK

density_by_heat_capacity = cons.POROSITY * (1 - saturation_hydrate - saturation_water) * density_gas * cons.HEAT_CAPACITY_GAS_P \
    + cons.POROSITY * saturation_hydrate * cons.DENSITY_HYDRATE * cons.HEAT_CAPACITY_HYDRATE \
    + cons.POROSITY * saturation_water * cons.DENSITY_WATER * cons.HEAT_CAPACITY_WATER \
    + (1 - cons.POROSITY) * cons.HEAT_CAPACITY_ROCK


density_gas_result = np.zeros((tx.write_span_size,comp.X_MESH_SIZE))
temperature_result = np.zeros((tx.write_span_size,comp.X_MESH_SIZE))
pressure_result = np.zeros((tx.write_span_size,comp.X_MESH_SIZE))
saturation_hydrate_result = np.zeros((tx.write_span_size,comp.X_MESH_SIZE))
saturation_water_result = np.zeros((tx.write_span_size,comp.X_MESH_SIZE))

density_gas_result[0,:] = cons.DENSITY_GAS_INITIAL
temperature_result[0,:] = cons.TEMPERATURE_INITIAL
pressure_result[0,:] = cons.PRESSURE_INITIAL
saturation_hydrate_result[0,:] = cons.SATURATION_HYDRATE_INITIAL
saturation_water_result[0,:] = cons.SATURATION_WATER_INITIAL