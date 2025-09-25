import numpy as np

import constants.constantssunx as cons
import constants.computationconstants as comp
import utils.generatetimespanmesh as tx

density_gas = np.zeros((tx.write_span_size,comp.X_MESH_SIZE))
temperature = np.zeros((tx.write_span_size,comp.X_MESH_SIZE))
pressure = np.zeros((tx.write_span_size,comp.X_MESH_SIZE))
saturation_hydrate = np.zeros((tx.write_span_size,comp.X_MESH_SIZE))
saturation_water = np.zeros((tx.write_span_size,comp.X_MESH_SIZE))

density_gas_previous = np.zeros(comp.X_MESH_SIZE)
temperature_previous = np.zeros(comp.X_MESH_SIZE)
pressure_previous = np.zeros(comp.X_MESH_SIZE)
saturation_hydrate_previous = np.zeros(comp.X_MESH_SIZE)
saturation_water_previous = np.zeros(comp.X_MESH_SIZE)
mass_rate_hydrate = np.zeros(comp.X_MESH_SIZE)
mass_rate_gas = np.zeros(comp.X_MESH_SIZE)
mass_rate_water = np.zeros(comp.X_MESH_SIZE)
heat_flow_boundary = np.zeros(comp.X_MESH_SIZE)
# SunX
viscosity_gas = np.zeros(comp.X_MESH_SIZE)
pressure_previous_previous = np.zeros(comp.X_MESH_SIZE)

pressure_equilibrium = np.zeros(comp.X_MESH_SIZE)

# saturation_water_residual_relative = np.zeros(comp.X_MESH_SIZE)
pressure_capillary = np.zeros(comp.X_MESH_SIZE)
pressure_water = np.zeros(comp.X_MESH_SIZE)
velocity_water = np.zeros(comp.X_MESH_SIZE)
dV_dx = np.zeros(comp.X_MESH_SIZE)
dP_dt = np.zeros(comp.X_MESH_SIZE)
interface_area = np.zeros(comp.X_MESH_SIZE)
abs_perm_with_hydrate = np.zeros(comp.X_MESH_SIZE)
phase_perm_gas = np.zeros(comp.X_MESH_SIZE)
phase_perm_water = np.zeros(comp.X_MESH_SIZE)
porosity_effective = np.zeros(comp.X_MESH_SIZE)

density_gas[0,:] = cons.DENSITY_GAS_INITIAL
temperature[0,:] = cons.TEMPERATURE_INITIAL
pressure[0,:] = cons.PRESSURE_INITIAL
saturation_hydrate[0,:] = cons.SATURATION_HYDRATE_INITIAL
saturation_water[0,:] = cons.SATURATION_WATER_INITIAL

density_gas_previous[:] = cons.DENSITY_GAS_INITIAL
temperature_previous[:] = cons.TEMPERATURE_INITIAL
saturation_hydrate_previous[:] = cons.SATURATION_HYDRATE_INITIAL
saturation_water_previous[:] = cons.SATURATION_WATER_INITIAL