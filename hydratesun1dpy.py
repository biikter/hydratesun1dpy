import math
import numpy as np

# === CONSTANTS FROM ANOTHER PY-FILE ===

import hydratesun1dconstantssunx as cons

# === TIME SPAN ===

max_time = 110000.0 # sec
time_step = np.full(4000,0.01)
time_step = np.append(time_step, np.full(6000,1.0))
time_step = np.append(time_step, np.full(10000,10.0))
time_step = np.append(time_step, np.full(math.ceil((max_time - 106040.0)/60) + 1,60.0))
time_span_size = len(time_step)

timespan = np.zeros(time_span_size)
for timecounter in range(1,time_span_size):
    timespan[timecounter] = timespan[timecounter - 1] + time_step[timecounter - 1]

write_step = 10.0 # sec
writespan = np.arange(0, max_time, write_step)
write_span_size = len(writespan)

# === MESH ===

x_mesh_size = 51
x_step = cons.total_length / (x_mesh_size - 1)
x_mesh = np.arange(0, cons.total_length, x_step)

# === INITIALIZE ===

density_gas = np.zeros(write_span_size,x_mesh_size)
temperature = np.zeros(write_span_size,x_mesh_size)
pressure = np.zeros(write_span_size,x_mesh_size)
saturation_hydrate = np.zeros(write_span_size,x_mesh_size)
saturation_water = np.zeros(write_span_size,x_mesh_size)

density_gas_previous = np.zeros(x_mesh_size)
temperature_previous = np.zeros(x_mesh_size)
pressure_previous = np.zeros(x_mesh_size)
saturation_hydrate_previous = np.zeros(x_mesh_size)
saturation_water_previous = np.zeros(x_mesh_size)
mass_rate_hydrate = np.zeros(x_mesh_size)
mass_rate_gas = np.zeros(x_mesh_size)
mass_rate_water = np.zeros(x_mesh_size)
heat_flow_boundary = np.zeros(x_mesh_size)
# SunX
viscosity_gas = np.zeros(x_mesh_size)
pressure_previous_previous = np.zeros(x_mesh_size)

pressure_equilibrium = np.zeros(x_mesh_size)

# saturation_water_residual_relative = np.zeros(x_mesh_size)
pressure_capillary = np.zeros(x_mesh_size)
pressure_water = np.zeros(x_mesh_size)
velocity_water = np.zeros(x_mesh_size)
dV_dx = np.zeros(x_mesh_size)
dP_dt = np.zeros(x_mesh_size)
interface_area = np.zeros(x_mesh_size)
abs_perm_with_hydrate = np.zeros(x_mesh_size)
phase_perm_gas = np.zeros(x_mesh_size)
phase_perm_water = np.zeros(x_mesh_size)
porosity_effective = np.zeros(x_mesh_size)

density_gas(1,:) = cons.DENSITY_GAS_INITIAL
temperature(1,:) = cons.TEMPERATURE_INITIAL;
pressure(1,:) = cons.PRESSURE_INITIAL;
saturation_hydrate(1,:) = cons.SATURATION_HYDRATE_INITIAL;
saturation_water(1,:) = cons.SATURATION_WATER_INITIAL;

print(time_step[20030])
print(timespan[20030])
print(timespan[20066])
print(writespan[100])
print(write_span_size)
print(time_span_size)
print(x_mesh[48])
print(x_mesh[49])