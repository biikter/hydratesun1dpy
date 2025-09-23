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
x_step = cons.TOTAL_LENGTH / (x_mesh_size - 1)
x_mesh = np.arange(0, cons.TOTAL_LENGTH, x_step)

# === INITIALIZE ===

density_gas = np.zeros((write_span_size,x_mesh_size))
temperature = np.zeros((write_span_size,x_mesh_size))
pressure = np.zeros((write_span_size,x_mesh_size))
saturation_hydrate = np.zeros((write_span_size,x_mesh_size))
saturation_water = np.zeros((write_span_size,x_mesh_size))

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

density_gas[0,:] = cons.DENSITY_GAS_INITIAL
temperature[0,:] = cons.TEMPERATURE_INITIAL
pressure[0,:] = cons.PRESSURE_INITIAL
saturation_hydrate[0,:] = cons.SATURATION_HYDRATE_INITIAL
saturation_water[0,:] = cons.SATURATION_WATER_INITIAL

density_gas_previous[:] = cons.DENSITY_GAS_INITIAL
temperature_previous[:] = cons.TEMPERATURE_INITIAL
saturation_hydrate_previous[:] = cons.SATURATION_HYDRATE_INITIAL
saturation_water_previous[:] = cons.SATURATION_WATER_INITIAL

phase_perm_gas[:] = (((1 - cons.SATURATION_WATER_INITIAL - cons.SATURATION_HYDRATE_INITIAL) / (1 - cons.SATURATION_HYDRATE_INITIAL) - cons.SATURATION_GAS_RESIDUAL)
        / (1 - cons.SATURATION_GAS_RESIDUAL - cons.SATURATION_WATER_RESIDUAL)) ** 2
phase_perm_water[:] = ((cons.SATURATION_WATER_INITIAL / (1 - cons.SATURATION_HYDRATE_INITIAL) - cons.SATURATION_WATER_RESIDUAL)
        / (1 - cons.SATURATION_WATER_RESIDUAL - cons.SATURATION_GAS_RESIDUAL)) ** 4
porosity_effective[:] = (1 - cons.SATURATION_HYDRATE_INITIAL) * cons.POROSITY

# === TIME CYCLE ===

write_counter = 2
current_time = 0.0

for time_counter in range(1,time_span_size):

    local_timespan = np.array([timespan[time_counter - 1], 0.5*(timespan[time_counter - 1] + timespan[time_counter]), timespan[time_counter]])
    
        # SunX - viscosity by formula, Orto Buluu - viscosity fixed
    viscosity_gas = 2.4504e-3 + 2.8764e-5 * temperature_previous + 3.279e-9 * temperature_previous * temperature_previous \
        - 3.7838e-12 * temperature_previous * temperature_previous * temperature_previous + 2.0891e-5 * density_gas_previous \
        + 2.5127e-7 * density_gas_previous * density_gas_previous - 5.822e-10 * density_gas_previous * density_gas_previous * density_gas_previous \
        + 1.8378e-13 * density_gas_previous * density_gas_previous * density_gas_previous * density_gas_previous
    
    viscosity_gas = 0.001 * viscosity_gas

    # === MAIN EQUATIONS ===

    density_gas_previous = density_gas_previous + 1
    temperature_previous = temperature_previous + 1
    pressure_previous = density_gas_previous * cons.GAS_CONSTANT_R * temperature_previous

    pressure_equilibrium = 1.15 * np.exp(cons.A_W + cons.B_W / temperature_previous)

    porosity_effective = (1 - saturation_hydrate_previous) * cons.POROSITY

        # Masuda et al., 1999 - N = 10
    abs_perm_with_hydrate = cons.PERMEABILITY * (1 - saturation_hydrate_previous) ** 15
    
    phase_perm_gas = (((1 - saturation_water_previous - saturation_hydrate_previous) / (1 - saturation_hydrate_previous) - cons.SATURATION_GAS_RESIDUAL) \
        / (1 - cons.SATURATION_GAS_RESIDUAL - cons.SATURATION_WATER_RESIDUAL)) ** 2
    
    phase_perm_water = ((saturation_water_previous / (1 - saturation_hydrate_previous) - cons.SATURATION_WATER_RESIDUAL) \
        / (1 - cons.SATURATION_WATER_RESIDUAL - cons.SATURATION_WATER_RESIDUAL)) ** 4
    
    coefficients = np.polyfit(x_mesh, pressure_previous, 3)
    polynomial_function = np.poly1d(coefficients)
    derivative = polynomial_function.deriv()
    result_at_x2 = derivative(0.1)

    dP_dt = (pressure_previous - pressure_previous_previous) / time_step[timecounter - 1]