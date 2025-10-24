import math
import numpy as np
import matplotlib.pyplot as plt
import fenics as fe

import constants.constantssunx as cons
import constants.computationconstants as comp
import utils.generatetimespanmesh as tx
import physics.twophaseflow as pf
import physics.hydratedissociation as hd
import utils.derivativebypolynomial as dr
import utils.initializeglobalvariables as gv


gv.pressure_equilibrium[:], \
gv.porosity_effective[:], \
gv.abs_perm_with_hydrate[:], \
gv.phase_perm_gas[:], \
gv.phase_perm_water[:] = pf.calc_multiphase_flow_params(cons.TEMPERATURE_INITIAL, cons.SATURATION_WATER_INITIAL, cons.SATURATION_HYDRATE_INITIAL)

# Mesh and function spaces
mesh = fe.IntervalMesh(comp.X_MESH_SIZE-1, 0, cons.TOTAL_LENGTH)
P1 = fe.FiniteElement('P', fe.interval, 1)
element = fe.MixedElement([P1, P1])
V = fe.FunctionSpace(mesh, element)
v_1, v_2 = fe.TestFunctions(V)
u = fe.Function(V)
rho, T = fe.split(u)

# Boundary conditions
tol = 1E-14
def boundary_left(x, on_boundary):
    return on_boundary and (fe.near(x[0], 0, tol) or fe.near(x[0], 1, tol))
bc_rho = fe.DirichletBC(V.sub(0), fe.Constant(cons.DENSITY_GAS_INITIAL), boundary_left)
bc_T = fe.DirichletBC(V.sub(1), fe.Constant(cons.TEMPERATURE_INITIAL), boundary_left)

# Initial conditions
u_0 = fe.Expression(('rho_i', 'T_i'), degree=1, rho_i=cons.DENSITY_GAS_INITIAL, T_i=cons.TEMPERATURE_INITIAL)
u_n = fe.project(u_0, V)

# Variational problem

# === TIME CYCLE ===

write_counter = 1
current_time = 0.0

for time_counter in range(1,tx.time_span_size):

    local_timespan = np.array([tx.timespan[time_counter - 1], 0.5*(tx.timespan[time_counter - 1] + tx.timespan[time_counter]), tx.timespan[time_counter]])

    # SunX - viscosity by formula, Orto Buluu - viscosity fixed
    gv.viscosity_gas = pf.gas_viscosity_sun_x(gv.temperature, gv.density_gas)

    # === MAIN EQUATIONS ===

    gv.density_gas = gv.density_gas + 0.0001
    gv.temperature = gv.temperature + 0.001
    gv.pressure = gv.density_gas * cons.GAS_CONSTANT_R * gv.temperature

    # === TWO-PHASE FLOW ===

    gv.pressure_equilibrium, \
    gv.porosity_effective, \
    gv.abs_perm_with_hydrate, \
    gv.phase_perm_gas, \
    gv.phase_perm_water = pf.calc_multiphase_flow_params(gv.temperature, gv.saturation_water, gv.saturation_hydrate)

    gv.velocity_water, \
    gv.dv_dx = pf.darcylaw_phase(tx.x_mesh, gv.pressure, gv.abs_perm_with_hydrate, gv.phase_perm_water)

    gv.dp_dt = (gv.pressure - gv.pressure_previous) / tx.time_step[time_counter - 1]
    gv.pressure_previous = gv.pressure

    # Orto Buluu
    gv.heat_flow_boundary[:] = 0

    # === HYDRATE DISSOCIATION ===

    gv.interface_area = hd.interface_area_amyx(gv.porosity_effective, gv.abs_perm_with_hydrate, gv.saturation_hydrate, gv.saturation_water)

    #Orto Buluu
    gv.mass_rate_gas, \
    gv.mass_rate_hydrate, \
    gv.mass_rate_water = hd.mass_rates_fun(gv.temperature, gv.interface_area, gv.pressure_equilibrium, gv.pressure, gv.saturation_hydrate)

    gv.saturation_hydrate, \
    gv.saturation_water = hd.calc_saturation_change(gv.saturation_hydrate, gv.saturation_water, gv.mass_rate_hydrate,gv.mass_rate_water , gv.dv_dx, tx.time_step[time_counter - 1])

    # === WRITE NEW LAYER ===

    if tx.writespan[write_counter] < current_time:

        gv.density_gas_result[write_counter,:] = gv.density_gas
        gv.temperature_result[write_counter,:] = gv.temperature
        gv.pressure_result[write_counter,:] = gv.pressure

        gv.saturation_hydrate_result[write_counter, :] = gv.saturation_hydrate
        gv.saturation_water_result[write_counter, :] = gv.saturation_hydrate
        
        write_counter = write_counter + 1

        print(current_time / 60)
        print(write_counter)
        print(time_counter)

    current_time = current_time + tx.time_step[time_counter - 1]  


# === SAVE AND SEE RESULTS ===

np.save('result/density_gas.npy', gv.density_gas_result)
np.save('result/temperature.npy', gv.temperature_result)
np.save('result/pressure.npy', gv.pressure_result)
np.save('result/saturation_hydrate.npy', gv.saturation_hydrate_result)
np.save('result/saturation_water.npy', gv.saturation_water_result)

plt.clf()
plt.plot(tx.x_mesh,gv.pressure)
plt.savefig("result/pres.png")

plt.clf()
plt.plot(tx.x_mesh,gv.temperature)
plt.savefig("result/temp.png")