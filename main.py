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
import utils.nonlinearcoeffs as coef


gv.pressure_equilibrium[:], \
gv.porosity_effective[:], \
gv.abs_perm_with_hydrate[:], \
gv.phase_perm_gas[:], \
gv.phase_perm_water[:] = pf.calc_multiphase_flow_params(cons.TEMPERATURE_INITIAL, cons.SATURATION_WATER_INITIAL, cons.SATURATION_HYDRATE_INITIAL)


# === FENICS INITIALIZATION ===

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
rho_D = fe.Expression('den_b + (den_i - den_b) * exp(- t / 2)', degree=1, den_b=cons.DENSITY_GAS_BOUNDARY, den_i=cons.DENSITY_GAS_INITIAL, t=0)
bc_rho = fe.DirichletBC(V.sub(0), rho_D, boundary_left)
bc_T = fe.DirichletBC(V.sub(1), fe.Constant(cons.TEMPERATURE_BOUNDARY), boundary_left)
bc = [bc_rho, bc_T]

# Initial conditions
u_0 = fe.Expression(('rho_i', 'T_i'), degree=1, rho_i=cons.DENSITY_GAS_INITIAL, T_i=cons.TEMPERATURE_INITIAL)
u_n = fe.project(u_0, V)

# Define constants used in variational forms
time_counter = 1
t = fe.Constant(time_counter - 1)
k = fe.Constant(cons.POROSITY)
R = fe.Constant(cons.GAS_CONSTANT_R)
sigma = fe.Constant(cons.THROTTLING_COEFFICIENT)
c_g = fe.Constant(cons.HEAT_CAPACITY_GAS_P)
c_w = fe.Constant(cons.HEAT_CAPACITY_WATER)
den_w = fe.Constant(cons.DENSITY_WATER)
e_A = fe.Constant(cons.ENTHALPY_A)
e_B = fe.Constant(cons.ENTHALPY_B)

# Variational problem
x = fe.SpatialCoordinate(mesh)
dt = tx.time_step[0]

F = k * rho * (1 - coef.saturation_hydrate_fun(x) - coef.saturation_water_fun(x)) * v_1 * fe.dx \
    + (k * coef.density_gas_fun(t, x) * (1 - coef.saturation_hydrate_fun(x) - coef.saturation_water_fun(x)) + dt * coef.mass_rate_gas_fun(x)) * v_1 * fe.dx \
    + R * dt * rho * coef.abs_perm_fun(x) * coef.phase_perm_gas_fun(x) / coef.viscosity_gas_fun(x) * fe.dot(fe.grad(rho * T), fe.grad(v_1)) * fe.dx \
    + coef.heat_cond_fun(x) * dt * fe.dot(fe.grad(T), fe.grad(v_2)) * fe.dx \
    + coef.den_by_heat_cap_fun(x) * (T - coef.temperature_fun(t, x)) * v_2 *fe.dx \
    + k * coef.density_gas_fun(t, x) * sigma * (1 - coef.saturation_hydrate_fun(x) - coef.saturation_water_fun(x)) * (rho - coef.density_gas_fun(t, x)) * v_2 * fe.dx \
    + dt * coef.density_gas_fun(x) * c_g * coef.velocity_gas_fun(x) * fe.grad(T) * v_2 * fe.dt \
    + dt * den_w * c_w * coef.velocity_water_fun(x) * fe.grad(T) * v_2 * fe.dt \
    + rho * sigma * coef.velocity_gas_fun(x) * R * dt * fe.grad(rho * T) * v_2 * fe.dx \
    + dt * coef.mass_rate_hydrate_fun(x) * (e_A - e_B * T) * v_2 * fe.dx

# Create VTK files for visualization output
vtkfile_pressure = fe.File('results_vtk/pressure.pvd')
vtkfile_temperature = fe.File('results_vtk/temperature.pvd')


# === TIME CYCLE ===

write_counter = 1
current_time = 0.0

for time_counter in range(1,tx.time_span_size):

    # === SOLVE ===

    rho_D.t = current_time

    fe.solve(F == 0, u, bc)

    gv.density_gas[time_counter, :], gv.temperature[time_counter, :] = u.split()
    gv.pressure[time_counter, :] = gv.density_gas[time_counter, :] * cons.GAS_CONSTANT_R * gv.temperature[time_counter, :]

    #gv.density_gas = gv.density_gas + 0.0001
    #gv.temperature = gv.temperature + 0.001


    # === TWO-PHASE FLOW ===

    # SunX - viscosity by formula, Orto Buluu - viscosity fixed
    gv.viscosity_gas = pf.gas_viscosity_sun_x(gv.temperature, gv.density_gas)

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

        # Save solution to file (VTK)
        vtkfile_pressure << (gv.pressure, current_time)
        vtkfile_temperature << (gv.temperature, current_time)

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