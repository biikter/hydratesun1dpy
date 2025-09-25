import math
import numpy as np
import matplotlib.pyplot as plt

import constants.constantssunx as cons
import constants.computationconstants as comp
import utils.generatetimespanmesh as tx
import physical_functions.phase_flow as pf
import utils.derivativebypolynomial as dr
import utils.initializeglobalvariables as gv


gv.pressure_equilibrium[:], \
gv.porosity_effective[:], \
gv.abs_perm_with_hydrate[:], \
gv.phase_perm_gas[:], \
gv.phase_perm_water[:] = pf.calc_multiphase_flow_params(cons.TEMPERATURE_INITIAL, cons.SATURATION_WATER_INITIAL, cons.SATURATION_HYDRATE_INITIAL)

# === TIME CYCLE ===

write_counter = 1
current_time = 0.0

for time_counter in range(1,tx.time_span_size):

    local_timespan = np.array([tx.timespan[time_counter - 1], 0.5*(tx.timespan[time_counter - 1] + tx.timespan[time_counter]), tx.timespan[time_counter]])

    # SunX - viscosity by formula, Orto Buluu - viscosity fixed
    gv.viscosity_gas = pf.gas_viscosity_sun_x(gv.temperature_previous, gv.density_gas_previous)

    # === MAIN EQUATIONS ===

    gv.density_gas_previous = gv.density_gas_previous + 0.0001
    gv.temperature_previous = gv.temperature_previous + 0.001
    gv.pressure_previous = gv.density_gas_previous * cons.GAS_CONSTANT_R * gv.temperature_previous

    gv.pressure_equilibrium, \
    gv.porosity_effective, \
    gv.abs_perm_with_hydrate, \
    gv.phase_perm_gas, \
    gv.phase_perm_water = pf.calc_multiphase_flow_params(gv.temperature_previous, gv.saturation_water_previous, gv.saturation_hydrate_previous)

    deriv_pres = dr.derivative_by_polynomial(tx.x_mesh, gv.pressure_previous)
    gv.velocity_water = - gv.abs_perm_with_hydrate * gv.phase_perm_water * deriv_pres / cons.VISCOSITY_WATER
    gv.dV_dx = dr.derivative_by_polynomial(tx.x_mesh, gv.velocity_water)

    gv.dP_dt = (gv.pressure_previous - gv.pressure_previous_previous) / tx.time_step[time_counter - 1]

        # by Amyx et al., 1960, Sun and Mohanty, 2006
    gv.interface_area = np.sqrt(gv.porosity_effective**3 / (2 * gv.abs_perm_with_hydrate)) \
        * (gv.saturation_hydrate_previous * gv.saturation_water_previous * (1 - gv.saturation_hydrate_previous - gv.saturation_water_previous)) ** (2/3)

        #Orto Buluu
    gv.mass_rate_gas = cons.DISSOCIATION_CONSTANT * np.exp( - cons.DELTA_E_BY_R / gv.temperature_previous) * gv.interface_area * (gv.pressure_equilibrium - gv.pressure_previous)

    no_hydrate_mask = (gv.mass_rate_gas < 0) | (gv.saturation_hydrate_previous <= 0)
    gv.mass_rate_gas[no_hydrate_mask] = 0

    gv.mass_rate_hydrate = - gv.mass_rate_gas * (cons.HYDRATE_NUMBER *  cons.MOLAR_WEIGHT_WATER + cons.MOLAR_WEIGHT_GAS) / cons.MOLAR_WEIGHT_GAS
    gv.mass_rate_water = gv.mass_rate_gas * cons.HYDRATE_NUMBER * cons.MOLAR_WEIGHT_WATER/ cons.MOLAR_WEIGHT_GAS

    gv.saturation_hydrate_previous = gv.saturation_hydrate_previous + gv.mass_rate_hydrate * tx.time_step[time_counter - 1] / (cons.DENSITY_HYDRATE * cons.POROSITY)
    no_hydrate_mask_2 = gv.saturation_hydrate_previous <= 0
    gv.saturation_hydrate_previous[no_hydrate_mask_2] = 0

    gv.saturation_water_previous = gv.saturation_water_previous + (gv.mass_rate_water - gv.dV_dx * cons.DENSITY_WATER) * tx.time_step[time_counter - 1] / (cons.DENSITY_WATER * cons.POROSITY)
    swp = cons.SATURATION_WATER_RESIDUAL * (1 - gv.saturation_hydrate_previous)
    saturation_mask = gv.saturation_water_previous < swp
    gv.saturation_water_previous[saturation_mask] = swp[saturation_mask]

    gv.saturation_water_previous[comp.X_MESH_SIZE - 1] = gv.saturation_water_previous[comp.X_MESH_SIZE - 3]
    gv.saturation_water_previous[comp.X_MESH_SIZE - 2] = gv.saturation_water_previous[comp.X_MESH_SIZE - 3]

    # Orto Buluu
    gv.heat_flow_boundary[:] = 0

    gv.pressure_previous_previous = gv.pressure_previous

    # === WRITE NEW LAYER ===

    if tx.writespan[write_counter] < current_time:

        gv.density_gas[write_counter,:] = gv.density_gas_previous
        gv.temperature[write_counter,:] = gv.temperature_previous
        gv.pressure[write_counter,:] = gv.pressure_previous

        gv.saturation_hydrate[write_counter, :] = gv.saturation_hydrate_previous
        gv.saturation_water[write_counter, :] = gv.saturation_hydrate_previous
        
        write_counter = write_counter + 1

        print(current_time / 60)
        print(write_counter)
        print(time_counter)

    current_time = current_time + tx.time_step[time_counter - 1]  


# === SEE RESULTS ===

y_plot = poly_2(tx.x_mesh)
plt.plot(tx.x_mesh, y_plot)
plt.savefig("plots/deriv.png")

plt.clf()
plt.plot(tx.x_mesh,gv.pressure_previous)
plt.savefig("plots/pres.png")

plt.clf()
plt.plot(tx.x_mesh,gv.temperature_previous)
plt.savefig("plots/temp.png")