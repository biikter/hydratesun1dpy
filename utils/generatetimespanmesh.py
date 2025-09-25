import math
import numpy as np
import constants.computationconstants as comp
import constants.constantssunx as cons

# === TIME SPAN ===

time_step = np.full(4000,0.01)
time_step = np.append(time_step, np.full(6000,1.0))
time_step = np.append(time_step, np.full(10000,10.0))
time_step = np.append(time_step, np.full(math.ceil((comp.MAX_TIME - 106040.0)/60) + 1,60.0))
time_span_size = len(time_step)

timespan = np.zeros(time_span_size)
for time_counter in range(1,time_span_size):
    timespan[time_counter] = timespan[time_counter - 1] + time_step[time_counter - 1]

writespan = np.arange(0, comp.MAX_TIME + comp.WRITE_STEP, comp.WRITE_STEP)
write_span_size = len(writespan)


# === MESH ===

x_step = cons.TOTAL_LENGTH / (comp.X_MESH_SIZE - 1)
x_mesh = np.linspace(0, cons.TOTAL_LENGTH, num=comp.X_MESH_SIZE)