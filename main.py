import scipy as sp
import matplotlib as plt
import matplotlib.pyplot as pyplt
import definitions as dfs
from scipy.integrate import odeint
import time
from os import path

# Purpose: Reproduce results of Time-Delayed Spatial Patterns in a Two-Dimensional Array of Coupled Oscillators
#           by Seong-Ok Jeong, Tae-Wook Ko, and Hie-Tae Moon of Korea Advanced Institute of Science and Technology.
# Project for Phys 278
# March 5, 2020
#######################################################################
### Parameters and Initial Conditions
#######################################################################
sp.random.seed(2020)

num_rows = 50  # 500 x 500 too big
num_cols = 50  # 150 x 150 gives a 4 GB file
                # 100 x 100 gives a 800 MB file

grid_arr = sp.array([[[i,j] for j in range(num_cols)] for i in range(num_rows)])

# k and i identify rows, l and j identify columns
w = 7.0*sp.pi / 32 # used in paper
K = 1.25 # varies, see fig 4
v = 1.0/1
Omega = 0.6*10
gamma = 5 * sp.pi / 32 # one of values used in paper
unit_vector_e = sp.array([1,1])
phi_0 = sp.zeros((num_rows, num_cols))
r_0 = 15 # coupling radius, experiment with this
initial_conditions = sp.random.random((num_rows, num_cols)).reshape(num_rows*num_cols) # Flattens initial conditions
numTimeSteps = 5
t = sp.linspace(0,1,numTimeSteps)

parameters = [num_rows, num_cols, w, K, v, Omega, gamma, unit_vector_e, r_0, initial_conditions,numTimeSteps, t]

#######################################################################
### Calculate or load W matrix
#######################################################################
W = sp.zeros((num_rows, num_cols, num_rows, num_cols))
N = sp.empty((num_rows, num_cols))
dist_grid_arr = -1*sp.ones((num_rows, num_cols, num_rows, num_cols)) # i, j, k, l
# Calculate W_klij for all ij and kl.
start_time = time.time()

# Make sure to delete old W matrix file if r_0 is changed.
loaded_r_0 = r_0 # default value
if path.exists("r_0.npz"):
    print("Loading previous r_0")
    loaded_r_0 = sp.load("r_0.npz")["r_0"]
    if loaded_r_0 != r_0:
        print("Using new r_0. Saving old r_0.")
        sp.savez("r_0_previous.npz", r_0=loaded_r_0)
    else:
        print("Using current r_0 (same as previous r_0).")
else:
    print("Previous r_0 not available. Using new r_0.")
    sp.savez("r_0.npz", r_0=r_0)


if path.exists("W_matrix_"+str((num_rows,num_cols,r_0))+".npz"): # If file exists
    W_loaded = sp.load("W_matrix_"+str((num_rows,num_cols,r_0))+".npz")["W"]
    if (sp.shape(W_loaded)[0]!=num_rows or sp.shape(W_loaded)[1]!=num_cols or r_0 != loaded_r_0): # if dimensions of loaded array not same as specified
        print("Loaded W matrix doesn't meet specified dimensions. Saving old W file and making new one.")
        sp.savez("W_matrix_previous.npz", W = W_loaded)
        W, dist_grid_arr = dfs.calc_and_save_W_and_dist_grid(num_rows, num_cols, r_0)
    if (sp.shape(W_loaded)[0]==num_rows and sp.shape(W_loaded)[1]==num_cols and (r_0 == loaded_r_0)):
        # if dimensions of loaded array are same as specified
        W = W_loaded
        print("W matrix loaded.")
else: # If file does not yet exist
    W, dist_grid_arr = dfs.calc_and_save_W_and_dist_grid(num_rows, num_cols, r_0)

#######################################################################
### Rest of the Calculations
#######################################################################
N = dfs.N(W, num_cols, num_rows)
N = 1
phi  = dfs.calc_phi(gamma, unit_vector_e, phi_0, grid_arr)
print("Calculating solution for all selected times.")
solution = odeint(dfs.theODEs, initial_conditions, t, args=(phi, w, K, N, v, Omega, num_rows, num_cols, W,
                                                            dist_grid_arr))
sp.savez("solution_("+str(num_cols)+","+str(num_rows)+")_r0_"+str(r_0), solution = solution, parameters = parameters, dist_grid_arr = dist_grid_arr)
#######################################################################
### Plot Solution
#######################################################################
print("Plotting solution")
solution_2d = solution.reshape((numTimeSteps,num_rows,num_cols))
print(sp.shape(solution))
pyplt.title("Solution at Final Time")
pyplt.imshow(solution_2d[numTimeSteps-1])
pyplt.show()

fig, ax = pyplt.subplots()
num_video_loops = 10
for j in range(num_video_loops):
    for i in range(len(solution_2d)):
        ax.cla()
        ax.imshow(solution_2d[i], cmap = 'hsv', interpolation = 'nearest')
        ax.set_title("Solution at frame {}".format(i))
        pyplt.pause(0.1)