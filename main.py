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

num_rows = 64  # 500 x 500 too big
num_cols = 64  # 150 x 150 gives a 4 GB file
                # 100 x 100 gives a 800 MB file

grid_arr = sp.array([[[i,j] for j in range(num_cols)] for i in range(num_rows)])

# The three most important tunable parameters:
use_3d_params = False
use_4a_params = True
#Fig 3d
if use_3d_params:
    K = 0.6 # varies, see fig 4
    r_0 = 10 # coupling radius, experiment with this
    v = 1.0/1.1
# Fig 4a
if use_4a_params:
    K = 1.0 # varies, see fig 4
    r_0 = 4 # coupling radius, experiment with this
    v = 1.0/1.0
# Further parameters:
w = sp.pi/5.0 # used in paper. Do not change this.
Omega = 0.6
gamma = 5 * sp.pi / 32 # one of values used in paper
unit_vector_e = sp.array([0,1])
phi_0 = 0
# initial_conditions = sp.zeros((num_rows, num_cols)).reshape(num_rows*num_cols) # Flattens initial conditions
initial_conditions = sp.random.random((num_rows, num_cols)).reshape(num_rows*num_cols) # Flattens initial conditions
# initial_conditions = sp.array([[gamma*i for j in range(num_cols)] for i in range(num_rows)]).reshape(num_rows*num_cols) # Flattens initial conditions
numTimeSteps = 30
t = sp.linspace(0,8,numTimeSteps)
# phi = dfs.calc_phi(gamma, unit_vector_e, phi_0, grid_arr)
phi = sp.array([[gamma*i for j in range(num_cols)] for i in range(num_rows)])
# phi = sp.random.random((num_rows,num_cols))

parameters = [num_rows, num_cols, w, K, v, Omega, gamma, unit_vector_e, r_0, initial_conditions,numTimeSteps, t]

#######################################################################
### Calculate W matrix
#######################################################################
W = sp.zeros((num_rows, num_cols, num_rows, num_cols))
N = sp.empty((num_rows, num_cols))
dist_grid_arr = sp.zeros((num_rows, num_cols, num_rows, num_cols)) # i, j, k, l
# Calculate W_klij for all ij and kl.
start_time = time.time()
W, dist_grid_arr = dfs.calc_W_and_dist_grid(num_rows, num_cols, r_0)
print("W and dist_grid_arr calculated.")
# Optional code to sample how W and dist look to make sure they look correct for chosen (i,j)
# pyplt.figure()
# pyplt.imshow(dist_grid_arr[round(num_rows/5),0,:,:])
# pyplt.figure()
# pyplt.imshow(W[round(num_rows/5),0,:,:])
# pyplt.show()

#######################################################################
### Rest of the Calculations
#######################################################################
N = dfs.calc_N(W)
# phi  = dfs.calc_phi(gamma, unit_vector_e, phi_0, grid_arr)
print("Calculating solution for all selected times.")
solution = odeint(dfs.theODEs, initial_conditions, t, args=(phi, w, K, N, v, Omega, num_rows, num_cols, W,
                                                            dist_grid_arr))
sp.savez("solution_("+str(num_cols)+","+str(num_rows)+")_r0_"+str(r_0), solution = solution, parameters = parameters, dist_grid_arr = dist_grid_arr)
#######################################################################
### Plot Solution
#######################################################################
# loading
solution = sp.load("solution_("+str(num_cols)+","+str(num_rows)+")_r0_"+str(r_0)+'.npz')['solution']
print("Plotting solution")
solution_2d = solution.reshape((numTimeSteps,num_rows,num_cols))
print(sp.shape(solution))
pyplt.title("Solution at Final Time")
pyplt.imshow(solution_2d[numTimeSteps-1])
pyplt.show()

fig, ax = pyplt.subplots()
num_video_loops = 100
for j in range(num_video_loops):
    for i in range(len(solution_2d)):
        ax.cla()
        ax.imshow(solution_2d[i])#, cmap = 'twilight',interpolation = 'bicubic')
        ax.set_title("Solution at frame {}".format(i))
        pyplt.pause(0.1)