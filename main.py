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

num_rows = 25  # 500 x 500 too big
num_cols = 25  # 150 x 150 gives a 4 GB file
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
# initial_conditions = sp.random.random((num_rows, num_cols)).reshape(num_rows*num_cols) # Flattens initial conditions
initial_conditions = sp.array([[gamma*i for j in range(num_cols)] for i in range(num_rows)]).reshape(num_rows*num_cols) # Flattens initial conditions
# initial_conditions = sp.array([[gamma*(i+j) for j in range(num_cols)] for i in range(num_rows)]).reshape(num_rows*num_cols) # Flattens initial conditions
# initial_conditions = sp.array([[gamma*(i*j) for j in range(num_cols)] for i in range(num_rows)]).reshape(num_rows*num_cols) # Flattens initial conditions
numTimeSteps = 10
t = sp.linspace(0,8,numTimeSteps)
# phi = dfs.calc_phi(gamma, unit_vector_e, phi_0, grid_arr)
phi = sp.array([[gamma*i for j in range(num_cols)] for i in range(num_rows)])
# phi = sp.random.random((num_rows,num_cols))


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

#######################################################################
### Rest of the Calculations
#######################################################################
N = dfs.calc_N(W)
scalars = sp.array([w, K, v, Omega, gamma, num_rows, num_cols, numTimeSteps])
dist_grid_arr_flattened = dist_grid_arr.reshape((num_rows*num_cols*num_rows*num_cols))
lags_whole = dist_grid_arr/v # needed to reduce number of lags to make memory manageable.
lags_reduced = sp.sort(sp.unique(dist_grid_arr_flattened/v))[1:] # needed to reduce number of lags to make memory manageable.
lag_indices = sp.zeros((num_rows,num_cols,num_rows,num_cols))
i=0
l=0
for j in range(num_cols):
    for k in range(num_rows):
        if sp.size(sp.where(lags_reduced == lags_whole[i,j,k,l])[0])!=0:
            # print(sp.where(lags_reduced == lags_whole[i,j,k,l])[0][0])
            lag_indices[i,j,k,l] = sp.where(lags_reduced == lags_whole[i,j,k,l])[0]+1
for i in range(1,num_rows):
    lag_indices[i, :, :, :] += sp.roll(lag_indices[0,:,:,:],i,axis=1)
for l in range(1,num_cols):
    lag_indices[:, :, :, l] += sp.roll(lag_indices[:,:,:,0],l,axis=1)
lag_indices[lag_indices==0] = sp.nan
sp.save('generated_values/W.npy', W)
sp.save('generated_values/dist_grid_arr.npy', dist_grid_arr)
sp.save('generated_values/dist_grid_arr_flattened.npy', dist_grid_arr_flattened)
sp.save('generated_values/lags_reduced.npy', lags_reduced)
sp.save('generated_values/lags_whole.npy', lags_whole)
sp.save('generated_values/lag_indices.npy', lag_indices)
sp.save('generated_values/scalars.npy', scalars)
sp.save('generated_values/phi.npy', phi)
sp.save('generated_values/N.npy', N)
sp.save('generated_values/initial_conditions.npy',initial_conditions)
print("Now run MATLAB code DDE.m to calculate solution for all selected times.")
# Now run MATLAB code DDE.m