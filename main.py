import scipy as sp
import matplotlib as plt
import definitions as dfs
from scipy.integrate import odeint

# Purpose: Reproduce results of Time-Delayed Spatial Patterns in a Two-Dimensional Array of Coupled Oscillators
#           by Seong-Ok Jeong, Tae-Wook Ko, and Hie-Tae Moon of Korea Advanced Institute of Science and Technology.
# Project for Phys 278
# Feb 27, 2020

num_rows = 50 # 500 x 500 too big
num_cols = 50
grid_arr = sp.array([[(i,j) for j in range(num_cols)] for i in range(num_rows)])

# k and i identify rows, l and j identify columns
w = sp.pi / 5 # used in paper
K = 1 # varies, see fig 4 
v = 1
#t = None
gamma = 5 * np.pi / 32 # one of values used in paper
unit_vector_e = sp.array([1,0])
phi_0 = sp.zeros((num_rows, num_cols))
r_0 = 4 # coupling radius, experiment with this 
initial_conditions = []

W = sp.empty((num_rows, num_cols, num_rows, num_cols))
N = sp.empty((num_rows, num_cols))
# Calculate W_klij for all ij and kl. SO SLOW. 
for i in range(num_rows):
    for j in range(num_cols):
        for k in range(num_rows):
            for l in range(num_cols):
                r_klij = sp.sqrt((i-k)**2+(j-l)**2)
                W[i,j,k,l] = dfs.W(r_klij)

W_limited = dfs.calc_W_limited(W, r_0, num_rows, num_cols)
N = dfs.N(W_limited, num_cols, num_rows)
phi  = dfs.phi(gamma, unit_vector_e, phi_0, num_rows, num_cols)

t = sp.linspace(0,10,101)
solution = odeint(dfs.theODEs, initial_conditions, t)

