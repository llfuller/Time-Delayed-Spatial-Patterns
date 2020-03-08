import scipy as sp
import time

"""
Calculates distance between two points (k,l) and (i, j) (named r_kl_ij).
"""
def calc_r(k,l,i,j,num_rows,num_cols,horizontal_dist,vertical_dist,h_dist_1, h_dist_2, v_dist_1, v_dist_2):
    # some content
    r11 = sp.sqrt((h_dist_1)**2 + (v_dist_1)**2)
    r12 = sp.sqrt((h_dist_1)**2 + (v_dist_2)**2)
    r21 = sp.sqrt((h_dist_2)**2 + (v_dist_1)**2)
    r22 = sp.sqrt((h_dist_2)**2 + (v_dist_2)**2)
    r_klij = min([r11,r12,r21,r22])
    # r_klij = sp.sqrt((horizontal_dist)**2 + (vertical_dist)**2)
    return r_klij


"""
Calculates W(r_klij).
"""
def calc_W(r):
    W = sp.divide(1.0,r)
    return W

"""
Fills out W and dist_grid arrays
"""
def calc_and_save_W_and_dist_grid(num_rows, num_cols, r_0):
    W = sp.zeros((num_rows, num_cols, num_rows, num_cols))
    dist_grid_arr = -1 * sp.ones((num_rows, num_cols, num_rows, num_cols))
    start_time = time.time()
    for i in range(num_rows):
        print("Row "+str(i)+" at time "+str(time.time()-start_time))
        for j in range(num_cols):
            for k in range(num_rows):
                h_dist_1 = abs(i-k)
                h_dist_2 = num_rows - abs(i-k)
                horizontal_dist = float(h_dist_1*(h_dist_1<h_dist_2) + h_dist_2*(h_dist_2<h_dist_1))
                for l in range(num_cols):
                    v_dist_1 = abs(j - l)
                    v_dist_2 = num_cols - abs(j - l)
                    vertical_dist = float(v_dist_1 * (v_dist_1 < v_dist_2) + v_dist_2 * (v_dist_2 < v_dist_1))
                    r_klij = calc_r(i,j,k,l,num_rows, num_cols,horizontal_dist, vertical_dist, h_dist_1, h_dist_2, v_dist_1, v_dist_2)
                    if r_klij<0 or horizontal_dist<0 or vertical_dist<0:
                        print("something is wrong here")
                    if r_klij<r_0:
                        dist_grid_arr[i,j,k,l] = r_klij
                        if r_klij != 0: # to avoid infinities
                            W[i,j,k,l] = calc_W(r_klij)
                        else: # This should probably be zero so that it doesn't affect any calculations.
                            W[i, j, k, l] = 0
    sp.savez("W_matrix_"+str((num_rows,num_cols,r_0))+".npz",W=W)
    print("W matrix saved.")
    return W, dist_grid_arr

"""
Calculates sum of weights.
"""
def N(W_limited, num_cols, num_rows):
    # Full matrix N with indices (i,j)
    N = sp.sum(W_limited, axis=(2,3))
    return N

"""
Phi matrix
"""
def calc_phi(gamma, unit_vector_e, phi_0, r):
    epsilon = 0.000000000000000001 # present to avoid errors when dividing by zero for r = 0
    r_times_cos_of_angle_between = (r*unit_vector_e[0])[:,:,0]+(r*unit_vector_e[1])[:,:,1]
    print("Shape of r times cosine of angle between: "+str(sp.shape(r_times_cos_of_angle_between)))
    phi = sp.multiply(gamma , r_times_cos_of_angle_between) + phi_0
    return phi

"""
Theta matrix evaluated at specific time  
"""
def calc_theta(Omega, t, phi):
    # phi is (num_rows, num_cols) dimensional.
    theta = Omega*t+phi
    return theta

"""
Accepts (i,j) positions and evaluates (then returns) d(theta)/dt matrix (elements are (i,j) component)
"""
def d_theta_d_t(theta, t, phi, w, K, N, v, Omega, num_rows, num_cols, W_matrix, dist_grid_arr):
    sum_sub_term = sp.zeros((num_rows, num_cols))# array with indices (i,j,k,l); W*sin() part of sum
    sum_term = 0
    print("Simulating for t="+str(t))
    # Should be 2d
    theta_ij_matrix_at_t = sp.array(theta).reshape((num_rows,num_cols)) # restores theta to initial shape for matrix algebra purposes
    phi_extend = sp.array([[phi for k in range(num_rows)] for l in range(num_cols)]).transpose((2,3,1,0))
    theta_ij_matrix_at_t_extend = sp.array([[theta_ij_matrix_at_t for k in range(num_rows)] for l in range(num_cols)]).transpose((2,3,1,0))
    # Should be 4d
    theta_kl_delayed = calc_theta(Omega, t-dist_grid_arr[:,:,:,:]/v, phi_extend[:,:])
    # Should be 2d
    sum_sub_term = sp.multiply(W_matrix, sp.sin(theta_kl_delayed -  theta_ij_matrix_at_t_extend)) # (k,l)
    # Should be 2d
    sum_term = sp.sum(sum_sub_term, axis = (2,3))

    dthetadt_final = w + sp.multiply(sp.divide(float(K),N) , sum_term) # (array with indices (i,j))
    return dthetadt_final


"""
System of ODEs to be solved
"""
def theODEs(theta, t, phi, w, K, N, v, Omega, num_rows, num_cols, W_matrix, dist_grid_arr):
    # t should be a time here, I think
    dthetadt = d_theta_d_t(theta, t, phi, w, K, N, v, Omega, num_rows, num_cols, W_matrix, dist_grid_arr)
    return dthetadt.reshape(num_cols*num_rows)