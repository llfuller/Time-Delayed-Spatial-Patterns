import scipy as sp
import time

"""
Calculates distance between two points (k,l) and (i, j) (named r_kl_ij).
"""
def calc_r(h_dist_1, h_dist_2, v_dist_1, v_dist_2):
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
def calc_W_and_dist_grid(num_rows, num_cols, r_0):
    W = sp.zeros((num_rows, num_cols, num_rows, num_cols))
    dist_grid_arr = sp.zeros((num_rows, num_cols, num_rows, num_cols))
    start_time = time.time()
    i = 0
    l = 0
    for j in range(num_cols):
        for k in range(num_rows):
            h_dist_1 = abs(i-k)
            h_dist_2 = num_rows - abs(i-k)
            v_dist_1 = abs(j - l)
            v_dist_2 = num_cols - abs(j - l)
            if v_dist_1!=v_dist_2:
                vertical_dist = float(v_dist_1 * (v_dist_1 < v_dist_2) + v_dist_2 * (v_dist_2 < v_dist_1))
            else:
                vertical_dist = float(v_dist_1)
            if h_dist_1!=h_dist_2:
                horizontal_dist = float(h_dist_1 * (h_dist_1 < h_dist_2) + h_dist_2 * (h_dist_2 < h_dist_1))
            else:
                horizontal_dist = float(h_dist_1)
            r_klij = sp.sqrt(horizontal_dist**2 + vertical_dist**2) #calc_r(h_dist_1, h_dist_2, v_dist_1, v_dist_2)
            if r_klij<=r_0:
                dist_grid_arr[i,j,k,l] = r_klij
                if r_klij != 0: # to avoid infinities
                    W[i,j,k,l] = calc_W(r_klij)
                else: # This should probably be zero so that it doesn't affect any calculations.
                    W[i, j, k, l] = 0
    for i in range(1,num_rows): # already did i = 0
        W[i, :, :, :] += sp.roll(W[0,:,:,:],i,axis=1)
        dist_grid_arr[i, :, :, :] += sp.roll(dist_grid_arr[0,:,:,:],i,axis=1)

    for l in range(1,num_cols): # already did l = 0
        W[:, :, :, l] += sp.roll(W[:,:,:,0],l,axis=1)
        dist_grid_arr[:, :, :, l] += sp.roll(dist_grid_arr[:,:,:,0],l,axis=1)
    return W, dist_grid_arr

"""
Calculates sum of weights.
"""
def calc_N(W):
    # Full matrix N with indices (i,j)
    N = sp.sum(W, axis=(2,3))
    return N

"""
Phi matrix
"""
# def calc_phi(gamma, unit_vector_e, phi_0, r):
#     # r_times_cos_of_angle_between = (r*unit_vector_e[0])[:,:,0]+(r*unit_vector_e[1])[:,:,1]
#     r_dot_e = (r*unit_vector_e[0])[:,:,0]+(r*unit_vector_e[1])[:,:,1]
#     phi = sp.multiply(gamma , r_dot_e) + phi_0
#     return phi

"""
Theta matrix evaluated at specific time
"""
# def calc_theta(Omega, t, phi):
#     # phi is (num_rows, num_cols) dimensional.
#     theta = sp.multiply(Omega,t)+phi
#     return theta

"""
Accepts (i,j) positions and evaluates (then returns) d(theta)/dt matrix (elements are (i,j) component)
When coding, it will help to treat all (i,j) matrices as if they are one element 
"""
# def d_theta_d_t(theta, t, phi, w, K, N, v, Omega, num_rows, num_cols, W_matrix, dist_grid_arr):
#
#     print("Simulating for t="+str(t))
#     # restore theta from flattened (num_rows*num_cols) to initial shape (num_rows, num_cols) for matrix algebra purposes
#     theta_ij_matrix_at_t = sp.array(theta).reshape((num_rows,num_cols))
#     # Get theta_ij_matrix_at_t into form (i,j,k,l) by broadcasting (1 length in k and l indices).
#     # This is an efficient way of making sure theta is the same for all k and l (value changes only based on i and j):
#     theta_ij_matrix_at_t_extend = theta_ij_matrix_at_t[:,:,None,None] # (i,j,k,l)
#     # (i,j,k,l)
#     phi_ij_extend = phi[:,:,None,None] # (i,j,k,l)
#     phi_kl_extend = phi_ij_extend.transpose((2,3,0,1)) # name the axes (i,j,k,l)
#     # Reminder that dist_grid_arr has shape (i, j, k, l)
#     theta_kl_delayed = sp.multiply(Omega,(t-sp.divide(dist_grid_arr,v)))+phi_kl_extend # (i,j,k,l)
#     sum_term = sp.einsum('ijkl,ijkl->ij',W_matrix, sp.sin(theta_kl_delayed - theta_ij_matrix_at_t_extend)) # (i,j)
#     dthetadt_final = w + sp.multiply(sp.divide(float(K),N) , sum_term) # (array with indices (i,j))
#     return dthetadt_final


"""
System of ODEs to be solved
"""
# def theODEs(theta, t, phi, w, K, N, v, Omega, num_rows, num_cols, W_matrix, dist_grid_arr):
#     dthetadt = d_theta_d_t(sp.mod(theta,2*sp.pi), t, phi, w, K, N, v, Omega, num_rows, num_cols, W_matrix, dist_grid_arr)
#     return dthetadt.reshape(num_cols*num_rows)