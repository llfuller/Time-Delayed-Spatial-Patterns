import scipy as sp

"""
Calculates distance between two points (k,l) and (i, j) (named r_kl_ij).
"""
def calc_r(k,l,i,j):
    # some content
    r_klij = sp.sqrt((i-k)**2 + (j-l)**2)
    return r_klij

"""
Calculates W(r_klij).
"""
def W(r_klij):
    W_klij = 1.0/r_klij
    return W_klij

"""
Calculates a full (i,j,k,l) matrix of W which is identical except for
zeros where r_klij > r_0
"""
def calc_W_limited(W, r_0, num_rows, num_cols):
    W_limited = sp.zeros((num_rows,num_cols,num_rows,num_cols))
    for i in range(num_rows):
        for j in range(num_cols):
            for k in range(num_rows):
                for l in range(num_cols):
                    if calc_r(k,l,i,j) > r_0:
                        W_limited[i,j,k,l] = 0
                    else:
                        W_limited[i,j,k,l] = W[i,j,k,l]
    return W_limited


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
def phi(gamma, unit_vector_e, phi_0, num_rows, num_cols):
    r = sp.array([[(i,j) for j in range(num_cols)] for i in range(num_rows)])
    cos_of_angle_between = sp.divide(r[0]*unit_vector_e[0]+r[1]*unit_vector_e[1], sp.sqrt(r[0]**2 + r[1]**2))
    # 1 x 50x50x2 + 50x50 = 50x2 + 50x50
    phi = sp.multiply(gamma , cos_of_angle_between) + phi_0
    return phi

"""
Theta matrix evaluated at specific time  
"""
def theta(Omega, t, phi):
    # phi is (num_rows, num_cols) dimensional.
    theta = Omega*t+phi
    return theta

"""
Accepts (i,j) positions and evaluates (then returns) d(theta)/dt matrix (elements are (i,j) component)
"""
def d_theta_d_t(w, K, N, t, v, Omega, num_rows, num_cols):
    sum_sub_term = sp.zeros((num_rows, num_cols, num_rows, num_cols))# array with indices (i,j,k,l); W*sin() part of sum
    sum_term = sp.zeros((num_rows,num_cols))
    for i in range(num_rows):
        for j in range(num_cols):
            for k in range(num_rows):
                for l in range(num_cols):
                    theta_kl_delayed = theta(Omega, t-float(calc_r(k,l,i,j))/v, phi[k,l])
                    theta_ij_at_t = theta(Omega, t, phi[i,j])
                    sum_sub_term[i,j,k,l] = sp.multiply(W(calc_r(i,j,k,l)), sp.sin(theta_kl_delayed -  theta_ij_at_t))
            sum_term[i,j] = sp.sum(sum_sub_term, axes=(2,3))
    dthetadt_final = w + sp.multiply(sp.divide(float(K),N) , sum_term) # (array with indices (i,j))
    return dthetadt_final


"""
System of ODEs to be solved
"""
def theODEs(theta, t, w, K, N, v, Omega, num_rows, num_cols):
    # t should be a time here, I think
    dthetadt = [d_theta_d_t(w, K, N, t, v, Omega, num_rows, num_cols)]
    return dthetadt