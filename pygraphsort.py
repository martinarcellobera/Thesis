import numpy as np
import scipy.linalg
from munkres import Munkres

#############################################################################################################
#
# This is James Nichols' code. He co-supervised my thesis. This code is required to run some notebooks.
#
#############################################################################################################

error_tol = 1e-4
max_sinkhorn_iter = 10000

#############################################################################################################
#
# First we start with an implementation of Sinkhorn's method, which is the bread and butter of our algorithm
#
#############################################################################################################

def sinkhorns_method(a, b, K, num_iter=max_sinkhorn_iter):
    
    # TODO: Implement the fast cost-matrix multiplication using FFT as it's a 
    # Toeplitz matrix (use Circulant embedding)

    v = np.ones(len(b))

    u = a / (K @ v)
    r = v * (K.T @ u)
    
    v = b / (K.T @ u)
    s = u * (K @ v)
    
    err_a = np.zeros(num_iter)
    err_b = np.zeros(num_iter)
    
    err_a[0] = np.linalg.norm(s - a, 1)
    err_b[0] = np.linalg.norm(r - b, 1)

    i = 1
    while (err_a[i-1] > error_tol and err_b[i-1] > error_tol) and (i < num_iter):

        # sinkhorn step 1
        u = a / (K @ v)
        r = v * (K.T @ u)
        err_b[i] = np.linalg.norm(r - b, 1)

        # sinkhorn step 2
        v = b / (K.T @ u)
        s = u * (K @ v)
        err_a[i] = np.linalg.norm(s - a, 1)
        
        i += 1
    
    return u, v, err_a[:i], err_b[:i]

def log_sinkhorns_method(a_log, b_log, K_log, num_iter=max_sinkhorn_iter):
    
    # TODO: Implement a log-sum-exp version of Sinkhorn's method in order to avoid numerical overflow
    pass

def cost(G, C, P=None):

    # TODO: Implement the fast cost-matrix multiplication using FFT as it's a 
    # Toeplitz matrix (use Circulant embedding)

    if P is None:
        P = np.eye(G.shape[0])
        return (G * C).sum()
    else:
        return ((P @ G @ P.T) * C).sum()
    
def entropy(P):
    # NOTE: Wherever P_ij = 0, it contributes 0 towards entropy

    with np.errstate(divide = 'ignore', invalid='ignore'):
        ent = - P * (np.log(P) - 1)
    
    ent[np.isnan(ent)] = 0
    
    return ent.sum()

def entropic_cost(G, C, epsilon, P=None):
    if P is None:
        return cost(G, C) + epsilon * entropy(np.eye(G.shape[0]))
    else:
        return cost(G, C, P) - epsilon * entropy(P)

def make_C(n, p):
    c = np.arange(n) ** p
    
    return scipy.linalg.toeplitz(c/c.max())

############################################################################################################
#
# Below are three minor variants of the same alternating minimization / Frank-Wolfe + Sinkhorn method 
# to find the optimal arrangement of the graph. The version that is supported by the theory is the
# ent_fw_ls_graph_sort() routine, meaning "Entropic Frank-Wolfe with line-search" algorithm.
#
############################################################################################################

def ent_graph_sort(G, C, P_0=None, epsilon=1e-1, eta=1., n_iter=100):
    
    n = G.shape[0]
    if P_0 is not None:
        P_i = P_0 
    else:
        P_i = np.eye(n)

    quadratic_energy = np.zeros(n_iter)
    entropic_energy = np.zeros(n_iter)

    for i in range(n_iter):
        K_P = np.exp(- (C @ P_i @ G.T) / epsilon)
            
        C_i = K_P**eta * P_i**(1-eta)
        # We call on Sinkhorns' method which iterates row and column multipliers on C_i 
        # to make it as close to a bistochastic matrix as possible (i.e. fixed marginals of 1/n)
        u, v, err_a, err_b = sinkhorns_method(np.ones(n), np.ones(n), C_i)
        
        P_i = np.diag(u) @ C_i @ np.diag(v)

        quadratic_energy[i] = cost(G, C, P_i)
        entropic_energy[i] = entropic_cost(G, C, epsilon, P_i)
    
    return P_i, quadratic_energy, entropic_energy


def ent_fw_graph_sort(G, C, P_0=None, epsilon=1e-1, eta=1., n_iter=100):
    
    n = G.shape[0]
    if P_0 is not None:
        P_i = P_0 
    else:
        P_i = np.eye(n)
    P_last = P_i

    quadratic_energy = np.zeros(n_iter)
    entropic_energy = np.zeros(n_iter)

    for i in range(n_iter):
        eta = 2 / (i + 2)
        epsilon_i = epsilon / (1 + i)
        epsilon_i = epsilon
        K_P = np.exp(- (C @ (eta*P_i + (1-eta)*P_last) @ G.T) / epsilon_i)
            
        # We call on Sinkhorns' method which iterates row and column multipliers on C_i 
        # to make it as close to a bistochastic matrix as possible (i.e. fixed marginals of 1/n)
        u, v, err_a, err_b = sinkhorns_method(np.ones(n), np.ones(n), K_P)
        
        P_last = P_i
        P_i = np.diag(u) @ K_P @ np.diag(v)
        
        # What is the error that we'll report? One is the entropic and the quadratic energy
        quadratic_energy[i] = cost(G, C, P_i)
        entropic_energy[i] = entropic_cost(G, C, epsilon_i, P_i)
    
    return P_i, quadratic_energy, entropic_energy

def ent_fw_ls_graph_sort(G, C, P_0=None, epsilon=1e-1, n_iter=100):
    
    n = G.shape[0]
    if P_0 is not None:
        P_i = P_0 
    else:
        P_i = np.eye(n)
    Q_i = P_i

    quadratic_energy = np.zeros(n_iter)
    entropic_energy = np.zeros(n_iter)
    c_P_i = c_Q_i = cost(G, C, P_i)

    for i in range(n_iter):
        
        epsilon_i = epsilon
        K_P = np.exp(- (C @ P_i @ G.T) / epsilon_i)
            
        # We call on Sinkhorns' method which iterates row and column multipliers on C_i 
        # to make it as close to a bistochastic matrix as possible (i.e. fixed marginals of 1/n)
        u, v, err_a, err_b = sinkhorns_method(np.ones(n), np.ones(n), K_P)
        
        Q_i = np.diag(u) @ K_P @ np.diag(v)

        # What is the error that we'll report? One is the entropic and the quadratic energy
        c_P_i = cost(G, C, P_i)
        c_Q_i = cost(G, C, Q_i)
        c_cross = ((P_i @ G @ Q_i.T) * C).sum()

        eta = np.clip((c_P_i - c_cross) / (c_Q_i + c_P_i - c_cross), 0., 1.)
        P_i = (eta*Q_i + (1-eta)*P_i)

        quadratic_energy[i] = c_P_i
        entropic_energy[i] = entropic_cost(G, C, epsilon_i, P_i)
    
    return P_i, quadratic_energy, entropic_energy


def nearest_permutation_hungarian(P):
    # This returns the nearest permutation to the positive bistochastic matrix that comes from the optimisations
    # It is by no means an optimal 
    m = Munkres()
    indexes = m.compute(1. - P)

    P_munkres = np.zeros(P.shape)
    for row, column in indexes:
        P_munkres[row, column] = 1

    return P_munkres

def nearest_permutation(P, col_blur=True):
    if not col_blur:
        P = P.T

    maxs = np.argmax(P, axis=0)
    n = P.shape[1]

    P_near = np.zeros(P.shape)

    available = set(range(n))
    for i in range(n):
        if maxs[i] in available:
            available.remove(maxs[i])
            P_near[maxs[i], i] = 1
        else:
            avail_list = list(available)
            nearest = avail_list[np.argmax(P[avail_list, i])]
            available.remove(nearest)
            P_near[nearest, i] = 1

    if not col_blur:
        return P_near.T

    return P_near

# This is the MAIN ROUTINE that brings it all together

def sort_matrix(G, C=None, p=None, epsilon=1e-1, n_iter=100):
    if not G.shape[0] == G.shape[1]:
        print(f'Error: G must be a square matrix, not of size {G.shape}')

    n = G.shape[0]

    if not C:
        if not p:
            C = make_C(n, 2)
        else:
            C = make_C(n, p)
    
    P_ent, q_e, e_e = ent_fw_ls_graph_sort(G, C, epsilon=epsilon)

    P_exact = nearest_permutation(P_ent)

    return P_exact @ G @ P_exact.T, P_exact, P_ent


###############################################################################################
#
# Below are helper functions to generate random block matrices / graphs and random permutations
#
###############################################################################################

def make_random_block_graph(ns, intra_prob, inter_prob, inter_scale):
    n = ns.sum()
    ns_sums = np.concatenate(([0], ns)).cumsum()

    G = np.zeros((n, n))
    
    for k, n_l in enumerate(ns):
        for i in range(ns_sums[k], ns_sums[k+1]):
            for j in range(i+1,ns_sums[k+1]):
                if np.random.random() < intra_prob:
                    G[i,j] = G[j,i] = np.random.random()

            for j in range(ns_sums[k+1], n):
                if np.random.random() < inter_prob:
                    G[i,j] = G[j,i] = inter_scale * np.random.random()

    
    return G

def make_random_permutation(n):
    
    permutation = np.random.choice(range(n), n, replace=False)
    
    return np.eye(n)[permutation], permutation

def quadratic_optimality_test(G, C, P, epsilon):

    n = G.shape[0]
    P_Basis = np.zeros(((n-1)**2,n,n))

    count = 0

    for i in range(n-1):
        P_Basis[count, i,i] = P_Basis[count,i+1,i+1] = -1
        P_Basis[count, i,i+1] = P_Basis[count,i+1,i] = 1
        count += 1

        for j in range(i+2,n):
            P_Basis[count, i,i] = P_Basis[count,i+1,j] = -1
            P_Basis[count, i,j] = P_Basis[count,i+1,i] = 1
            count += 1

            P_Basis[count, i,i] = P_Basis[count,j,i+1] = -1
            P_Basis[count, i,i+1] = P_Basis[count,j,i] = 1
            count += 1

    
    P_Basis_flat = np.reshape(P_Basis, ((n-1)**2, -1))

    grammian = P_Basis_flat @ P_Basis_flat.T

    optimality_vec = P_Basis_flat.T @ P_Basis_flat @ C.flatten() + epsilon * P_Basis_flat.T @ P_Basis_flat @ np.log(P.flatten())

    return optimality_vec