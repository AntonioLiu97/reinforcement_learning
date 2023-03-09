import numpy as np

def compute_Q_params(A, B, m, Q, R, M, q, r, b, P, y, p):
    """
    Compute the Q function parameters for time step t.
    Let the shape of s_t be (n_s,), the shape of a_t be (n_a,)
        Parameters:
        A (2d numpy array): A numpy array with shape (n_s, n_s)
        B (2d numpy array): A numpy array with shape (n_s, n_a)
        m (2d numpy array): A numpy array with shape (n_s, 1)
        Q (2d numpy array): A numpy array with shape (n_s, n_s). Q is PD
        R (2d numpy array): A numpy array with shape (n_a, n_a). R is PD.
        M (2d numpy array): A numpy array with shape (n_s, n_a)
        q (2d numpy array): A numpy array with shape (n_s, 1)
        r (2d numpy array): A numpy array with shape (n_a, 1)
        b (1d numpy array): A numpy array with shape (1,)
        P (2d numpy array): A numpy array with shape (n_s, n_s). This is the quadratic term of the
            value function equation from time step t+1. P is PSD.
        y (2d numpy array): A numpy array with shape (n_s, 1).  This is the linear term
            of the value function equation from time step t+1
        p (1d numpy array): A numpy array with shape (1,).  This is the constant term of the
            value function equation from time step t+1
    Returns:
        C (2d numpy array): A numpy array with shape (n_s, n_s)
        D (2d numpy array): A numpy array with shape (n_s, n_a)
        E (2d numpy array): A numpy array with shape (n_s, n_a)
        f (2d numpy array): A numpy array with shape (n_s,1)
        g (2d numpy array): A numpy array with shape (n_a,1)
        h (1d numpy array): A numpy array with shape (1,)

        where the following equation should hold
        Q_t^*(s) = s^T C s + a^T D s + s^T E a + f^T s  + g^T a + h

    """
    n_s, n_a = B.shape
    # assert A.shape == (n_s, n_s)
    # assert B.shape == (n_s, n_a)
    # assert m.shape == (n_s, 1)
    # assert Q.shape == (n_s, n_s)
    # assert R.shape == (n_a, n_a)
    # assert M.shape == (n_s, n_a)
    # assert q.shape == (n_s, 1)
    # assert r.shape == (n_a, 1)
    # assert b.shape == (1, )
    # assert P.shape == (n_s, n_s)
    # assert y.shape == (n_s, 1)
    # assert p.shape == (1, )

    C = Q + A.T @ P @ A
    D = R + B.T @ P @ B
    E = M + 2 * A.T @ P @ B
    f = (q.T + 2 * m.T @ P @ A + y.T @ A).T
    g = (r.T + 2 * m.T @ P @ B + y.T @ B).T
    h = b + m.T @ P @ m + y.T @ m + p
    
    return C, D, E, f, g, h.flatten()


def compute_policy(A, B, m, C, D, E, f, g, h):
    """
    Compute the optimal policy at the current time step t
    Let the shape of s_t be (n_s,), the shape of a_t be (n_a,)
    Let Q_t^*(s) = s^T C s + a^T D a + s^T E a + f^T s  + g^T a  + h
    Parameters:

        D (2d numpy array): A numpy array with shape (n_a, n_a). D is PD.
        E (2d numpy array): A numpy array with shape (n_s, n_a)
        g (2d numpy array): A numpy array with shape (n_a, 1)
    Returns:
        K_t (2d numpy array): A numpy array with shape (n_a, n_s)
        k_t (2d numpy array): A numpy array with shape (n_a, 1)

        where the following holds
        \pi*_t(s) = K_t s + k_t
    """
    n_s, n_a = B.shape
    assert D.shape == (n_a, n_a)
    assert E.shape == (n_s, n_a)
    assert g.shape == (n_a, 1)


    K_t = - 1/2 * np.linalg.inv(D) @ E.T
    k_t = - 1/2 * np.linalg.inv(D) @ g
    # print(K_t, k_t)
    return K_t, k_t.flatten()


def compute_V_params(A, B, m, C, D, E, f, g, h, K, k):
    """
    Compute the V function parameters for the next time step
    Let the shape of s_t be (n_s,), the shape of a_t be (n_a,)
    Let V_t^*(s) = s^T P_t s + y_t^T s + p_t
    Parameters:
        A (2d numpy array): A numpy array with shape (n_s, n_s)
        B (2d numpy array): A numpy array with shape (n_s, n_a)
        m (2d numpy array): A numpy array with shape (n_s, 1)
        C (2d numpy array): A numpy array with shape (n_s, n_s). C is PD.
        D (2d numpy array): A numpy array with shape (n_a, n_a). D is PD.
        E (2d numpy array): A numpy array with shape (n_s, n_a)
        f (2d numpy array): A numpy array with shape (n_s, 1)
        g (2d numpy array): A numpy array with shape (n_a, 1)
        h (1d numpy array): A numpy array with shape (1, )
        K (2d numpy array): A numpy array with shape (n_a, n_s)
        k (2d numpy array): A numpy array with shape (n_a, 1)

    Returns:
        P_h (2d numpy array): A numpy array with shape (n_s, n_s)
        y_h (2d numpy array): A numpy array with shape (n_s, 1)
        p_h (1d numpy array): A numpy array with shape (1,)
    """
    #TODO
    n_s, n_a = B.shape
    # assert A.shape == (n_s, n_s)
    # assert B.shape == (n_s, n_a)
    # assert m.shape == (n_s, 1)
    # assert C.shape == (n_s, n_s)
    # assert D.shape == (n_a, n_a)
    # assert E.shape == (n_s, n_a)
    # assert f.shape == (n_s, 1)
    # assert g.shape == (n_a, 1)
    # assert h.shape == (1, )
    # assert K.shape == (n_a, n_s)
    
    P_t = C + K.T @ D @ K + E @ K
    y_t = (f.T + 2 * k.T @ D @ K + k.T @ E.T + g.T @ K).T
    p_t = k.T @ D @ k + g.T @ k + h

    return P_t, y_t, p_t.flatten()


def lqr(A, B, m, Q, R, M, q, r, b, T):
    """
    Compute optimal policies by solving
    argmin_{\pi_0,...\pi_{T-1}} \sum_{t=0}^{T-1} s_t^T Q s_t + a_t^T R a_t + s_t^T M a_t + q^T s_t + r^T a_t
    subject to s_{t+1} = A s_t + B a_t + m, a_t = \pi_t(s_t)

    Let the shape of s_t be (n_s,), the shape of a_t be (n_a,)
    Let optimal \pi*_t(s) = K_t s + k_t

    Parameters:
    A (2d numpy array): A numpy array with shape (n_s, n_s)
    B (2d numpy array): A numpy array with shape (n_s, n_a)
    m (2d numpy array): A numpy array with shape (n_s, 1)
    Q (2d numpy array): A numpy array with shape (n_s, n_s). Q is PD.
    R (2d numpy array): A numpy array with shape (n_a, n_a). R is PD.
    M (2d numpy array): A numpy array with shape (n_s, n_a)
    q (2d numpy array): A numpy array with shape (n_s, 1)
    r (2d numpy array): A numpy array with shape (n_a, 1)
    b (1d numpy array): A numpy array with shape (1,)
    T (int): The number of total steps in finite horizon settings

    Returns:
        ret (list): A list, [(K_0, k_0), (K_1, k_1), ..., (K_{T-1}, k_{T-1})]
        and the shape of K_t is (n_a, n_s), the shape of k_t is (n_a,)
    """
    n_s, n_a = B.shape
    
    ### set base case variables to get first Q for free
    C,D,E,f,g,h = Q,R,M,q,r,b
    policy_list = []
    for i in range(T):
        # compute and save policy
        K, k = compute_policy(A, B, m, C, D, E, f, g, h)
        policy_list += [(K, k)]
        
        # compute optimal V
        P, y, p = compute_V_params(A, B, m, C, D, E, f, g, h, K, k)
        
        # compute next Q
        C, D, E, f, g, h = compute_Q_params(A, B, m, Q, R, M, q, r, b, P, y, p)
    
    return policy_list[::-1]

def ilqr(ABmQRMqrb_list):
    """
        input: 
            ABmQRMqrb_list: array of linearized dynamics and costs, in the order of increasing time
            shape: [[A0, B0, m0, Q0, R0, M0, q0, r0, b0],
                    [A1, B1, m1, Q1, R1, M1, q1, r1, b1],
                    ...
                    ]
            len(ABmQRMqrb_list) = T
    """
    ### set base case variables to get first Q for free
    # print(len(ABmQRMqrb_list[0]))
    T = len(ABmQRMqrb_list)
    A,B,m,Q,R,M,q,r,b = ABmQRMqrb_list[-1]
    C,D,E,f,g,h = Q,R,M,q,r,b
    policy_list = []
    
    # compute and save first policy
    K, k = compute_policy(A, B, m, C, D, E, f, g, h)
    policy_list += [(K, k)]
    
    for i in range(2,T+1):

        # compute optimal V_t+1
        P, y, p = compute_V_params(A, B, m, C, D, E, f, g, h, K, k)
        
        # load next linearized dynamics
        # handout p4
        A,B,m,Q,R,M,q,r,b = ABmQRMqrb_list[-i]
        # C,D,E,f,g,h = Q,R,M,q,r,b
        
        # compute Q_t
        C, D, E, f, g, h = compute_Q_params(A, B, m, Q, R, M, q, r, b, P, y, p)
        
        # extract and save policy
        K, k = compute_policy(A, B, m, C, D, E, f, g, h)
        policy_list += [(K, k)]        
    
    ### return policies in increasing time
    return policy_list[::-1]