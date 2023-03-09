import numpy as np
from finite_difference_method import gradient, jacobian, hessian
from lqr import lqr, ilqr

class LocalLinearizationController:
    def __init__(self, env):
        """
        Parameters:
            env: an customized openai gym environment with reset function to reset 
                 the state to any state
        """
        self.env = env

    def c(self, s, a):
        """
        Cost function of the env.
        It sets the state of environment to `s` and then execute the action `a`, and
        then return the cost. 
        Parameter:
            s (1D numpy array): (4,) 
            a (1D numpy array): (1,)
        Returns:
            cost (double)
        """
        assert s.shape == (4,)
        # assert a.shape == (1,)
        env = self.env
        env.reset(state=s)
        observation, cost, done, info = env.step(np.ravel(a))
        return cost

    def f(self, s, a):
        """
        State transition function of the environment.
        Return the next state by executing action `a` at the state `s`
        Parameter:
            s (1D numpy array): (4,)
            a (1D numpy array): (1,)
        Returns:
            next_observation (1D numpy array): (4,)
        """
        # assert s.shape == (4,)
        # assert a.shape == (1,)
        env = self.env
        env.reset(state=s.flatten())
        
        # print(a.shape)
        next_observation, cost, done, info = env.step(np.ravel(a))
        return next_observation


    def compute_local_policy(self, s_star, a_star, T):
        """
        This function perform a first order taylar expansion function f and
        second order taylor expansion of cost function around (s_star, a_star). Then
        compute the optimal polices using lqr.
        outputs:
        Parameters:
            T (int) maximum number of steps
            s_star (numpy array) with shape (4,)
            a_star (numpy array) with shape (1,)
        return
            Ks(List of tuples (K_i,k_i)): A list [(K_0,k_0), (K_1, k_1),...,(K_T,k_T)] with length T
                                          Each K_i is 2D numpy array with shape (1,4) and k_i is 1D numpy
                                          array with shape (1,)
                                          such that the optimial policies at time are i is K_i * x_i + k_i
                                          where x_i is the state
        """
        # taylor expand around f and cost to obtain A, B, Q, R, M, q, r
        ## ff and cc: temporary functions to single out argument to expand in
        
        delta = 5e-5
        
        def ff(s): return self.f(s, a_star)
        A = jacobian(ff, s_star,delta)
        
        def ff(a): return self.f(s_star, a)
        B = jacobian(ff, a_star,delta)
        
        def cc(sa): return self.c(sa[0:4],sa[4:])
        sa_star = np.concatenate((s_star,a_star))
        H = hessian(cc, sa_star,delta)
        
        # print(f"H is positive definite: {np.all(np.linalg.eigvals(H) > 0)}")
        
        Q = H[0:4, 0:4]
        R = H[4:, 4:]
        M = H[0:4, 4:]
        
        G = gradient(cc, sa_star)
        q = G[0:4]
        r = G[4:]
        
        ### convert to Q, R, q, r, b, m in the LQR formulation
        Q = Q / 2
        R = R / 2
        q = (q.T - s_star.T @ Q - a_star.T @ M.T).T
        r = (r.T - a_star.T @ R - s_star.T @ M).T
        b = self.c(s_star,a_star) + s_star.T@Q@s_star/4 + a_star.T@R@a_star/2 + s_star.T@M@a_star - q.T@s_star - r.T@a_star
        m = (self.f(s_star,a_star) - A@s_star - B@a_star)

        return lqr(A, B, m.reshape(4,1), Q, R, M, q.reshape(4,1), r.reshape(1,1), b.reshape(1,), T)
    
    @staticmethod
    def nearest_PD(A, lam = 0.001):
        n = len(A)
        eigval, eigvec = np.linalg.eig(A)
        Q = np.matrix(eigvec)
        xdiag = np.matrix(np.diag(np.maximum(eigval, 0)))
        return Q*xdiag*Q.T + np.eye(n)*lam
    
    @staticmethod
    def policy_mixture(l1, l2, alpha=1/2):
        l3 = []
        for i in range(len(l1)):
            K_mixture = (1-alpha)*l1[i][0] + alpha*l2[i][0]
            k_mixture = (1-alpha)*l1[i][1] + alpha*l2[i][1]
            l3+=[(K_mixture,k_mixture)]
        return l3
    
    def are_lists_close(self, l1, l2, tolerance=0.01):
        """
        recursively check if all np elements in list (of lists) are close
        """
        if isinstance(l1, (list, tuple)) and isinstance(l2, (list, tuple)):
            if len(l1) != len(l2):
                return False
            for i in range(len(l1)):
                if not self.are_lists_close(l1[i], l2[i], tolerance):
                    return False
            return True
        else:
            return np.all(abs(l1 - l2) <= tolerance)

    def linearized_traject(self, s_ini, policy_list, delta = 1e-7):
        """
        inputs:
            s_ini: initial state
            a_array: array of action 
            policy_list: 
                list of policies, 
                if present, then actions will be obtain from policy_list
            
        """
        
        T = len(policy_list)

        ABmQRMqrb_list = []
        
        ### initialize s_star and a_star as initial conditions
        s_star = s_ini
        # print(s_star)
        # a_star = np.array([0]).reshape(1,)
        (K,k) = policy_list[0]
        a_star = np.ravel((K @ s_star + k))
        
        
        for i in range(1,T+1):
            # print(f"step {i}")
            
            ### linear and quadratic expansion around current trajectory     
            # print(a_star)
            def ff(s): return self.f(s, a_star.flatten())
            # print(s_star)
            A = jacobian(ff, s_star,delta)
            def ff(a): return self.f(s_star, a)
            B = jacobian(ff, a_star,delta)
            def cc(sa): return self.c(sa[0:4],sa[4:])
            sa_star = np.concatenate((s_star,np.ravel(a_star)))
            H = hessian(cc, sa_star,delta)
            

            if not np.all(np.linalg.eigvals(H) > 0):
                H = self.nearest_PD(H, lam = 0.001)
            
            assert np.all(np.linalg.eigvals(H) > 0), "H is not positive definite"
            
            a_star = np.ravel(a_star)
            Q = H[0:4, 0:4]
            R = H[4:, 4:]
            M = H[0:4, 4:]  
            G = gradient(cc, sa_star)
            q = G[0:4]
            r = G[4:]
            Q = Q / 2
            R = R / 2
            q = (q.T - s_star.T @ Q - a_star.T @ M.T).T
            r = (r.T - a_star.T @ R - s_star.T @ M).T
            b = self.c(s_star,a_star) + s_star.T@Q@s_star/4 + a_star.T@R@a_star/2 + s_star.T@M@a_star - q.T@s_star - r.T@a_star
            m = (self.f(s_star,a_star) - A@s_star - B@a_star)
            
            # print(m.shape)
            lin_expansion = [A, B, m.reshape(4,1), Q, R, M, q.reshape(4,1), r.reshape(1,1), b.flatten()]
            ABmQRMqrb_list += [lin_expansion]
            
            if i < T:
                ## set next action
                (K,k) = policy_list[i]
                a_star = (K @ s_star + k).reshape(1,)
                    
                # compute next state 
                s_star = self.f(s_star, a_star)   
                # print(s_star)

        return(ABmQRMqrb_list)
        
    def compute_global_policy(self, s_ini, T, N):
        """
        This function perform a first order taylar expansion function f and
        second order taylor expansion of cost function around (s_star, a_star). Then
        compute the optimal polices using iterative lqr.
        outputs:
        Parameters:
            T (int) maximum number of steps
            N (int) number of iterations in iLQR
            s_star (numpy array) with shape (4,)
            a_star (numpy array) with shape (1,)
        return
            Ks(List of tuples (K_i,k_i)): A list [(K_0,k_0), (K_1, k_1),...,(K_T,k_T)] with length T
                                          Each K_i is 2D numpy array with shape (1,4) and k_i is 1D numpy
                                          array with shape (1,)
                                          such that the optimial policies at time are i is K_i * x_i + k_i
                                          where x_i is the state
        """
        
        delta = 5e-7
        
        ### initial state
        a_star = np.array([0], dtype=np.double)
    
        ### initial policy
        K_0 = np.zeros((1,4)).astype(np.double)
        k_0 = np.array([0])
        policies = [(K_0,k_0)]*T
        
        for n in range(N):
            print(f"iteration {n}")
            # compute new trajectory following policies, and linearize the dynamics and costs around it
            # in this problem, s_ini is fixed
            ABmQRMqrb_list = self.linearized_traject(s_ini, policy_list=policies, delta = delta)
        
            # a linear mixture of pi_t and pi_(t+1)
            new_policies = ilqr(ABmQRMqrb_list)
            
            # break when policies have converged
            if self.are_lists_close(policies, new_policies, tolerance=1):
                break         
                
            policies = self.policy_mixture(policies, new_policies, alpha = 0.5)
            
            


        return policies

class PIDController:
    """
    Parameters:
        P, I, D: Controller gains
    """

    def __init__(self, P, I, D):
        """
        Parameters:
            env: an customized openai gym environment with reset function to reset
                 the state to any state
        """
        self.P, self.I, self.D = P, I, D
        self.err_sum = 0.
        self.err_prev = 0.

    def get_action(self, err):
        self.err_sum += err
        a = self.P * err + self.I * self.err_sum + self.D * (err - self.err_prev)
        self.err_prev = err
        return a



