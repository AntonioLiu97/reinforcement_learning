import numpy as np

def derivative(f, x, delta=1e-5):
    return (f(x+delta)-f(x-delta))/(2*delta)

def gradient(f, x, delta=1e-5):
    """
    Returns the gradient of function f at the point x
    Parameters:
        f (numpy.array -> double): A scalar function accepts numpy array x
        x (numpy.array): A numpy array which is the same form as the argument supplied to f
        delta (double): delta used in the finite difference method


    Returns:
        ret (numpy.array): gradient of f at the point x
    """
    n = len(x)
    grad = np.zeros(n).astype('float64')
    for i in range(n):
        delta_array = np.zeros(n) 
        delta_array[i] = delta
        g1 = (f(x+delta_array)-f(x-delta_array))/(2*delta)
        grad[i] = g1
        
    return grad.astype('float64')
    


def jacobian(f, x, delta=1e-5):
    """
    Returns the Jacobian of function f at the point x
    Parameters:
        f (numpy.array -> numpy.array): A function accepts numpy array x
        x (numpy.array): A numpy array which is the same form as the argument supplied to f
        delta (double): delta used in the finite difference method

    Returns:
        ret (numpy.array): A 2D numpy array with shape (f(x).shape[0], x.shape[0])
                            which is the jacobian of f at the point x
    """
    n = len(x)
    m = len(f(x))
    jac = np.zeros((n,m)).astype('float64')
    for i in range(n):
        delta_array = np.zeros(n) 
        delta_array[i] = delta
        g1 = (f(x+delta_array)-f(x-delta_array))/(2*delta)
        jac[i] = g1
        
    return jac.T.astype('float64')


def hessian(f, x, delta=1e-5):
    """
    Returns the Hessian of function f at the point x
    Parameters:
        f (numpy.array -> double): A scalar function accepts numpy array x
        x (numpy.array): A numpy array which is the same form as the argument supplied to f
        delta (double): delta used in the finite difference method

    Returns:
        ret (numpy.array): A 2D numpy array with shape (x.shape[0], x.shape[0])
                            which is the Hessian of f at the point x
    """
    ### hessian is jacobian of the gradient
    def ff(x):
        return gradient(f, x, delta)
    hess = jacobian(ff, x, delta)
    return hess
    

