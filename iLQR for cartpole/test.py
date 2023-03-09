import gym

from cartpole_controller import LocalLinearizationController
from finite_difference_method import *
from lqr import *


def rand_pd_matrix(n):
    A = np.random.rand(n, n)
    return np.dot(A, A.T)


def test_finite_difference():
    Q = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float64)
    q = np.array([10, 11, 12], dtype=np.float64)
    x_s = np.array([0.1, 0.2, 0.3], dtype=np.float64)

    def f1(x): return Q @ (x - x_s) + np.ones(3)

    def f2(x):
        return (x - x_s) @ Q @ (x - x_s) + q @ (x - x_s) + 1

    # Gradient Tests

    assert np.allclose(gradient(f2, x_s.copy()), q)

    # Jacobian Tests
    
    assert np.allclose(jacobian(f1, x_s), Q)

    # Hessian Tests
    # print(Q + Q.T)
    assert np.allclose(hessian(f2, x_s), Q + Q.T)


def test_compute_policy():
    print("Testing Compute Policy...")
    A = np.array([[1., 0.02, 0., 0.], [0., 1., -0.01434146, 0.], [0., 0., 1., 0.02], [0., 0., 0.3155122, 1.]])
    B = np.array([[0.], [0.0195122], [0.], [-0.02926829]])
    m = np.array([[0.], [0.], [0.], [0.]])
    C = np.array([[575.55653791, 312.02835942, 995.6067982, 258.142569],
                  [312.02835942, 257.86675784, 887.96357121, 229.49340235],
                  [995.6067982, 887.96357121, 3744.74678006, 928.77687649],
                  [258.142569, 229.49340235, 928.77687649, 239.47387544]])
    D = np.array([[0.13241779]])
    E = np.array([[-2.30117315], [-2.6490898], [-16.41089798], [-4.2545946]])
    f = np.array([[0.], [0.], [0.], [0.]])
    g = np.array([[0.]])
    h = np.array([0.])
    K = np.array([[8.68906344, 10.00277156, 61.96636439, 16.06504165]])
    k = np.array([-0.])

    Kout, kout = compute_policy(A, B, m, C, D, E, f, g, h)
    assert np.allclose(Kout, K)
    assert np.allclose(kout, k)

    A = np.array([[1., .02, 0., 0.], [0., 1., -0.00873472, -0.00114593],
                  [0., 0., 1., 0.02], [0., 0., 0.30070484, 1.00164212]])
    B = np.array([[0.], [0.0193883], [0.], [-0.02778353]])
    m = np.array([[2.58593147e-12], [-2.54870172e-03], [8.50208792e-13], [4.07885872e-03]])
    C = np.array([[576.58162709, 313.21303493, 1034.03217331, 271.15499211],
                  [313.21303493, 258.58886297, 921.32315348, 240.96143033],
                  [1034.03213331, 921.32300574, 3985.00027296, 1004.64995445],
                  [271.15498168, 240.96139182, 1004.64995647, 262.90840939]])
    D = np.array([[0.13195463]])
    E = np.array([[-2.29742938], [-2.64931525], [-16.79490653], [-4.43093705]])
    f = np.array([[3.35892857], [2.15230041], [12.17786295], [4.34810397]])
    g = np.array([[-0.05937225]])
    h = np.array([1.13851287])
    K = np.array([[8.70537595, 10.03873523, 63.63894219, 16.78962286]])
    k = np.array([0.2249722])

    Kout, kout = compute_policy(A, B, m, C, D, E, f, g, h)
    assert np.allclose(Kout, K)
    assert np.allclose(kout, k)

    print('Test Compute Policy Successfully!\n')


def test_compute_Q_params():
    print("Test Compute Q Params...")
    all_data = {
        1: {
            "n_s": 3,
            "n_a": 5,
            "C": [[1.86509963, 2.46735175, 0.98151751],
                  [2.46735175, 3.39048201, 1.46508863],
                  [0.98151751, 1.46508863, 1.16780443]],
            "D": [[3.70345222, 3.29591738, 3.37993428, 2.13183605, 3.27928912],
                  [3.29591738, 5.77824626, 6.01660597, 3.16621757, 5.41834974],
                  [3.37993428, 6.01660597, 6.87569569, 3.49237589, 5.97921483],
                  [2.13183605, 3.16621757, 3.49237589, 2.07309201, 3.24103175],
                  [3.27928912, 5.41834974, 5.97921483, 3.24103175, 5.75876595]],
            "E": [[2.44698614, 4.95923297, 5.40589472, 2.32975286, 4.91948032],
                  [3.6758361, 6.68421483, 8.23234657, 3.85638945, 7.35983873],
                  [1.20639399, 2.42938586, 3.46944514, 1.62808741, 2.48490153]],
            "f": [[5.4897261],
                  [7.00930217],
                  [3.34661883]],
            "g": [[5.06744287],
                  [10.28950867],
                  [11.60855887],
                  [5.73485108],
                  [9.87607626]],
            "h": [5.55503602]
        },
        2: {
            "n_s": 4,
            "n_a": 4,
            "C": [[3.8200755, 3.60719867, 4.08455495, 5.5019658],
                  [3.60719867, 4.13020928, 3.9883352, 5.60200102],
                  [4.08455495, 3.9883352, 4.63927805, 6.20595991],
                  [5.5019658, 5.60200102, 6.20595991, 9.74128904]],
            "D": [[4.86154953, 3.91279569, 5.54873046, 1.83141928],
                  [3.91279569, 5.08345454, 5.0526337, 1.52609573],
                  [5.54873046, 5.0526337, 6.73812631, 2.14759014],
                  [1.83141928, 1.52609573, 2.14759014, 0.83362595]],
            "E": [[6.92979364, 5.76476509, 7.40212395, 2.10973581],
                  [5.74316138, 5.34368912, 7.2214022, 2.41132396],
                  [7.3346873, 6.4354931, 8.90245605, 2.38891491],
                  [11.69142368, 10.01524992, 13.57919998, 3.75394727]],
            "f": [[8.34985883],
                  [7.72212252],
                  [9.8734455],
                  [15.29872877]],
            "g": [[10.82501261],
                  [9.28890487],
                  [11.82440483],
                  [3.05679771]],
            "h": [7.89630723]
        },
        3: {
            "n_s": 5,
            "n_a": 3,
            "C": [[9.65933602, 7.45774817, 7.78943428, 5.16067427, 10.56498348],
                  [7.45774817, 6.44121548, 6.57902625, 4.03440182, 8.73037125],
                  [7.78943428, 6.57902625, 7.68419875, 4.74935718, 9.12981223],
                  [5.16067427, 4.03440182, 4.74935718, 3.17741297, 5.82318891],
                  [10.56498348, 8.73037125, 9.12981223, 5.82318891, 13.06944097]],
            "D": [[11.66089488, 10.38138664, 10.60025769],
                  [10.38138664, 9.72434538, 9.8785213],
                  [10.60025769, 9.8785213, 10.5124801]],
            "E": [[18.08622261, 16.8973519, 18.58286012],
                  [15.65786759, 15.52417323, 16.25289446],
                  [15.35034003, 14.5988045, 15.33294946],
                  [10.04009522, 9.34694594, 9.86016284],
                  [22.46248241, 21.4786098, 22.85529556]],
            "f": [[14.19004082],
                  [12.21864055],
                  [11.82745666],
                  [7.14626187],
                  [16.78751498]],
            "g": [[15.32133184],
                  [14.06823418],
                  [16.06171689]],
            "h": [7.259854]
        }
    }

    for seed, data in all_data.items():
        # Use the key of the map (1, 2, 3) as the seed to generate matrices
        np.random.seed(seed)
        n_s, n_a = data['n_s'], data['n_a']

        A = np.random.rand(n_s, n_s)
        B = np.random.rand(n_s, n_a)
        m = np.random.rand(n_s, 1)
        Q = rand_pd_matrix(n_s)
        R = rand_pd_matrix(n_a)
        M = np.random.rand(n_s, n_a)
        q = np.random.rand(n_s, 1)
        r = np.random.rand(n_a, 1)
        b = np.random.rand(1)
        P = rand_pd_matrix(n_s)
        y = np.random.rand(n_s, 1)
        p = np.random.rand(1)

        C, D, E, f, g, h = compute_Q_params(A, B, m, Q, R, M, q, r, b, P, y, p)

        assert np.allclose(C, data['C'])
        assert np.allclose(D, data['D'])
        assert np.allclose(E, data['E'])
        assert np.allclose(f, data['f'])
        assert np.allclose(g, data['g'])
        assert np.allclose(h, data['h'])

    print('Test Compute Q Params Successfully!\n')

def test_compute_V_params():
    print("Test Compute V Params...")
    all_data = {
        1: {
            "n_s": 3,
            "n_a": 5,
            "P": [[16.30582914,  8.7705151,  14.10180752], 
                  [10.55265668,  5.87484624,  9.25468152], 
                  [14.24656006,  7.52481802, 13.51330598]],
            "y": [[27.0395914 ], [15.29967968], [24.01732247]],
            "p": [11.37308215]
        },
        2: {
            "n_s": 4,
            "n_a": 4,
            "P": [[ 6.04542824,  5.08607438,  8.03010968,  7.80303096],
                  [ 5.22889503,  5.67300279,  7.66175543,  7.4896922 ],
                  [ 7.06784508,  6.74871699, 10.51744485, 10.05401772],
                  [ 6.93189422,  6.71119989, 10.1205708,  10.44862051]],
            "y": [[ 9.50846806],  [ 8.62389051], [13.39205133], [13.59320259]],
            "p": [5.43465755]
        },
        3: {
            "n_s": 5,
            "n_a": 3,
            "P": [[5.1514923,  3.80911917, 4.25747437, 2.61608615, 2.41962335],
                  [4.76821926, 4.0253703,  4.04985891, 2.40053877, 2.01955345],
                  [4.8755044,  3.91444821, 4.45391518, 2.72337515, 2.58641461],
                  [4.00233719, 3.01690584, 3.74291081, 2.4178779, 2.1016789 ],
                  [3.64343692, 2.74925196, 3.46411278, 2.2648767,  2.36686415]],
            "y": [[3.41555383],  [3.34405936], [3.16593122], [2.16415378], [2.07039901]],
            "p": [1.36116248]
        },
        4: {
            "n_s": 4,
            "n_a": 6,
            "P": [[20.03662266, 19.11700919, 20.9792493,  15.78053114],
                  [19.763512,   19.45192309, 21.39947089, 15.76488044],
                  [20.96010043, 20.43964484, 23.0167965,  16.75689682],
                  [17.00401108, 16.28280645, 17.48501405, 13.93606037]],
            "y": [[42.16739314], [41.23101131], [45.57536773], [34.20707695]],
            "p": [23.38706856]
        },
        5: {
            "n_s": 7,
            "n_a": 6,
            "P": [[19.04226047, 18.58424285, 22.03467985, 20.3619423, 20.97048077,  7.96664526, 18.05007676],
                  [18.46840457, 19.44238595, 22.97046439, 20.68452592, 21.00390257,  8.55704801, 17.99843591],
                  [21.09394603, 21.70704556, 26.90906571, 23.27897397, 24.25257034,  9.59337823, 20.51437317],
                  [18.84520543, 19.34171965, 23.01056417, 21.36099126, 20.73510457,  8.79343899, 18.40851763],
                  [20.6408985,  20.67226685, 25.10766622, 21.95282945, 24.07132936,  8.95084123, 19.64168851],
                  [8.79227283,  9.451966,   11.63739723, 10.19622988, 10.26399786,  4.86408027, 8.61099812],
                  [18.01936005, 18.29182422, 21.80395583, 19.56306917, 19.96339745,  7.70517049, 18.22627516]],
            "y": [[22.44254923],  [22.68719858], [27.09213323], [23.81228565], [25.93865058],  [10.90636508], [22.24272718]],
            "p": [8.25320862]
        },
    }

    for seed, data in all_data.items():
        # Use the key of the map (1, 2, 3) as the seed to generate matrices
        np.random.seed(seed)
        n_s, n_a = data['n_s'], data['n_a']

        A = np.random.rand(n_s, n_s)
        B = np.random.rand(n_s, n_a)
        m = np.random.rand(n_s, 1)
        C = rand_pd_matrix(n_s)
        D = rand_pd_matrix(n_a)
        E = np.random.rand(n_s, n_a)
        f = np.random.rand(n_s, 1)
        g = np.random.rand(n_a, 1)
        h = np.random.rand(1)
        K = np.random.rand(n_a, n_s)
        k = np.random.rand(n_a, 1)

        P, y, p = compute_V_params(A, B, m, C, D, E, f, g, h, K, k)

        assert np.allclose(P, data['P']), "Test {} Failed for P".format(seed)
        assert np.allclose(y, data['y']), "Test {} Failed for y".format(seed)
        assert np.allclose(p, data['p']), "Test {} Failed for p".format(seed)

    print('Test Compute V Params Successfully!\n')



def test_lqr():
    print("Test LQR...")
    A = np.array([[1., 0.02, 0., 0.],
                  [0., 1., -0.01434146, 0.],
                  [0., 0., 1., 0.02],
                  [0., 0., 0.3155122, 1.]])
    B = np.array([[0.],
                  [0.0195122],
                  [0.],
                  [-0.02926829]])
    m = np.array([[0.],
                  [0.],
                  [0.],
                  [0.]])
    Q = np.array([[1.00000005e+01, 2.00000000e-01, 3.48977574e-18,
                   -1.84585420e-17],
                  [2.00000000e-01, 1.00400050e+00, -1.43414634e-02,
                   7.75421394e-16],
                  [3.48977574e-18, -1.43414634e-02, 1.00997541e+01,
                   5.15512195e-01],
                  [-1.84585420e-17, 7.75421394e-16, 5.15512195e-01,
                   1.00400050e+00]])
    R = np.array([[0.10123786]])
    M = np.array([[1.12740085e-17],
                  [3.90243902e-02],
                  [-1.90286734e-02],
                  [-5.85365854e-02]])
    q = np.array([[0.],
                  [0.],
                  [0.],
                  [0.]])
    r = np.array([[0.]])
    b = np.array([0])
    T = 10

    output = [(np.array([[-1.27542759, -1.45070775, 5.92134911, 2.57168376]]), np.array([-0.])),
              (np.array([[-1.07520239, -1.36877901, 5.00438754, 2.34687466]]), np.array([-0.])),
              (np.array([[-0.87774937, -1.27095189, 4.12137414, 2.11533011]]), np.array([-0.])),
              (np.array([[-0.68809393, -1.15741932, 3.28535234, 1.8767259]]), np.array([-0.])),
              (np.array([[-0.51148038, -1.02860758, 2.51066727, 1.63080034]]), np.array([-0.])),
              (np.array([[-0.35321216, -0.88522816, 1.81260216, 1.3773701]]), np.array([-0.])),
              (np.array([[-0.21845602, -0.72832714, 1.20688535, 1.11635292]]), np.array([-0.])),
              (np.array([[-0.11201571, -0.5593266, 0.70906867, 0.84779666]]), np.array([-0.])),
              (np.array([[-0.03808556, -0.3800517, 0.33379341, 0.57191194]]), np.array([-0.])),
              (np.array([[-5.56807931e-17, -1.92736150e-01, 9.39800271e-02, 2.89104225e-01]]), np.array([-0.]))]

    student_output = lqr(A, B, m, Q, R, M, q, r, b, T)
    for i in range(len(output)):
        print(f"Testing policy for iteration {i}")
        assert np.allclose(student_output[i][0], output[i][0])
        assert np.allclose(student_output[i][1], output[i][1])
    print("Test LQR Successfully!\n")


def run_policy(init_state, x_s, u_s, T=500, num_episodes=100):
    env = gym.make("env:CartPoleControlEnv-v0")
    controller = LocalLinearizationController(env)
    # policies = controller.compute_local_policy(x_s, u_s, T)
    policies = controller.compute_global_policy(x_s, T, N = 5)

    # For testing, we use a noisy environment which adds small Gaussian noise to
    # state transition. Your controller only need to consider the env without noise.
    env = gym.make("env:NoisyCartPoleControlEnv-v0")
    env.seed(0)  # Seed env, don't remove!
    total_cost = 0
    for _ in range(num_episodes):
        observation = env.reset(state=init_state)
        for (K, k) in policies:
            action = (K @ observation + k)
            observation, cost, done, info = env.step(action)
            
            total_cost += cost
            if done:
                break
        env.close()

    return total_cost / num_episodes


def test_policy():
    print("Running cartpole with different initializations....")
    init_states = [np.array([0.0, 0.0, 0.0, 0.0]),
                   np.array([0.0, 0.0, 0.2, 0.0]),
                   np.array([0.0, 0.0, 0.4, 0.0]),
                   np.array([0.0, 0.0, 0.6, 0.0]),
                   np.array([0.0, 0.0, 0.8, 0.0]),
                   np.array([0.0, 0.0, 1.0, 0.0]),
                   np.array([0.0, 0.0, 1.2, 0.0]),
                   np.array([0.0, 0.0, 1.4, 0.0])]
    x_s = np.array([0, 0, 0, 0], dtype=np.float64)
    u_s = np.array([0], dtype=np.float64)
    for i, s in enumerate(init_states):
        print("case {} average cost:".format(i), run_policy(s, x_s, u_s))


if __name__ == "__main__":
    test_finite_difference()
    test_compute_Q_params()
    test_compute_policy()
    test_compute_V_params()
    test_lqr()
    test_policy()
