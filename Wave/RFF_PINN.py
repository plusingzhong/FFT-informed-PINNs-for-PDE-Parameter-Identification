import tensorflow as tf
import numpy as np
import scipy.io
from Compute_Jacobian import jacobian
import timeit
from pyDOE import lhs
import matplotlib.pyplot as plt

np.random.seed(2345)
tf.set_random_seed(2345)

class Sampler:
    # Initialize the class
    def __init__(self, dim, coords, func, name=None):
        self.dim = dim
        self.coords = coords
        self.func = func
        self.name = name

    def sample(self, N):
        x = self.coords[0:1, :] + (self.coords[1:2, :] - self.coords[0:1, :]) * np.random.rand(N, self.dim)
        y = self.func(x)
        return x, y


class LHSSampler:
    def __init__(self, dim, coords, func=None, name=None):
        self.dim = dim
        self.coords = coords
        self.func = func
        self.name = name

    def sample(self, N):
        samples = lhs(self.dim, N)
        x = self.coords[0:1, :] + (self.coords[1:2, :] - self.coords[0:1, :]) * samples
        if self.func is not None:
            y = self.func(x)
        else:
            y = np.zeros((x.shape[0], 1))
        return x, y


class DataSampler:
    # Initialize the class for direct data sampling
    def __init__(self, X_data, Y_data, name=None):
        self.X_data = X_data
        self.Y_data = Y_data
        self.name = name
        self.dim = X_data.shape[1]
        self.total_points = X_data.shape[0]

    def sample(self, N):
        if N > self.total_points:
            idx = np.random.choice(self.total_points, N, replace=True)
        else:
            idx = np.random.choice(self.total_points, N, replace=False)
        return self.X_data[idx], self.Y_data[idx]


class Wave1D_NTK_ST_mFF:
    # Initialize the class
    def __init__(self, layers, operator, ics_sampler, bcs_sampler, res_sampler, data_sampler, kernel_size):
        # Normalization constants
        X, _ = res_sampler.sample(np.int32(1e5))
        self.mu_X, self.sigma_X = X.mean(0), X.std(0)
        self.mu_t, self.sigma_t = self.mu_X[0], self.sigma_X[0]
        self.mu_x, self.sigma_x = self.mu_X[1], self.sigma_X[1]
        self.operator = operator
        # Samplers
        self.ics_sampler = ics_sampler
        self.bcs_sampler = bcs_sampler
        self.res_sampler = res_sampler
        self.data_sampler = data_sampler

        # Initialize spatial and temporal Fourier features
        self.W1_t = tf.Variable(tf.random_normal([1, layers[0] // 2], dtype=tf.float32) * 1.0,
                               dtype=tf.float32, trainable=False)

        self.W2_t = tf.Variable(tf.random_normal([1, layers[0] // 2], dtype=tf.float32) * 10.0,
                               dtype=tf.float32, trainable=False)

        self.W1_x = tf.Variable(tf.random_normal([1, layers[0] // 2], dtype=tf.float32) * 1.0,
                               dtype=tf.float32, trainable=False)

        # Initialize network weights and biases
        self.layers = layers
        self.weights, self.biases = self.initialize_NN(layers)

        # weights
        self.lambda_u_val = np.array(1.0)
        self.lambda_ut_val = np.array(1.0)
        self.lambda_r_val = np.array(1.0)
        self.lambda_data_val = np.array(1.0)

        # Wave velocity constant
        self.c = tf.Variable([15.0], dtype=tf.float32)
        self.b = tf.Variable([0.0], dtype=tf.float32)
        # Size of NTK
        self.kernel_size = kernel_size

        D1 = self.kernel_size  # size of K_u
        D2 = self.kernel_size  # size of K_ut
        D3 = self.kernel_size  # size of K_r

        # Define Tensorflow session
        self.sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))

        # Define placeholders and computational graph
        self.t_u_tf = tf.placeholder(tf.float32, shape=(None, 1))
        self.x_u_tf = tf.placeholder(tf.float32, shape=(None, 1))
        self.u_u_tf = tf.placeholder(tf.float32, shape=(None, 1))

        self.t_ics_tf = tf.placeholder(tf.float32, shape=(None, 1))
        self.x_ics_tf = tf.placeholder(tf.float32, shape=(None, 1))
        self.u_ics_tf = tf.placeholder(tf.float32, shape=(None, 1))

        self.t_bc1_tf = tf.placeholder(tf.float32, shape=(None, 1))
        self.x_bc1_tf = tf.placeholder(tf.float32, shape=(None, 1))

        self.t_bc2_tf = tf.placeholder(tf.float32, shape=(None, 1))
        self.x_bc2_tf = tf.placeholder(tf.float32, shape=(None, 1))

        self.t_r_tf = tf.placeholder(tf.float32, shape=(None, 1))
        self.x_r_tf = tf.placeholder(tf.float32, shape=(None, 1))

        self.lambda_u_tf = tf.placeholder(tf.float32, shape=self.lambda_u_val.shape)
        self.lambda_ut_tf = tf.placeholder(tf.float32, shape=self.lambda_ut_val.shape)
        self.lambda_r_tf = tf.placeholder(tf.float32, shape=self.lambda_r_val.shape)
        self.lambda_data_tf = tf.placeholder(tf.float32, shape=self.lambda_data_val.shape)

        self.t_u_ntk_tf = tf.placeholder(tf.float32, shape=(D1, 1))
        self.x_u_ntk_tf = tf.placeholder(tf.float32, shape=(D1, 1))

        self.t_ut_ntk_tf = tf.placeholder(tf.float32, shape=(D2, 1))
        self.x_ut_ntk_tf = tf.placeholder(tf.float32, shape=(D2, 1))

        self.t_r_ntk_tf = tf.placeholder(tf.float32, shape=(D3, 1))
        self.x_r_ntk_tf = tf.placeholder(tf.float32, shape=(D3, 1))

        # Evaluate predictions
        self.u_ics_pred = self.net_u(self.t_ics_tf, self.x_ics_tf)
        self.u_t_ics_pred = self.net_u_t(self.t_ics_tf, self.x_ics_tf)
        self.u_bc1_pred = self.net_u(self.t_bc1_tf, self.x_bc1_tf)
        self.u_bc2_pred = self.net_u(self.t_bc2_tf, self.x_bc2_tf)

        self.u_pred = self.net_u(self.t_u_tf, self.x_u_tf)
        self.r_pred = self.net_r(self.t_r_tf, self.x_r_tf)

        self.u_ntk_pred = self.net_u(self.t_u_ntk_tf, self.x_u_ntk_tf)
        self.ut_ntk_pred = self.net_u_t(self.t_ut_ntk_tf, self.x_ut_ntk_tf)
        self.r_ntk_pred = self.net_r(self.t_r_ntk_tf, self.x_r_ntk_tf)

        # Boundary loss and Initial loss
        self.loss_ics_u = tf.reduce_mean(tf.square(self.u_ics_tf - self.u_ics_pred))
        self.loss_ics_u_t = tf.reduce_mean(tf.square(self.u_t_ics_pred))
        self.loss_bc1 = tf.reduce_mean(tf.square(self.u_bc1_pred))
        self.loss_bc2 = tf.reduce_mean(tf.square(self.u_bc2_pred))
        self.loss_data = tf.reduce_mean(tf.square(self.u_u_tf - self.u_pred))

        self.loss_bcs = self.loss_ics_u + self.loss_bc1 + self.loss_bc2

        # Residual loss
        self.loss_res = tf.reduce_mean(tf.square(self.r_pred))

        # Total loss
        self.loss = self.lambda_r_tf * self.loss_res + self.lambda_u_tf * self.loss_bcs + self.lambda_ut_tf * self.loss_ics_u_t + self.lambda_data_tf * self.loss_data

        # Define optimizer with learning rate schedule
        self.global_step = tf.Variable(0, trainable=False)
        starter_learning_rate = 1e-3
        self.learning_rate = tf.train.exponential_decay(starter_learning_rate, self.global_step,
                                                        1000, 0.9, staircase=False)

        self.learning_rate_b = tf.train.exponential_decay(starter_learning_rate * 2.0, self.global_step,
                                                          1000, 0.9, staircase=False)

        var_list_main = [var for var in tf.trainable_variables() if var is not self.b]
        var_list_b = [self.b]

        optimizer_main = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        optimizer_b = tf.train.AdamOptimizer(learning_rate=self.learning_rate_b)

        train_op_main = optimizer_main.minimize(self.loss, global_step=self.global_step, var_list=var_list_main)
        train_op_b = optimizer_b.minimize(self.loss, var_list=var_list_b)

        self.train_op = tf.group(train_op_main, train_op_b)

        # Compute the Jacobian for weights and biases in each hidden layer
        self.J_u = self.compute_jacobian(self.u_ntk_pred)
        self.J_ut = self.compute_jacobian(self.ut_ntk_pred)
        self.J_r = self.compute_jacobian(self.r_ntk_pred)

        self.K_u = self.compute_ntk(self.J_u, D1, self.J_u, D1)
        self.K_ut = self.compute_ntk(self.J_ut, D2, self.J_ut, D2)
        self.K_r = self.compute_ntk(self.J_r, D3, self.J_r, D3)

        # Loss logger
        self.b_log = []
        self.c_log = []

        self.loss_bcs_log = []
        self.loss_data_log = []
        self.loss_ut_ics_log = []
        self.loss_res_log = []
        self.l2_error_log = []
        self.loss_log = []
        self.iter_log = []

        # NTK logger
        self.K_u_log = []
        self.K_ut_log = []
        self.K_r_log = []

        # weights logger
        self.lambda_u_log = []
        self.lambda_data_log = []
        self.lambda_ut_log = []
        self.lambda_r_log = []

        # Initialize Tensorflow variables
        init = tf.global_variables_initializer()
        self.sess.run(init)

        # Saver
        self.saver = tf.train.Saver()

    # Initialize network weights and biases using Xavier initialization
    def xavier_init(self, size):
        in_dim = size[0]
        out_dim = size[1]
        xavier_stddev = 1. / np.sqrt((in_dim + out_dim) / 2.)
        return tf.Variable(tf.random_normal([in_dim, out_dim], dtype=tf.float32) * xavier_stddev,
                           dtype=tf.float32)

    def initialize_NN(self, layers):
        weights = []
        biases = []

        num_layers = len(layers)
        for l in range(0, num_layers - 2):
            W = self.xavier_init(size=[layers[l], layers[l + 1]])
            b = tf.Variable(tf.random_normal([1, layers[l + 1]], dtype=tf.float32), dtype=tf.float32)
            weights.append(W)
            biases.append(b)

        W = self.xavier_init(size=[2 * layers[-2], layers[-1]])
        b = tf.Variable(tf.random_normal([1, layers[-1]], dtype=tf.float32), dtype=tf.float32)
        weights.append(W)
        biases.append(b)

        return weights, biases

    def forward_pass(self, H):
        num_layers = len(self.layers)

        t = H[:, 0:1]
        x = H[:, 1:2]

        # Temporal and spatial Fourier feature encodings
        H1_t = tf.concat([tf.sin(tf.matmul(t, self.W1_t)),
                         tf.cos(tf.matmul(t, self.W1_t))], 1)

        H2_t = tf.concat([tf.sin(tf.matmul(t, self.W2_t)),
                          tf.cos(tf.matmul(t, self.W2_t))], 1)

        H1_x = tf.concat([tf.sin(tf.matmul(x, self.W1_x)),
                          tf.cos(tf.matmul(x, self.W1_x))], 1)

        for l in range(0, num_layers - 2):
            W = self.weights[l]
            b = self.biases[l]

            H1_t = tf.tanh(tf.add(tf.matmul(H1_t, W), b))
            H2_t = tf.tanh(tf.add(tf.matmul(H2_t, W), b))
            H1_x = tf.tanh(tf.add(tf.matmul(H1_x, W), b))

        # Merge outputs
        H1 = tf.multiply(H1_t, H1_x)
        H2 = tf.multiply(H2_t, H1_x)
        H = tf.concat([H1, H2], 1)

        W = self.weights[-1]
        b = self.biases[-1]
        H = tf.add(tf.matmul(H, W), b)

        return H

    # Forward pass for u
    def net_u(self, t, x):
        u = self.forward_pass(tf.concat([t, x], 1))
        return u

    def net_u_t(self, t, x):
        u_t = tf.gradients(self.net_u(t, x), t)[0] / self.sigma_t
        return u_t

    # Forward pass for f
    def net_r(self, t, x):
        u = self.net_u(t, x)
        residual = self.operator(u, t, x,
                                 self.b,
                                 self.c,
                                 self.sigma_t,
                                 self.sigma_x)
        return residual

    # Compute Jacobian for each weights and biases in each layer and retrun a list
    def compute_jacobian(self, f):
        J_list = []
        L = len(self.weights)
        for i in range(L):
            J_w = jacobian(f, self.weights[i])
            J_list.append(J_w)

        for i in range(L):
            J_b = jacobian(f, self.biases[i])
            J_list.append(J_b)
        return J_list

    # Compute the empirical NTK
    def compute_ntk(self, J1_list, D1, J2_list, D2):

        N = len(J1_list)

        Ker = tf.zeros((D1, D2))
        for k in range(N):
            J1 = tf.reshape(J1_list[k], shape=(D1, -1))
            J2 = tf.reshape(J2_list[k], shape=(D2, -1))

            K = tf.matmul(J1, tf.transpose(J2))
            Ker = Ker + K
        return Ker

    def fetch_minibatch(self, sampler, N):
        X, Y = sampler.sample(N)
        X = (X - self.mu_X) / self.sigma_X
        return X, Y

    def train(self, nIter=10000, batch_size=200, log_NTK=False, update_weights=False):

        training_start_time = timeit.default_timer()
        start_time = timeit.default_timer()
        for it in range(nIter):
            X_ics_batch, u_ics_batch = self.fetch_minibatch(self.ics_sampler, batch_size // 4)
            X_bc1_batch, _ = self.fetch_minibatch(self.bcs_sampler[0], batch_size // 4)
            X_bc2_batch, _ = self.fetch_minibatch(self.bcs_sampler[1], batch_size // 4)
            X_data_batch, u_u_batch = self.fetch_minibatch(self.data_sampler, batch_size // 4)
            X_res_batch, _ = self.fetch_minibatch(self.res_sampler, batch_size)

            tf_dict = {self.t_ics_tf: X_ics_batch[:, 0:1], self.x_ics_tf: X_ics_batch[:, 1:2],
                       self.u_ics_tf: u_ics_batch,
                       self.t_u_tf: X_data_batch[:, 0:1], self.x_u_tf: X_data_batch[:, 1:2],
                       self.u_u_tf: u_u_batch,
                       self.t_bc1_tf: X_bc1_batch[:, 0:1], self.x_bc1_tf: X_bc1_batch[:, 1:2],
                       self.t_bc2_tf: X_bc2_batch[:, 0:1], self.x_bc2_tf: X_bc2_batch[:, 1:2],
                       self.t_r_tf: X_res_batch[:, 0:1], self.x_r_tf: X_res_batch[:, 1:2],
                       self.lambda_u_tf: self.lambda_u_val,
                       self.lambda_data_tf: self.lambda_data_val,
                       self.lambda_ut_tf: self.lambda_ut_val,
                       self.lambda_r_tf: self.lambda_r_val}

            self.sess.run(self.train_op, tf_dict)

            # print
            if it % 100 == 0:
                elapsed = timeit.default_timer() - start_time
                loss_value, loss_bcs_value, loss_ics_ut_value, loss_res_value, loss_data_value, b_value, c_value = self.sess.run(
                    [self.loss, self.loss_bcs, self.loss_ics_u_t, self.loss_res, self.loss_data, self.b, self.c],
                    tf_dict)

                # log
                self.loss_bcs_log.append(loss_bcs_value)
                self.loss_data_log.append(loss_data_value)
                self.loss_res_log.append(loss_res_value)
                self.loss_ut_ics_log.append(loss_ics_ut_value)
                self.loss_log.append(loss_value)
                self.b_log.append(b_value)
                self.c_log.append(c_value)
                self.iter_log.append(it)

                print(
                    'It: %d, Loss: %.3e, Loss_res: %.3e,  Loss_bcs: %.3e, Loss_ut_ics: %.3e, Loss_data: %.3e, Time: %.2f' %
                    (it, loss_value, loss_res_value, loss_bcs_value, loss_ics_ut_value, loss_data_value, elapsed))

                print('lambda_u: {}'.format(self.lambda_u_val))
                print('lambda_data: {}'.format(self.lambda_data_val))
                print('lambda_ut: {}'.format(self.lambda_ut_val))
                print('lambda_r: {}'.format(self.lambda_r_val))
                print('b: %.3e' % (b_value))
                print('c: %.3e' % (c_value))

                start_time = timeit.default_timer()

            # NTK
            if log_NTK:
                X_bc_batch = np.vstack([X_ics_batch, X_bc1_batch, X_bc2_batch, X_data_batch])
                X_ics_batch, u_ics_batch = self.fetch_minibatch(self.ics_sampler, kernel_size)
                if it % 100 == 0:
                    print("Compute NTK...")
                    tf_dict = {self.t_u_ntk_tf: X_bc_batch[:, 0:1], self.x_u_ntk_tf: X_bc_batch[:, 1:2],
                               self.t_ut_ntk_tf: X_ics_batch[:, 0:1], self.x_ut_ntk_tf: X_ics_batch[:, 1:2],
                               self.t_r_ntk_tf: X_res_batch[:, 0:1], self.x_r_ntk_tf: X_res_batch[:, 1:2]}
                    K_u_value, K_ut_value, K_r_value = self.sess.run([self.K_u, self.K_ut, self.K_r], tf_dict)

                    self.K_u_log.append(K_u_value)
                    self.K_ut_log.append(K_ut_value)
                    self.K_r_log.append(K_r_value)

                    if update_weights:
                        lambda_K_sum = np.trace(K_u_value) + np.trace(K_ut_value) + \
                                       np.trace(K_r_value)

                        self.lambda_u_val = lambda_K_sum / np.trace(K_u_value)
                        self.lambda_data_val = lambda_K_sum / np.trace(K_u_value)
                        self.lambda_ut_val = lambda_K_sum / np.trace(K_ut_value)
                        self.lambda_r_val = lambda_K_sum / np.trace(K_r_value)

                    self.lambda_u_log.append(self.lambda_u_val)
                    self.lambda_data_log.append(self.lambda_data_val)
                    self.lambda_ut_log.append(self.lambda_ut_val)
                    self.lambda_r_log.append(self.lambda_r_val)

        training_end_time = timeit.default_timer()
        total_training_time = training_end_time - training_start_time
        print("time = %.2f s" % (total_training_time))

    # Evaluates predictions at test points
    def predict_u(self, X_star):
        X_star = (X_star - self.mu_X) / self.sigma_X
        tf_dict = {self.t_u_tf: X_star[:, 0:1], self.x_u_tf: X_star[:, 1:2]}
        u_star = self.sess.run(self.u_pred, tf_dict)
        return u_star

    # Evaluates predictions at test points
    def predict_r(self, X_star):
        X_star = (X_star - self.mu_X) / self.sigma_X
        tf_dict = {self.t_r_tf: X_star[:, 0:1], self.x_r_tf: X_star[:, 1:2]}
        r_star = self.sess.run(self.r_pred, tf_dict)
        return r_star


if __name__ == '__main__':
    data = scipy.io.loadmat('D:\\FFT-PINN\\FFT-PINN\\Wave\\wave.mat')

    t_full = data['t'].flatten()
    x_full = data['x'].flatten()
    Exact_full = data['u']

    T_full, X_full = np.meshgrid(t_full, x_full)
    X_full = np.hstack((T_full.flatten()[:, None], X_full.flatten()[:, None]))
    u_full = Exact_full.flatten()[:, None]

    t_small = data['t_small'].flatten()
    x_small = data['x_small'].flatten()
    u_small = data['u_small']

    T, X = np.meshgrid(t_small, x_small)
    X_train = np.hstack((
        T.flatten()[:, None],
        X.flatten()[:, None]
    ))
    U_train = u_small.flatten()[:, None]  # u(t, x)

    total_points = X_train.shape[0]

    def f(x):
        N = x.shape[0]
        return np.zeros((N, 1))


    def operator(u, t, x, b, c, sigma_t=1.0, sigma_x=1.0):
        u_t = tf.gradients(u, t)[0] / sigma_t
        u_x = tf.gradients(u, x)[0] / sigma_x
        u_tt = tf.gradients(u_t, t)[0] / sigma_t
        u_xx = tf.gradients(u_x, x)[0] / sigma_x
        residual = u_tt + b * u_t - c ** 2 * u_xx
        return residual

    ics_coords = np.array([[0.0, 0.0],
                           [0.0, 1.0]])
    bc1_coords = np.array([[0.0, 0.0],
                           [1.0, 0.0]])
    bc2_coords = np.array([[0.0, 1.0],
                           [1.0, 1.0]])
    dom_coords = np.array([[0.0, 0.0],
                           [1.0, 1.0]])

    ics_mask = (X_train[:, 0] == 0.0)
    X_ics = X_train[ics_mask]
    U_ics = U_train[ics_mask]

    bc1_mask = (X_train[:, 1] == 0.0)
    X_bc1 = X_train[bc1_mask]
    U_bc1 = U_train[bc1_mask]

    bc2_mask = (X_train[:, 1] == 1.0)
    X_bc2 = X_train[bc2_mask]
    U_bc2 = U_train[bc2_mask]

    internal_mask = ~(ics_mask | bc1_mask | bc2_mask)
    X_internal = X_train[internal_mask]
    U_internal = U_train[internal_mask]
    noise_level = 0.05

    U_noisy = U_internal + noise_level * np.std(U_internal) * np.random.randn(*U_internal.shape)

    ics_sampler = DataSampler(X_ics, U_ics, name='Initial Condition')

    bc1_sampler = DataSampler(X_bc1, U_bc1, name='Boundary Condition 1')
    bc2_sampler = DataSampler(X_bc2, U_bc2, name='Boundary Condition 2')
    bcs_sampler = [bc1_sampler, bc2_sampler]

    data_sampler = DataSampler(X_internal, U_noisy, name='Data Condition')

    res_sampler = LHSSampler(2, dom_coords, lambda x: f(x), name='Forcing')

    layers = [96, 96, 96, 1]
    kernel_size = 40

    model = Wave1D_NTK_ST_mFF(layers,
                              operator,
                              ics_sampler,
                              bcs_sampler,
                              res_sampler,
                              data_sampler,
                              kernel_size)

    iterations = 40000
    model.train(nIter=iterations, batch_size=kernel_size, log_NTK=True, update_weights=True)

    b_value, c_value = model.sess.run([model.b, model.c])
    print('b: {:.5f}'.format(b_value[0]))
    print('c: {:.5f}'.format(c_value[0]))

    absolute_error_lambda_1 = np.abs(b_value - 2.0)
    absolute_error_lambda_2 = np.abs(c_value - 20.0)

    relative_error_lambda_1 = np.abs(b_value - 2.0) / 2.0 * 100
    relative_error_lambda_2 = np.abs(c_value - 20.0) / 20.0 * 100

    print('Absolute Error lambda_1: {:.5f}'.format(absolute_error_lambda_1[0]))
    print('Absolute Error lambda_2: {:.5f}'.format(absolute_error_lambda_2[0]))
    print('Relative Error lambda_1: {:.5f}%'.format(relative_error_lambda_1[0]))
    print('Relative Error lambda_2: {:.5f}%'.format(relative_error_lambda_2[0]))

    u_pred_full = model.predict_u(X_full)
    err_u_full = np.linalg.norm(u_full - u_pred_full, 2) / np.linalg.norm(u_full, 2)
    print(f"Relative L2 error (full field) - u: {err_u_full:.5e}")