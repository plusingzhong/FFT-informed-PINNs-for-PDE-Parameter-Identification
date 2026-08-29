import tensorflow as tf
import numpy as np
import scipy.io as sio
import time
from pyDOE import lhs
from Compute_Jacobian import jacobian
import matplotlib.pyplot as plt

np.random.seed(1234)
tf.set_random_seed(1234)

class Sampler:
    def __init__(self, dim, coords, func, name=None):
        self.dim = dim
        self.coords = coords
        self.func = func
        self.name = name

    def sample(self, N):
        x = self.coords[0:1, :] + (self.coords[1:2, :] - self.coords[0:1, :]) * np.random.rand(N, self.dim)
        y = self.func(x)
        return x, y


class DataSampler:
    # Initialize the class for direct data sampling
    def __init__(self, X, Y, name=None):
        self.X = X
        self.Y = Y
        self.name = name
        self.dim = X.shape[1]
        self.total_points = X.shape[0]

    def sample(self, N):
        if N > self.total_points:
            idx = np.random.choice(self.total_points, N, replace=True)
        else:
            idx = np.random.choice(self.total_points, N, replace=False)
        X_batch = self.X[idx, :]
        Y_batch = self.Y[idx, :]
        return X_batch, Y_batch


class LHSSampler:
    def __init__(self, dim, coords, name=None):
        self.dim = dim
        self.coords = coords
        self.name = name

    def sample(self, N):
        samples = lhs(self.dim, N)
        x = self.coords[0:1, :] + (self.coords[1:2, :] - self.coords[0:1, :]) * samples
        return x, np.zeros((x.shape[0], 2))


class FHN_PINN_NTK:
    def __init__(self, data_sampler, res_sampler, bc_sampler,
                 ic_sampler,
                 layers, nu1, nu2, kernel_size):
        X, _ = res_sampler.sample(np.int32(1e3))
        self.mu_X, self.sigma_X = X.mean(0), X.std(0)
        self.mu_t, self.sigma_t = self.mu_X[0], self.sigma_X[0]
        self.mu_x, self.sigma_x = self.mu_X[1], self.sigma_X[1]

        # Normalization bounds for input scaling
        self.lb = np.array([0.0, 0.0])
        self.ub = np.array([20.0, 1.0])

        # Samplers
        self.data_sampler = data_sampler
        self.res_sampler = res_sampler
        self.bc_sampler = bc_sampler
        self.ic_sampler = ic_sampler

        # net
        self.layers = layers
        self.weights, self.biases = self.initialize_NN(layers)

        self.nu1 = tf.constant(nu1, dtype=tf.float32)
        self.nu2 = tf.constant(nu2, dtype=tf.float32)
        self.lambda1 = tf.Variable([0.0], dtype=tf.float32, name='lambda1')
        self.lambda2 = tf.Variable([0.0], dtype=tf.float32, name='lambda2')
        self.lambda3 = tf.Variable([0.0], dtype=tf.float32, name='lambda3')

        self.kernel_size = kernel_size
        self.D_data = 2 * self.kernel_size
        self.D_res = 2 * self.kernel_size
        self.D_bc = 2 * self.kernel_size
        self.D_ic = 2 * self.kernel_size

        self.sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))

        self.u_tf = tf.placeholder(tf.float32, shape=(None, 1))
        self.v_tf = tf.placeholder(tf.float32, shape=(None, 1))
        self.t_data_tf = tf.placeholder(tf.float32, shape=(None, 1))
        self.x_data_tf = tf.placeholder(tf.float32, shape=(None, 1))

        self.u_ic_tf = tf.placeholder(tf.float32, shape=(None, 1))
        self.v_ic_tf = tf.placeholder(tf.float32, shape=(None, 1))
        self.t_ic_tf = tf.placeholder(tf.float32, shape=(None, 1))
        self.x_ic_tf = tf.placeholder(tf.float32, shape=(None, 1))

        self.t_bc_tf = tf.placeholder(tf.float32, shape=(None, 1))
        self.x_bc_tf = tf.placeholder(tf.float32, shape=(None, 1))

        self.t_r_tf = tf.placeholder(tf.float32, shape=(None, 1))
        self.x_r_tf = tf.placeholder(tf.float32, shape=(None, 1))

        self.u_data_pred, self.v_data_pred = self.net_uv(self.t_data_tf, self.x_data_tf)
        self.u_ic_pred, self.v_ic_pred = self.net_uv(self.t_ic_tf, self.x_ic_tf)
        self.u_bc_pred, self.v_bc_pred = self.net_uv_x(self.t_bc_tf, self.x_bc_tf)
        self.r_u_pred, self.r_v_pred = self.net_r(self.t_r_tf, self.x_r_tf)

        self.loss_u_data = tf.reduce_mean(tf.square(self.u_tf - self.u_data_pred))
        self.loss_v_data = tf.reduce_mean(tf.square(self.v_tf - self.v_data_pred))
        self.loss_data = self.loss_u_data + self.loss_v_data

        self.loss_u_ic = tf.reduce_mean(tf.square(self.u_ic_tf - self.u_ic_pred))
        self.loss_v_ic = tf.reduce_mean(tf.square(self.v_ic_tf - self.v_ic_pred))
        self.loss_ic = self.loss_u_ic + self.loss_v_ic

        self.loss_res_u = tf.reduce_mean(tf.square(self.r_u_pred))
        self.loss_res_v = tf.reduce_mean(tf.square(self.r_v_pred))
        self.loss_res = self.loss_res_u + self.loss_res_v

        self.loss_bc_u = tf.reduce_mean(tf.square(self.u_bc_pred))
        self.loss_bc_v = tf.reduce_mean(tf.square(self.v_bc_pred))
        self.loss_bc = self.loss_bc_u + self.loss_bc_v

        self.lambda_data_val = np.array(1.0)
        self.lambda_res_val = np.array(1.0)
        self.lambda_bc_val = np.array(1.0)
        self.lambda_ic_val = np.array(1.0)

        self.lambda_data_tf = tf.placeholder(tf.float32, shape=self.lambda_data_val.shape)
        self.lambda_res_tf = tf.placeholder(tf.float32, shape=self.lambda_res_val.shape)
        self.lambda_bc_tf = tf.placeholder(tf.float32, shape=self.lambda_bc_val.shape)
        self.lambda_ic_tf = tf.placeholder(tf.float32, shape=self.lambda_ic_val.shape)

        self.loss = (self.lambda_data_tf * self.loss_data +
                     self.lambda_ic_tf * self.loss_ic +
                     self.lambda_res_tf * self.loss_res +
                     self.lambda_bc_tf * self.loss_bc)

        self.optimizer_Adam = tf.train.AdamOptimizer()
        self.train_op_Adam = self.optimizer_Adam.minimize(self.loss)

        # NTK
        self.t_data_ntk_tf = tf.placeholder(tf.float32, shape=(self.kernel_size, 1))
        self.x_data_ntk_tf = tf.placeholder(tf.float32, shape=(self.kernel_size, 1))

        self.t_res_ntk_tf = tf.placeholder(tf.float32, shape=(self.kernel_size, 1))
        self.x_res_ntk_tf = tf.placeholder(tf.float32, shape=(self.kernel_size, 1))

        self.t_bc_ntk_tf = tf.placeholder(tf.float32, shape=(self.kernel_size, 1))
        self.x_bc_ntk_tf = tf.placeholder(tf.float32, shape=(self.kernel_size, 1))

        self.t_ic_ntk_tf = tf.placeholder(tf.float32, shape=(self.kernel_size, 1))
        self.x_ic_ntk_tf = tf.placeholder(tf.float32, shape=(self.kernel_size, 1))

        u_ntk_data, v_ntk_data = self.net_uv(self.t_data_ntk_tf, self.x_data_ntk_tf)
        self.f_data_ntk = tf.concat([u_ntk_data, v_ntk_data], axis=0)

        r_u_ntk, r_v_ntk = self.net_r(self.t_res_ntk_tf, self.x_res_ntk_tf)
        self.f_res_ntk = tf.concat([r_u_ntk, r_v_ntk], axis=0)

        u_ntk_bc, v_ntk_bc = self.net_uv_x(self.t_bc_ntk_tf, self.x_bc_ntk_tf)
        self.f_bc_ntk = tf.concat([u_ntk_bc, v_ntk_bc], axis=0)

        u_ntk_ic, v_ntk_ic = self.net_uv(self.t_ic_ntk_tf, self.x_ic_ntk_tf)
        self.f_ic_ntk = tf.concat([u_ntk_ic, v_ntk_ic], axis=0)

        self.J_data = self.compute_jacobian(self.f_data_ntk)
        self.J_res = self.compute_jacobian(self.f_res_ntk)
        self.J_bc = self.compute_jacobian(self.f_bc_ntk)
        self.J_ic = self.compute_jacobian(self.f_ic_ntk)

        self.K_data = self.compute_ntk(self.J_data, self.D_data, self.J_data, self.D_data)
        self.K_res = self.compute_ntk(self.J_res, self.D_res, self.J_res, self.D_res)
        self.K_bc = self.compute_ntk(self.J_bc, self.D_bc, self.J_bc, self.D_bc)
        self.K_ic = self.compute_ntk(self.J_ic, self.D_ic, self.J_ic, self.D_ic)

        # Logger
        self.iterations = []
        self.loss_total = []
        self.loss_data_all = []
        self.loss_res_all = []
        self.loss_bc_all = []
        self.loss_ic_all = []
        self.lambda1_all = []
        self.lambda2_all = []
        self.lambda3_all = []

        self.saver = tf.train.Saver()

        # init
        init = tf.global_variables_initializer()
        self.sess.run(init)

    def xavier_init(self, size):
        in_dim, out_dim = size[0], size[1]
        xavier_stddev = 1. / np.sqrt((in_dim + out_dim) / 2.)
        return tf.Variable(tf.random_normal([in_dim, out_dim], stddev=xavier_stddev), dtype=tf.float32)

    def initialize_NN(self, layers):
        weights = []
        biases = []
        num_layers = len(layers)
        for l in range(0, num_layers - 1):
            W = self.xavier_init(size=[layers[l], layers[l + 1]])
            b = tf.Variable(tf.zeros([1, layers[l + 1]], dtype=tf.float32), dtype=tf.float32)
            weights.append(W)
            biases.append(b)
        return weights, biases

    def neural_net(self, H):
        H = 2.0 * (H - self.lb) / (self.ub - self.lb) - 1.0

        for l in range(len(self.weights) - 1):
            W = self.weights[l]
            b = self.biases[l]
            H = tf.tanh(tf.add(tf.matmul(H, W), b))

        W = self.weights[-1]
        b = self.biases[-1]
        Y = tf.add(tf.matmul(H, W), b)
        return Y

    def net_uv(self, t, x):
        X = tf.concat([t, x], 1)
        out = self.neural_net(X)
        u = out[:, 0:1]
        v = out[:, 1:2]
        return u, v

    def net_uv_x(self, t, x):
        u, v = self.net_uv(t, x)
        u_x = tf.gradients(u, x)[0]
        v_x = tf.gradients(v, x)[0]
        return u_x, v_x

    def net_r(self, t, x):
        u, v = self.net_uv(t, x)
        u_t = tf.gradients(u, t)[0]
        u_x = tf.gradients(u, x)[0]
        u_xx = tf.gradients(u_x, x)[0]
        v_t = tf.gradients(v, t)[0]
        v_x = tf.gradients(v, x)[0]
        v_xx = tf.gradients(v_x, x)[0]
        r_u = u_t - self.nu1 * u_xx - (self.lambda1[0] * u - v - u ** 3)
        r_v = v_t - self.nu2 * v_xx - (self.lambda2[0] * u - self.lambda3[0] * v)
        return r_u, r_v

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

    def compute_ntk(self, J1_list, D1, J2_list, D2):
        Ker = tf.zeros((D1, D2), dtype=tf.float32)
        for k in range(len(J1_list)):
            J1 = tf.reshape(J1_list[k], shape=(D1, -1))
            J2 = tf.reshape(J2_list[k], shape=(D2, -1))
            Ker += tf.matmul(J1, tf.transpose(J2))
        return Ker

    def fetch_minibatch_data(self, N):
        X, Y = self.data_sampler.sample(N)
        return X, Y

    def fetch_minibatch_residual(self, N):
        X, _ = self.res_sampler.sample(N)
        return X

    def fetch_minibatch_bc(self, N):
        X, Y = self.bc_sampler.sample(N)
        return X, Y

    def fetch_minibatch_ic(self, N):
        X, Y = self.ic_sampler.sample(N)
        return X, Y

    def train(self, nIter=10000, batch_size=200, log_NTK=True, update_weights=True):
        total_start_time = time.time()
        start_time = time.time()

        for it in range(nIter):
            X_data_batch, UV_data_batch = self.fetch_minibatch_data(batch_size)
            X_ic_batch, UV_ic_batch = self.fetch_minibatch_ic(batch_size)
            X_bc_batch, _ = self.fetch_minibatch_bc(batch_size)
            X_r_batch = self.fetch_minibatch_residual(batch_size)

            tf_dict = {
                self.t_data_tf: X_data_batch[:, 0:1],
                self.x_data_tf: X_data_batch[:, 1:2],
                self.u_tf: UV_data_batch[:, 0:1],
                self.v_tf: UV_data_batch[:, 1:2],
                self.t_ic_tf: X_ic_batch[:, 0:1],
                self.x_ic_tf: X_ic_batch[:, 1:2],
                self.u_ic_tf: UV_ic_batch[:, 0:1],
                self.v_ic_tf: UV_ic_batch[:, 1:2],
                self.t_bc_tf: X_bc_batch[:, 0:1],
                self.x_bc_tf: X_bc_batch[:, 1:2],
                self.t_r_tf: X_r_batch[:, 0:1],
                self.x_r_tf: X_r_batch[:, 1:2],
                self.lambda_data_tf: self.lambda_data_val,
                self.lambda_res_tf: self.lambda_res_val,
                self.lambda_bc_tf: self.lambda_bc_val,
                self.lambda_ic_tf: self.lambda_ic_val
            }

            self.sess.run(self.train_op_Adam, tf_dict)

            if it % 100 == 0:
                elapsed = time.time() - start_time
                loss_value, loss_data_value, loss_res_value, loss_bc_value, loss_ic_value = self.sess.run(
                    [self.loss, self.loss_data, self.loss_res, self.loss_bc, self.loss_ic], tf_dict)
                lambda1_value = self.sess.run(self.lambda1)
                lambda2_value = self.sess.run(self.lambda2)
                lambda3_value = self.sess.run(self.lambda3)

                current_total_iter = it
                self.iterations.append(current_total_iter)
                self.loss_total.append(loss_value)
                self.loss_data_all.append(loss_data_value)
                self.loss_res_all.append(loss_res_value)
                self.loss_bc_all.append(loss_bc_value)
                self.loss_ic_all.append(loss_ic_value)
                self.lambda1_all.append(lambda1_value[0])
                self.lambda2_all.append(lambda2_value[0])
                self.lambda3_all.append(lambda3_value[0])

                print('It: %d, Loss: %.5e, Lambda1: %.5f, Lambda2: %.5f, Lambda3: %.5f, Time: %.2f' %
                      (it, loss_value, lambda1_value, lambda2_value, lambda3_value, elapsed))

            if log_NTK and (it % 100 == 0):
                tf_ntk = {
                    self.t_data_ntk_tf: X_data_batch[:, 0:1],
                    self.x_data_ntk_tf: X_data_batch[:, 1:2],
                    self.t_res_ntk_tf: X_r_batch[:, 0:1],
                    self.x_res_ntk_tf: X_r_batch[:, 1:2],
                    self.t_bc_ntk_tf: X_bc_batch[:, 0:1],
                    self.x_bc_ntk_tf: X_bc_batch[:, 1:2],
                    self.t_ic_ntk_tf: X_ic_batch[:, 0:1],
                    self.x_ic_ntk_tf: X_ic_batch[:, 1:2]
                }
                Kd, Kr, Kb, Ki = self.sess.run([self.K_data, self.K_res, self.K_bc, self.K_ic], tf_ntk)

                if update_weights:
                    lam_sum = np.trace(Kd) + np.trace(Kr) + np.trace(Kb) + np.trace(Ki)
                    self.lambda_data_val = lam_sum / np.trace(Kd)
                    self.lambda_res_val = lam_sum / np.trace(Kr)
                    self.lambda_bc_val = lam_sum / np.trace(Kb)
                    self.lambda_ic_val = lam_sum / np.trace(Ki)

        total_elapsed_time = time.time() - total_start_time
        print(f"Total training time: {total_elapsed_time:.2f} seconds")

    def predict(self, X_star):
        tf_dict = {self.t_data_tf: X_star[:, 0:1],
                   self.x_data_tf: X_star[:, 1:2]}
        u_pred, v_pred = self.sess.run([self.u_data_pred, self.v_data_pred], tf_dict)
        return u_pred, v_pred

    def get_pred_parameters(self):
        lambda1_value = self.sess.run(self.lambda1)[0]
        lambda2_value = self.sess.run(self.lambda2)[0]
        lambda3_value = self.sess.run(self.lambda3)[0]
        return lambda1_value, lambda2_value, lambda3_value


if __name__ == '__main__':
    data = sio.loadmat('D:\\FFT-PINN\\FFT-PINN\\FHN\\FHN.mat')

    t_full = data['t'].flatten()
    x_full = data['x'].flatten()
    Exact_u_full = data['u1']
    Exact_v_full = data['u2']

    T_full, X_full = np.meshgrid(t_full, x_full)
    X_full = np.hstack((T_full.flatten()[:, None], X_full.flatten()[:, None]))
    u_full = Exact_u_full.T.flatten()[:, None]
    v_full = Exact_v_full.T.flatten()[:, None]
    UV_full = np.hstack([u_full, v_full])

    t = data['t_small'].flatten()[:, None]
    x = data['x_small'].flatten()[:, None]
    Exact_u = data['u1_small']
    Exact_v = data['u2_small']

    T, X = np.meshgrid(t, x)
    X_train = np.hstack((T.flatten()[:, None], X.flatten()[:, None]))
    u_train = Exact_u.T.flatten()[:, None]
    v_train = Exact_v.T.flatten()[:, None]
    UV_train = np.hstack([u_train, v_train])

    ics_mask = X_train[:, 0] == 0.0
    X_ics = X_train[ics_mask]
    UV_ics = UV_train[ics_mask]
    ic_sampler = DataSampler(X_ics, UV_ics, name='ICS')

    bc1_mask = X_train[:, 1] == 0.0  # x=0
    bc2_mask = X_train[:, 1] == 1.0  # x=1
    X_bc = np.vstack([X_train[bc1_mask], X_train[bc2_mask]])
    UV_bc = np.vstack([UV_train[bc1_mask], UV_train[bc2_mask]])
    bc_sampler = DataSampler(X_bc, UV_bc, name='BC')

    internal_mask = ~(ics_mask | bc1_mask | bc2_mask)
    X_internal = X_train[internal_mask]
    UV_internal = UV_train[internal_mask]
    noise_level = 0.05

    u_internal = UV_internal[:, 0:1]
    v_internal = UV_internal[:, 1:2]

    u_noisy = u_internal + noise_level * np.std(u_internal) * np.random.randn(*u_internal.shape)
    v_noisy = v_internal + noise_level * np.std(v_internal) * np.random.randn(*v_internal.shape)

    UV_noisy = np.hstack([u_noisy, v_noisy])
    data_sampler = DataSampler(X_internal, UV_noisy, name='Internal Data')

    dom_coords = np.array([[0.0, 0.0], [20.0, 1.0]])

    res_sampler = LHSSampler(2, dom_coords, name='Residual')

    nu1, nu2 = 0.001, 0.004
    lambda1_true = 1.0
    lambda2_true = 0.4
    lambda3_true = 0.2

    layers = [2, 36, 36, 36, 36, 2]

    kernel_size = 32

    model = FHN_PINN_NTK(data_sampler, res_sampler, bc_sampler,
                             ic_sampler,
                             layers, nu1, nu2, kernel_size)

    model.train(nIter=40000, batch_size=kernel_size, log_NTK=True, update_weights=True)

    u_pred_full, v_pred_full = model.predict(X_full)
    err_u_full = np.linalg.norm(u_full - u_pred_full, 2) / np.linalg.norm(u_full, 2)
    err_v_full = np.linalg.norm(v_full - v_pred_full, 2) / np.linalg.norm(v_full, 2)
    print(f"Relative L2 error (full field) - u: {err_u_full:.5e}, v: {err_v_full:.5e}")

    lambda1_pred, lambda2_pred, lambda3_pred = model.get_pred_parameters()

    abs_error_lambda1 = np.abs(lambda1_pred - lambda1_true)
    abs_error_lambda2 = np.abs(lambda2_pred - lambda2_true)
    abs_error_lambda3 = np.abs(lambda3_pred - lambda3_true)
    rel_error_lambda1 = (abs_error_lambda1 / np.abs(lambda1_true)) * 100
    rel_error_lambda2 = (abs_error_lambda2 / np.abs(lambda2_true)) * 100
    rel_error_lambda3 = (abs_error_lambda3 / np.abs(lambda3_true)) * 100

    print(f"True lambda1: {lambda1_true}, Identified: {lambda1_pred:.5f}")
    print(f"Absolute Error: {abs_error_lambda1:.5f}")
    print(f"Relative Error: {rel_error_lambda1:.5f}%")

    print(f"True lambda2: {lambda2_true}, Identified: {lambda2_pred:.5f}")
    print(f"Absolute Error: {abs_error_lambda2:.5f}")
    print(f"Relative Error: {rel_error_lambda2:.5f}%")

    print(f"True lambda3: {lambda3_true}, Identified: {lambda3_pred:.5f}")
    print(f"Absolute Error: {abs_error_lambda3:.5f}")
    print(f"Relative Error: {rel_error_lambda3:.5f}%")
