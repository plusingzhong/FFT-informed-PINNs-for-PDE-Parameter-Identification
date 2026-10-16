import tensorflow as tf
import numpy as np
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

class MAXWELL_RFF:
    def __init__(self, data_sampler, res_sampler, bc_sampler,
                 ic_sampler,
                 layers, kernel_size):
        X, _ = res_sampler.sample(np.int32(1e3))
        self.mu_X, self.sigma_X = X.mean(0), X.std(0)
        self.mu_t, self.sigma_t = self.mu_X[0], self.sigma_X[0]
        self.mu_x, self.sigma_x = self.mu_X[1], self.sigma_X[1]

        # Samplers
        self.data_sampler = data_sampler
        self.res_sampler = res_sampler
        self.bc_sampler = bc_sampler
        self.ic_sampler = ic_sampler

        # net
        self.layers = layers
        self.weights, self.biases = self.initialize_NN(layers)

        self.W_t = tf.Variable(tf.random_normal([1, layers[0] // 2], dtype=tf.float32) * 1.0,
                                dtype=tf.float32, trainable=False)

        self.W_x = tf.Variable(tf.random_normal([1, layers[0] // 2], dtype=tf.float32) * 1.0,
                                dtype=tf.float32, trainable=False)

        self.sigma = tf.constant(0.2, dtype=tf.float32)
        self.mu = tf.Variable([0.5], dtype=tf.float32, name='mu')
        self.eps = tf.Variable([0.0], dtype=tf.float32, name='sigma')

        self.kernel_size = kernel_size
        self.D_data = 2 * self.kernel_size
        self.D_res = 2 * self.kernel_size
        self.D_bc = self.kernel_size
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
        self.u_bc_pred, _ = self.net_uv(self.t_bc_tf, self.x_bc_tf)
        self.r_u_pred, self.r_v_pred = self.net_r(self.t_r_tf, self.x_r_tf)

        self.loss_u_data = tf.reduce_mean(tf.square(self.u_tf - self.u_data_pred))
        self.loss_v_data = tf.reduce_mean(tf.square(self.v_tf - self.v_data_pred))
        self.loss_data = self.loss_u_data + self.loss_v_data

        self.loss_u_ic = tf.reduce_mean(tf.square(self.u_ic_tf - self.u_ic_pred))
        self.loss_v_ic = tf.reduce_mean(tf.square(self.v_ic_tf))
        self.loss_ic = self.loss_u_ic + self.loss_v_ic

        self.loss_res_u = tf.reduce_mean(tf.square(self.r_u_pred))
        self.loss_res_v = tf.reduce_mean(tf.square(self.r_v_pred))
        self.loss_res = self.loss_res_u + self.loss_res_v

        self.loss_bc_u = tf.reduce_mean(tf.square(self.u_bc_pred))
        # self.loss_bc_v = tf.reduce_mean(tf.square(self.v_bc_pred))
        # self.loss_bc = self.loss_bc_u + self.loss_bc_v
        self.loss_bc = self.loss_bc_u

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

        self.global_step = tf.Variable(0, trainable=False)
        starter_learning_rate = 1e-3

        self.learning_rate = tf.train.exponential_decay(starter_learning_rate, self.global_step,
                                                        1000, 0.9, staircase=False)

        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)

        self.train_op_Adam = optimizer.minimize(self.loss, global_step=self.global_step)


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

        u_ntk_bc, v_ntk_bc = self.net_uv(self.t_bc_ntk_tf, self.x_bc_ntk_tf)
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

        self.saver = tf.train.Saver()

        # init
        init = tf.global_variables_initializer()
        self.sess.run(init)

    def xavier_init(self, size):
        in_dim, out_dim = size[0], size[1]
        xavier_stddev = np.sqrt(2 / (in_dim + out_dim))
        # xavier_stddev = 1. / np.sqrt((in_dim + out_dim) / 2.)
        return tf.Variable(tf.random_normal([in_dim, out_dim], stddev=xavier_stddev), dtype=tf.float32)

    def initialize_NN(self, layers):
        weights = []
        biases = []
        num_layers = len(layers)
        input_dim = layers[0]

        for l in range(0, num_layers - 2):
            W = self.xavier_init(size=[input_dim, layers[l + 1]])
            b = tf.Variable(tf.random_normal([1, layers[l + 1]], dtype=tf.float32), dtype=tf.float32)
            weights.append(W)
            biases.append(b)
            input_dim = layers[l + 1]

        W_last = self.xavier_init(size=[layers[-2], layers[-1]])
        b_last = tf.Variable(tf.random_normal([1, layers[-1]], dtype=tf.float32), dtype=tf.float32)
        weights.append(W_last)
        biases.append(b_last)

        return weights, biases

    def neural_net(self, H):
        t = H[:, 0:1]
        x = H[:, 1:2]

        H_t = tf.concat([tf.sin(tf.matmul(t, self.W_t)),
                         tf.cos(tf.matmul(t, self.W_t))], 1)

        H_x = tf.concat([tf.sin(tf.matmul(x, self.W_x)),
                         tf.cos(tf.matmul(x, self.W_x))], 1)

        num_layers = len(self.layers)
        for l in range(0, num_layers - 2):
            W = self.weights[l]
            b = self.biases[l]
            H_t = tf.tanh(tf.add(tf.matmul(H_t, W), b))
            H_x = tf.tanh(tf.add(tf.matmul(H_x, W), b))

        H_final = tf.multiply(H_t, H_x)

        W = self.weights[-1]
        b = self.biases[-1]
        Y = tf.add(tf.matmul(H_final, W), b)
        return Y

    def net_uv(self, t, x):
        X = tf.concat([t, x], 1)
        out = self.neural_net(X)
        u = out[:, 0:1]
        v = out[:, 1:2]
        return u, v

    def net_uv_x(self, t, x):
        u, v = self.net_uv(t, x)
        u_x = tf.gradients(u, x)[0] / self.sigma_x
        v_x = tf.gradients(v, x)[0] / self.sigma_x
        return u_x, v_x

    def net_r(self, t, x):
        u, v = self.net_uv(t, x)

        E_t = tf.gradients(u, t)[0] / self.sigma_t
        E_x = tf.gradients(u, x)[0] / self.sigma_x

        H_t = tf.gradients(v, t)[0] / self.sigma_t
        H_x = tf.gradients(v, x)[0] / self.sigma_x

        r_E = self.eps[0] * E_t - H_x + self.sigma * u
        r_H = self.mu[0] * H_t - E_x
        return r_E, r_H


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
        X = (X - self.mu_X) / self.sigma_X
        return X, Y

    def fetch_minibatch_residual(self, N):
        X, _ = self.res_sampler.sample(N)
        X = (X - self.mu_X) / self.sigma_X
        return X

    def fetch_minibatch_bc(self, N):
        X, Y = self.bc_sampler.sample(N)
        X = (X - self.mu_X) / self.sigma_X
        return X, Y
    def fetch_minibatch_ic(self, N):
        X, Y = self.ic_sampler.sample(N)
        X = (X - self.mu_X) / self.sigma_X
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
                lambda1_value = self.sess.run(self.eps)

                lambda2_value = self.sess.run(self.mu)

                current_total_iter = it
                self.iterations.append(current_total_iter)
                self.loss_total.append(loss_value)
                self.loss_data_all.append(loss_data_value)
                self.loss_res_all.append(loss_res_value)
                self.loss_bc_all.append(loss_bc_value)
                self.loss_ic_all.append(loss_ic_value)
                self.lambda1_all.append(lambda1_value[0])
                self.lambda2_all.append(lambda2_value[0])

                print('It: %d, Loss: %.5e, Lambda1: %.5f, Lambda2: %.5f, Time: %.2f' %
                      (it, loss_value, lambda1_value, lambda2_value, elapsed))

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

                    self.lambda_data_val = np.clip(self.lambda_data_val, 0.1, 100.0)
                    self.lambda_res_val = np.clip(self.lambda_res_val, 0.1, 100.0)
                    self.lambda_bc_val = np.clip(self.lambda_bc_val, 0.1, 100.0)
                    self.lambda_ic_val = np.clip(self.lambda_ic_val, 0.1, 100.0)

        total_elapsed_time = time.time() - total_start_time
        print(f"Total training time: {total_elapsed_time:.2f} seconds")

    def predict(self, X_star):
        X_star = (X_star - self.mu_X) / self.sigma_X
        tf_dict = {self.t_data_tf: X_star[:, 0:1],
                   self.x_data_tf: X_star[:, 1:2],
                   self.lambda_data_tf: self.lambda_data_val,
                   self.lambda_res_tf: self.lambda_res_val,
                   self.lambda_bc_tf: self.lambda_bc_val}
        u_pred, v_pred = self.sess.run([self.u_data_pred, self.v_data_pred], tf_dict)
        return u_pred, v_pred

    def get_pred_parameters(self):
        eps_value = self.sess.run(self.eps)[0]
        mu_value = self.sess.run(self.mu)[0]
        return eps_value, mu_value

if __name__ == '__main__':

    data = np.load('Maxwell_PINN_Data.npz')

    t_train = data['t_ds'].flatten()[:, None]
    x_train = data['x_E_ds'].flatten()[:, None]
    x_H_train = data['x_H_ds'].flatten()[:, None]
    E_train = data['E_exact']
    H_train_raw = data['H_exact']

    H_train_aligned = np.zeros_like(E_train)
    for i in range(E_train.shape[0]):
        H_train_aligned[i, :] = np.interp(x_train.flatten(),
                                          x_H_train.flatten(),
                                          H_train_raw[i, :])

    T, X = np.meshgrid(t_train, x_train, indexing='ij')
    X_train = np.hstack((T.flatten()[:, None], X.flatten()[:, None]))

    t_full = data['t_original'].flatten()[:, None]
    x_full = data['x_E_original'].flatten()[:, None]
    E_full = data['E_full']
    H_full_raw = data['H_full']
    x_H_full = data['x_H_original'].flatten()[:, None]
    H_full_aligned = np.zeros_like(E_full)
    for i in range(E_full.shape[0]):
        H_full_aligned[i, :] = np.interp(x_full.flatten(),
                                          x_H_full.flatten(),
                                          H_full_raw[i, :])

    T_full, X_full = np.meshgrid(t_full, x_full, indexing='ij')
    X_star = np.hstack((T_full.flatten()[:, None], X_full.flatten()[:, None]))

    E_flat = E_train.flatten()[:, None]
    H_flat = H_train_aligned.flatten()[:, None]
    UV_train = np.hstack([E_flat, H_flat])

    ics_mask = X_train[:, 0] == 0.0
    X_ics = X_train[ics_mask]
    UV_ics = UV_train[ics_mask]
    ic_sampler = DataSampler(X_ics, UV_ics, name='ICS')

    L_val = 1.0
    bc1_mask = X_train[:, 1] == 0.0
    bc2_mask = X_train[:, 1] == L_val
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

    dom_coords = np.array([[0.0, 0.0], [1.0, L_val]])

    res_sampler = LHSSampler(2, dom_coords, name='Residual')

    eps_true = 0.2
    sigma_true = 0.2
    mu_true = 0.3

    layers = [48, 48, 48, 48, 2]
    kernel_size = 32

    model = MAXWELL_RFF(data_sampler, res_sampler, bc_sampler,
                             ic_sampler,
                             layers, kernel_size)

    model.train(nIter=20000, batch_size=kernel_size, log_NTK=True, update_weights=True)

    eps_pred, mu_pred = model.get_pred_parameters()

    abs_error_eps = np.abs(eps_pred - eps_true)
    abs_error_mu = np.abs(mu_pred - mu_true)
    rel_error_eps = (abs_error_eps / np.abs(eps_true)) * 100
    rel_error_mu = (abs_error_mu / np.abs(mu_true)) * 100

    print(f"True eps: {eps_true}, Identified: {eps_pred:.5f}")
    print(f"Absolute Error: {abs_error_eps:.5f}, Relative Error: {rel_error_eps:.5f}%")

    print(f"True mu: {mu_true}, Identified: {mu_pred:.5f}")
    print(f"Absolute Error: {abs_error_mu:.5f}, Relative Error: {rel_error_mu:.5f}%")

    u_pred_full, v_pred_full = model.predict(X_star)
    E_full_flat = E_full.flatten()[:, None]
    H_full_aligned_flat = H_full_aligned.flatten()[:, None]
    err_u_full = np.linalg.norm(E_full_flat - u_pred_full, 2) / np.linalg.norm(E_full_flat, 2)
    err_v_full = np.linalg.norm(H_full_aligned_flat - v_pred_full, 2) / np.linalg.norm(H_full_aligned_flat, 2)
    print(f"Relative L2 error (full field) - u: {err_u_full:.5e}, v: {err_v_full:.5e}")
