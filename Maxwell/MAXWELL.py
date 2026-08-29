import numpy as np
import matplotlib.pyplot as plt

L = 1.0
T = 1.0
eps = 0.2
mu = 0.3
sigma = 0.2

nx = 100
nt = 500
dx = L / nx
dt = T / nt

Ca = (1.0 - sigma * dt / (2 * eps)) / (1.0 + sigma * dt / (2 * eps))
Cb = (dt / (eps * dx)) / (1.0 + sigma * dt / (2 * eps))
Ch = dt / (mu * dx)

E = np.zeros(nx + 1)
H = np.zeros(nx)
x_E = np.linspace(0, L, nx + 1)

E[:] = (0.5 * np.sin(4 * np.pi * x_E / L) +
        0.8 * np.sin(6 * np.pi * x_E / L))

E_history_all = np.zeros((nt, nx + 1))
H_history_all = np.zeros((nt, nx))

for n in range(nt):
    E_history_all[n, :] = E
    H_history_all[n, :] = H

    for i in range(nx):
        H[i] = H[i] + Ch * (E[i + 1] - E[i])

    for i in range(1, nx):
        E[i] = Ca * E[i] + Cb * (H[i] - H[i - 1])

    E[0] = 0.0
    E[-1] = 0.0

noise_level = 0.03

E_noisy = E_history_all + noise_level * np.std(E_history_all) * np.random.randn(*E_history_all.shape)
H_noisy = H_history_all + noise_level * np.std(H_history_all) * np.random.randn(*H_history_all.shape)

# plt.rcParams['font.sans-serif'] = ['SimHei']
# plt.rcParams['axes.unicode_minus'] = False
#
# fig, axes = plt.subplots(1, 2, figsize=(14, 6))
#
# im1 = axes[0].imshow(E_history_all.T, aspect='auto', origin='lower',
#                      extent=[0, T, 0, L], cmap='RdBu')
# axes[0].set_title('E 场时空演化伪彩色图')
# axes[0].set_xlabel('时间 t')
# axes[0].set_ylabel('空间位置 x')
# fig.colorbar(im1, ax=axes[0], label='E 场幅度')
#
# im2 = axes[1].imshow(H_history_all.T, aspect='auto', origin='lower',
#                      extent=[0, T, 0, L], cmap='RdBu')
# axes[1].set_title('H 场时空演化伪彩色图')
# axes[1].set_xlabel('时间 t')
# axes[1].set_ylabel('空间位置 x')
# fig.colorbar(im2, ax=axes[1], label='H 场幅度')

# plt.tight_layout()
# plt.show()

dt_downsample = 5
dx_downsample = 5

E_exact = E_noisy[::dt_downsample, ::dx_downsample]
E_exact_clean = E_history_all[::dt_downsample, ::dx_downsample]
Nt_ds, Nx_ds = E_exact.shape

H_exact = H_noisy[::dt_downsample, ::dx_downsample]
H_exact_clean = H_history_all[::dt_downsample, ::dx_downsample]

dt_new = dt * dt_downsample
dx_new = dx * dx_downsample

print(f"原始数据形状: {E_noisy.shape} (时间点×空间点)")
print(f"降采样后数据形状: {E_exact.shape} (时间点×空间点)\n")

Fs_t = 1.0 / dt_new
F_time_all = np.fft.fft(E_exact, axis=0)

n_pos_t = Nt_ds // 2 + 1 if Nt_ds % 2 == 0 else (Nt_ds + 1) // 2
P_time_all = np.abs(F_time_all[:n_pos_t, :]) ** 2
P_time_all[1:-1, :] *= 2
P_time_mean = np.mean(P_time_all, axis=1)

f_axis = np.arange(n_pos_t) * (Fs_t / Nt_ds)
w_axis = 2.0 * np.pi * f_axis

sorted_idx_t = np.argsort(P_time_mean)[::-1]
cum_energy_t = np.cumsum(P_time_mean[sorted_idx_t]) / np.sum(P_time_mean)
n_keep_t = np.where(cum_energy_t >= 0.98)[0][0] + 1
keep_idx_t = sorted_idx_t[:n_keep_t]

freq_main = f_axis[keep_idx_t]
w_main = w_axis[keep_idx_t]
energy_frac_t = P_time_mean[keep_idx_t] / np.sum(P_time_mean)

print(f'--- 全场主时间频率（覆盖≥ {0.98 * 100:.2f}% 平均能量） ---')
print(f'主频率分量总数: {n_keep_t}')
print('%-14s %-18s %-14s' % ('频率 (Hz)', '角频率 (rad/s)', '能量占比 (%)'))
for ii in range(n_keep_t):
    print('%-14.6f %-18.6f %-14.6f' % (freq_main[ii], w_main[ii], 100 * energy_frac_t[ii]))
print(f'累计能量占比: {100 * cum_energy_t[n_keep_t - 1]:.6f}%\n')

Fs_x = 1.0 / dx_new

u_space_fft = E_exact[:, :-1]
Nx_ds_fft = u_space_fft.shape[1]

F_space_all = np.fft.fft(u_space_fft, axis=1)

n_pos_x = Nx_ds_fft // 2 + 1 if Nx_ds_fft % 2 == 0 else (Nx_ds_fft + 1) // 2
P_space_all = np.abs(F_space_all[:, :n_pos_x]) ** 2
P_space_all[:, 1:-1] *= 2
P_space_mean = np.mean(P_space_all, axis=0)

k_axis = np.arange(n_pos_x) * (Fs_x / Nx_ds_fft)
k_angular = 2.0 * np.pi * k_axis

sorted_idx_x = np.argsort(P_space_mean)[::-1]
cum_energy_x = np.cumsum(P_space_mean[sorted_idx_x]) / np.sum(P_space_mean)
n_keep_x = np.where(cum_energy_x >= 0.98)[0][0] + 1
keep_idx_x = sorted_idx_x[:n_keep_x]

k_main = k_axis[keep_idx_x]
k_angular_main = k_angular[keep_idx_x]
energy_frac_x = P_space_mean[keep_idx_x] / np.sum(P_space_mean)

print(f'--- 全场主空间波数（覆盖≥ 98.00% 平均能量） ---')
print(f'主波数分量总数: {n_keep_x}')
print('%-16s %-20s %-14s' % ('空间频率 (1/m)', '角波数 (rad/m)', '能量占比 (%)'))
for ii in range(n_keep_x):
    print('%-16.6f %-20.6f %-14.6f' % (k_main[ii], k_angular_main[ii], 100 * energy_frac_x[ii]))
print(f'累计能量占比: {100 * cum_energy_x[n_keep_x - 1]:.6f}%')

print(f"原始数据形状: {H_noisy.shape} (时间点×空间点)")
print(f"降采样后数据形状: {H_exact.shape} (时间点×空间点)\n")

Fs_t = 1.0 / dt_new
F_time_all = np.fft.fft(H_exact, axis=0)

n_pos_t = Nt_ds // 2 + 1 if Nt_ds % 2 == 0 else (Nt_ds + 1) // 2
P_time_all = np.abs(F_time_all[:n_pos_t, :]) ** 2
P_time_all[1:-1, :] *= 2
P_time_mean = np.mean(P_time_all, axis=1)

f_axis = np.arange(n_pos_t) * (Fs_t / Nt_ds)
w_axis = 2.0 * np.pi * f_axis

sorted_idx_t = np.argsort(P_time_mean)[::-1]
cum_energy_t = np.cumsum(P_time_mean[sorted_idx_t]) / np.sum(P_time_mean)
n_keep_t = np.where(cum_energy_t >= 0.98)[0][0] + 1
keep_idx_t = sorted_idx_t[:n_keep_t]

freq_main = f_axis[keep_idx_t]
w_main = w_axis[keep_idx_t]
energy_frac_t = P_time_mean[keep_idx_t] / np.sum(P_time_mean)

print(f'--- 全场主时间频率（覆盖≥ {0.98 * 100:.2f}% 平均能量） ---')
print(f'主频率分量总数: {n_keep_t}')
print('%-14s %-18s %-14s' % ('频率 (Hz)', '角频率 (rad/s)', '能量占比 (%)'))
for ii in range(n_keep_t):
    print('%-14.6f %-18.6f %-14.6f' % (freq_main[ii], w_main[ii], 100 * energy_frac_t[ii]))
print(f'累计能量占比: {100 * cum_energy_t[n_keep_t - 1]:.6f}%\n')

Fs_x = 1.0 / dx_new
F_space_all = np.fft.fft(H_exact, axis=1)
Nx_ds_fft = H_exact.shape[1]

n_pos_x = Nx_ds_fft // 2 + 1 if Nx_ds_fft % 2 == 0 else (Nx_ds_fft + 1) // 2
P_space_all = np.abs(F_space_all[:, :n_pos_x]) ** 2
P_space_all[:, 1:-1] *= 2
P_space_mean = np.mean(P_space_all, axis=0)

k_axis = np.arange(n_pos_x) * (Fs_x / Nx_ds_fft)
k_angular = 2.0 * np.pi * k_axis

sorted_idx_x = np.argsort(P_space_mean)[::-1]
cum_energy_x = np.cumsum(P_space_mean[sorted_idx_x]) / np.sum(P_space_mean)
n_keep_x = np.where(cum_energy_x >= 0.98)[0][0] + 1
keep_idx_x = sorted_idx_x[:n_keep_x]

k_main = k_axis[keep_idx_x]
k_angular_main = k_angular[keep_idx_x]
energy_frac_x = P_space_mean[keep_idx_x] / np.sum(P_space_mean)

print(f'--- 全场主空间波数（覆盖≥ 98.00% 平均能量） ---')
print(f'主波数分量总数: {n_keep_x}')
print('%-16s %-20s %-14s' % ('空间频率 (1/m)', '角波数 (rad/m)', '能量占比 (%)'))
for ii in range(n_keep_x):
    print('%-16.6f %-20.6f %-14.6f' % (k_main[ii], k_angular_main[ii], 100 * energy_frac_x[ii]))
print(f'累计能量占比: {100 * cum_energy_x[n_keep_x - 1]:.6f}%')

# t_array = np.linspace(0, T, nt, endpoint=False)
# x_H = np.linspace(dx / 2, L - dx / 2, nx)
# t_ds = t_array[::dt_downsample]
# x_E_ds = x_E[::dx_downsample]
# x_H_ds = x_H[::dx_downsample]
# save_filename = "Maxwell_PINN_Data_10_100.npz"
# np.savez_compressed(
#     save_filename,
#     t_original=t_array,
#     x_E_original=x_E,
#     x_H_original=x_H,
#     t_ds=t_ds,
#     x_E_ds=x_E_ds,
#     x_H_ds=x_H_ds,
#     E_exact=E_exact_clean,
#     H_exact=H_exact_clean,
#     E_full=E_history_all,
#     H_full=H_history_all
# )