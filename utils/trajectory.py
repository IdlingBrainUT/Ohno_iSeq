import numpy as np

def ellipse_params(c, width=1.177):
    a = np.diag(c).sum()
    b = np.sqrt(np.square(c[0, 0] - c[1, 1]) + 4 * np.square(c[0, 1]))
    u2 = (a + b) / 2
    v2 = (a - b) / 2
    cu = width * np.sqrt(u2)
    cv = width * np.sqrt(v2)
    theta = np.arctan((u2 - c[0, 0]) / c[0, 1]) * 180 / np.pi
    return [cu, cv, theta]

def pick_succ_fail(traj, lick, reward, T=100):
    succ = np.where(reward == 1)[0]
    fail = np.where((lick == 1) & (reward == 0))[0]
    succ = succ[succ >= T - 1]
    fail = fail[fail >= T - 1]
    succ_trial = np.array([traj[i - T + 1 : i + 1] for i in succ])
    fail_trial = np.array([traj[i - T + 1 : i + 1] for i in fail])
    return succ_trial, fail_trial

def cmp_traj_succ_fail(succ_traj, fail_traj, T=100, width=1.177):
    params = np.zeros((T, 2, 5)) # time, succ/fail, mean:x/mean:y/axis:long/axis:short/angle
    dist = np.zeros((T, 2)) # time, euclid/mahalanobis
    for ti in range(T):
        succ_t = succ_traj[:, ti, :]
        c = np.cov(succ_t.T)
        params[ti, 0, :2] = succ_t.mean(axis=0)
        params[ti, 0, 2:] = ellipse_params(c, width)
        fail_t = fail_traj[:, ti, :]
        c = np.cov(fail_t.T)
        params[ti, 1, :2] = fail_t.mean(axis=0)
        params[ti, 1, 2:] = ellipse_params(c, width)
        diff = params[ti, 0, :2] - params[ti, 1, :2]
        dist[ti, 1] = (np.dot(diff, np.linalg.inv(c)) * diff).sum(axis=-1)
    dist[:, 0] = np.sqrt(np.square(params[:, 0, :2] - params[:, 1, :2]).sum(axis=-1))
    return params, dist

def cmp_traj_succ_fail_mahal_pool(succ_traj, fail_traj, T=100, width=1.177):
    params = np.zeros((T, 2, 5)) # time, succ/fail, mean:x/mean:y/axis:long/axis:short/angle
    dist = np.zeros((T, 2)) # time, euclid/mahalanobis
    for ti in range(T):
        succ_t = succ_traj[:, ti, :]
        n_s, _ = succ_t.shape
        c_s = np.cov(succ_t.T)
        params[ti, 0, :2] = succ_t.mean(axis=0)
        params[ti, 0, 2:] = ellipse_params(c_s, width)
        fail_t = fail_traj[:, ti, :]
        n_f, _ = fail_t.shape
        c_f = np.cov(fail_t.T)
        params[ti, 1, :2] = fail_t.mean(axis=0)
        params[ti, 1, 2:] = ellipse_params(c_f, width)
        diff = params[ti, 0, :2] - params[ti, 1, :2]
        dist[ti, 1] = (np.dot(diff, np.linalg.inv(((n_s - 1) * c_s + (n_f - 1) * c_f) / (n_s + n_f - 2))) * diff).sum(axis=-1)
    dist[:, 0] = np.sqrt(np.square(params[:, 0, :2] - params[:, 1, :2]).sum(axis=-1))
    return params, dist

def compute_procrustes(x, ref):
    x0 = x - x.mean(axis=0)
    r0 = ref - ref.mean(axis=0)
    norm_x = np.linalg.norm(x0)
    norm_r = np.linalg.norm(r0)
    x0 /= norm_x
    r0 /= norm_r
    U, _, Vt = np.linalg.svd(x0.T @ r0)
    R = Vt.T @ U.T
    s = norm_r / norm_x
    t = ref.mean(axis=0) - s * x.mean(axis=0) @ R
    return R, s, t

def transform_procrustes(X, R, s, t):
    return np.array([s * x @ R + t for x in X])