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

def cmp_traj(traj, sf, T=100, bias=0, width=1.177):
    params = np.zeros((T, 2, 5)) # time, succ/fail, mean:x/mean:y/axis:long/axis:short/angle
    dist = np.zeros((T, 2)) # time, euclid/mahalanobis
    traj = traj[bias:]
    succ = np.where(sf[:, 1] == 1)[0]
    fail = np.where((sf[:, 0] == 1) & (sf[:, 1] == 0))[0]
    succ = succ[succ >= T - 1]
    fail = fail[fail >= T - 1]
    for ti in range(T):
        tj = T - 1 - ti
        succ_t = traj[succ - ti, :]
        c = np.cov(succ_t.T)
        params[tj, 0, :2] = succ_t.mean(axis=0)
        params[tj, 0, 2:] = ellipse_params(c, width)
        fail_t = traj[fail - ti, :]
        c = np.cov(fail_t.T)
        params[tj, 1, :2] = fail_t.mean(axis=0)
        params[tj, 1, 2:] = ellipse_params(c, width)
        diff = params[tj, 0, :2] - params[tj, 1, :2]
        dist[tj, 1] = (np.dot(diff, np.linalg.inv(c)) * diff).sum(axis=-1)
    dist[:, 0] = np.sqrt(np.square(params[:, 0, :2] - params[:, 1, :2]).sum(axis=-1))
    return params, dist