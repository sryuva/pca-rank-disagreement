"""
When do stability-based and prediction-based PCA rank selection disagree?

Stability-based : Besse (1992)-style bootstrap instability of the k-dim principal
                  subspace; choose k in 1..p-1 with the lowest instability
                  (k = p is excluded: the full-space projector cannot move).
Prediction-based: 5-fold cross-validation of principal component regression (PCR)
                  test error; choose k in 1..p.

Controlled variables: sample size n, eigenvalue separation (size of the eigenvalue gap at index M = 3),
and the principal component j that carries the predictive signal.
Signal-to-noise ratio of y is held fixed at SNR = Var(X beta) / sigma^2 = 1.
"""
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
RES, FIG = ROOT / "results", ROOT / "figures"
import numpy as np, pandas as pd, itertools, time

P = 6
SNR = 1.0
SIGMA = 1.0
N_TEST = 2000
B_BOOT = 50
K_FOLDS = 5
SEEDS = 200
NS = [30, 60, 120, 500]
M = 3                                   # true structural rank (location of the gap)
SEPARATIONS = {"small": 1.5, "medium": 3.0, "large": 10.0}   # lambda_3 / lambda_4
SIGNAL_PCS = range(1, P + 1)


def eigenvalues(g):
    """Spiked spectrum: three leading eigenvalues, a gap of size g at index M, three trailing.
    Mild decay inside each block keeps every PC identifiable."""
    lam = np.r_[g * np.array([1.2, 1.1, 1.0]), np.array([1.0, 0.9, 0.8])]
    return lam * P / lam.sum()            # total variance = P for every level


def generate(n, lam, j, rng, Q):
    """X ~ N(0, Q diag(lam) Q^T); y = X beta + noise, beta along the j-th PC."""
    c = np.sqrt(SNR * SIGMA**2 / lam[j - 1])   # Var(X beta) = c^2 lam_j = SNR*sigma^2
    beta = c * Q[:, j - 1]
    def draw(m):
        X = rng.standard_normal((m, P)) * np.sqrt(lam) @ Q.T
        y = X @ beta + SIGMA * rng.standard_normal(m)
        return X, y
    return draw(n), draw(N_TEST)


def pcs(X):
    Xc = X - X.mean(0)
    w, V = np.linalg.eigh(Xc.T @ Xc / (len(X) - 1))
    return V[:, ::-1]                      # columns sorted by decreasing variance


def projector(V, k):
    return V[:, :k] @ V[:, :k].T


def stability_rank(X, rng):
    """Mean squared projector distance between full-sample and bootstrap subspaces."""
    V = pcs(X)
    inst = np.zeros(P - 1)
    for _ in range(B_BOOT):
        Vb = pcs(X[rng.integers(0, len(X), len(X))])
        for k in range(1, P):
            inst[k - 1] += np.sum((projector(V, k) - projector(Vb, k)) ** 2) / 2
    return int(np.argmin(inst)) + 1


def pcr_fit_predict(Xtr, ytr, Xte, k):
    mu, ym = Xtr.mean(0), ytr.mean()
    V = pcs(Xtr)[:, :k]
    Ztr = (Xtr - mu) @ V
    gamma = np.linalg.lstsq(Ztr, ytr - ym, rcond=None)[0]
    return ym + (Xte - mu) @ V @ gamma


def cv_rank(X, y, rng):
    folds = np.array_split(rng.permutation(len(X)), K_FOLDS)
    err = np.zeros(P)
    for f in folds:
        tr = np.setdiff1d(np.arange(len(X)), f)
        for k in range(1, P + 1):
            err[k - 1] += np.sum((y[f] - pcr_fit_predict(X[tr], y[tr], X[f], k)) ** 2)
    return int(np.argmin(err)) + 1


def test_mse(X, y, Xt, yt, k):
    return np.mean((yt - pcr_fit_predict(X, y, Xt, k)) ** 2)


def run(out_name="results.csv", seed_scale=10):
    rows = []
    t0 = time.time()
    for n, (sep, r), j in itertools.product(NS, SEPARATIONS.items(), SIGNAL_PCS):
        lam = eigenvalues(r)
        for s in range(SEEDS):
            rng = np.random.default_rng([n, int(round(r * seed_scale)), j, s])
            Q, _ = np.linalg.qr(rng.standard_normal((P, P)))
            (X, y), (Xt, yt) = generate(n, lam, j, rng, Q)
            k_s = stability_rank(X, rng)
            k_c = cv_rank(X, y, rng)
            e_s, e_c = test_mse(X, y, Xt, yt, k_s), test_mse(X, y, Xt, yt, k_c)
            rows.append(dict(n=n, separation=sep, ratio=r, signal_pc=j, seed=s,
                             k_stability=k_s, k_cv=k_c, disagree=int(k_s != k_c),
                             mse_stability=e_s, mse_cv=e_c, delta_mse=e_s - e_c))
        print(f"n={n:4d} sep={sep:6s} j={j}  done  ({time.time()-t0:.0f}s)", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(RES / out_name, index=False)
    return df


def subspace_distance_study():
    """Graph 3 data: empirical distance of the top-M subspace vs n, against the
    first-order perturbation prediction E||P_hat - P||_F^2 ~ (2/n) sum_{i<=M<j} li lj/(li-lj)^2."""
    rows = []
    for (sep, r), n in itertools.product(SEPARATIONS.items(), [20, 30, 60, 120, 250, 500, 1000]):
        lam = eigenvalues(r)
        rng = np.random.default_rng([n, int(r * 10), 999])
        Q = np.eye(P)
        d = []
        for _ in range(500):
            X = rng.standard_normal((n, P)) * np.sqrt(lam)
            V = pcs(X)
            d.append(np.sum((projector(V, M) - projector(Q, M)) ** 2))
        pred = 2 / n * sum(lam[i] * lam[k] / (lam[i] - lam[k]) ** 2 for i in range(M) for k in range(M, P))
        rows.append(dict(separation=sep, n=n, empirical=np.sqrt(np.mean(d)),
                         predicted=np.sqrt(pred)))
    df = pd.DataFrame(rows)
    df.to_csv(RES / "subspace_distance.csv", index=False)
    return df


def geometric_eigenvalues(r):
    """Pilot spectrum (Section 4.1): geometric decay, no preferred gap."""
    lam = r ** np.arange(P)
    return lam * P / lam.sum()


if __name__ == "__main__":
    import sys
    if "--pilot" in sys.argv:
        # Section 4.1 pilot: geometric spectrum, ratio lambda_{k+1}/lambda_k
        eigenvalues = geometric_eigenvalues
        SEPARATIONS = {"small": 0.85, "medium": 0.6, "large": 0.35}
        run("results_geometric_pilot.csv", seed_scale=100)
    else:
        run()
        subspace_distance_study()
