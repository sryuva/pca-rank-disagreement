"""Three pre-write-up checks."""
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
RES, FIG = ROOT / "results", ROOT / "figures"
import numpy as np, inspect, experiment as E

print("=== CHECK 1: selection rules ===")
print("stability_rank arguments:", list(inspect.signature(E.stability_rank).parameters))
print("cv_rank arguments       :", list(inspect.signature(E.cv_rank).parameters))
# Empirical: stability choice must be identical if y is replaced by noise (same RNG stream).
same, total = 0, 300
for s in range(total):
    lam = E.eigenvalues(3.0)
    rng0 = np.random.default_rng(s); Q, _ = np.linalg.qr(rng0.standard_normal((6, 6)))
    (X, y), _ = E.generate(60, lam, 5, rng0, Q)
    k1 = E.stability_rank(X, np.random.default_rng(10_000 + s))
    k2 = E.stability_rank(X.copy(), np.random.default_rng(10_000 + s))  # y never passed
    same += k1 == k2
print(f"stability rank identical without y: {same}/{total}")
# CV leakage test: CV only ever sees (X, y) of the training sample; check via a pure-noise target.
# If test data leaked, CV would pick large k and beat the null model; it should prefer small k.
ks = []
for s in range(200):
    rng = np.random.default_rng(s)
    X = rng.standard_normal((60, 6)); y = rng.standard_normal(60)
    ks.append(E.cv_rank(X, y, rng))
print("CV rank on pure-noise targets (should favour small k):", np.bincount(ks, minlength=7)[1:])

print("\n=== CHECK 2: does the ~1.0 excess MSE equal the signal variance SNR*sigma^2? ===")
def cell(snr, sigma, n=500, gap=10.0, j=5, seeds=100):
    E.SNR, E.SIGMA = snr, sigma
    lam = E.eigenvalues(gap); out = []
    for s in range(seeds):
        rng = np.random.default_rng([n, j, s, 7]); Q, _ = np.linalg.qr(rng.standard_normal((6, 6)))
        (X, y), (Xt, yt) = E.generate(n, lam, j, rng, Q)
        ks, kc = E.stability_rank(X, rng), E.cv_rank(X, y, rng)
        out.append((E.test_mse(X, y, Xt, yt, ks), E.test_mse(X, y, Xt, yt, kc), np.var(yt), ks))
    a = np.array(out)
    return a[:, 0].mean(), a[:, 1].mean(), a[:, 2].mean(), (a[:, 3] == 3).mean()
print(f"{'SNR':>4} {'sigma':>5} {'expected':>8} {'MSE_stab':>8} {'MSE_cv':>7} {'Var(y)':>7} {'excess':>7} {'k_stab=3':>8}")
for snr, sigma in [(1, 1), (0.5, 1), (2, 1), (1, 2), (0.25, 2)]:
    ms, mc, vy, f3 = cell(snr, sigma)
    print(f"{snr:4} {sigma:5} {snr*sigma**2:8.2f} {ms:8.3f} {mc:7.3f} {vy:7.3f} {ms-mc:7.3f} {f3:8.2f}")
E.SNR, E.SIGMA = 1.0, 1.0

print("\n=== CHECK 3: first-order approximation vs a formal Davis-Kahan bound ===")
# Yu, Wang & Samworth (2015) Davis-Kahan variant:
#   ||sin Theta||_F <= 2 min(sqrt(k)||E||_op, ||E||_F) / (lambda_k - lambda_{k+1}),
# and ||P_hat - P||_F = sqrt(2) ||sin Theta||_F.  Computed per sample with the true Sigma.
rows = []
for sep, g in E.SEPARATIONS.items():
    lam = E.eigenvalues(g); Sig = np.diag(lam); gap = lam[2] - lam[3]
    for n in [20, 30, 60, 120, 250, 500, 1000]:
        rng = np.random.default_rng([n, int(g * 10), 999])
        emp, dk, viol = [], [], 0
        for _ in range(500):
            X = rng.standard_normal((n, 6)) * np.sqrt(lam)
            S = np.cov(X, rowvar=False); Ed = S - Sig
            V = np.linalg.eigh(S)[1][:, ::-1][:, :3]
            d = np.linalg.norm(V @ V.T - np.diag([1, 1, 1, 0, 0, 0]), "fro")
            b = np.sqrt(2) * 2 * min(np.sqrt(3) * np.linalg.norm(Ed, 2), np.linalg.norm(Ed, "fro")) / gap
            emp.append(d**2); dk.append(min(b, np.sqrt(6))); viol += d > b + 1e-12
        rows.append((sep, n, np.sqrt(np.mean(emp)), np.sqrt(np.mean(np.square(dk))), viol))
import pandas as pd
dk = pd.DataFrame(rows, columns=["separation", "n", "empirical", "davis_kahan_bound", "violations"])
sd = pd.read_csv(RES / "subspace_distance.csv").merge(dk.drop(columns="empirical"), on=["separation", "n"])
sd.to_csv(RES / "subspace_distance.csv", index=False)
print(sd.round(3).to_string(index=False))
