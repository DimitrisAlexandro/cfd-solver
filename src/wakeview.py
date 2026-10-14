"""Wake analysis for the results viewer: what the flow does behind the body.

2D (Wake2D, on a postview.Case2D)
  rake(d)          a virtual wake rake d chords behind the trailing edge, normal to the
                   free stream: u/U, v/U, total-pressure loss
  survey(ds)       wake drag at each station (momentum / total-pressure integral)

3D (Wake3D, on a postview.Case3D)
  plane(d)         a plane normal to the free stream d chords behind the trailing edge
  survey(ds)       the wake-survey drag split of Betz / Maskell:
                     profile drag  = integral of the total-pressure loss
                     induced drag  = crossflow kinetic energy = 1/2 integral of psi * xi
                   (xi: streamwise vorticity, psi: crossflow stream function with the
                   symmetry plane as a mirror), the shed circulation and the tip vortex
  plan(...)        top view of the wake (a slice, or the maximum through its thickness)

Units as the solvers: rho = 1, |U_inf| = 1 for 3D, p_inf = 0 (pt_inf from the rake ends
for 2D, where inlet/outlet domains do not fix p_inf).

Exact relation (control volume, lateral sides far away, viscous stress on the plane
neglected):  D = int (pt_inf - pt) dS + 1/2 int (v^2 + w^2 - (U - u)^2) dS
"""
import math

import numpy as np


def _axes(alpha_deg):
    a = math.radians(alpha_deg)
    es = np.array([math.cos(a), math.sin(a), 0.0])          # along the free stream
    en = np.array([-math.sin(a), math.cos(a), 0.0])         # normal to it, "up"
    return es, en


def _stretched(half, n, k=3.0):
    """points on [-half, half] clustered at 0, and their integration weights"""
    s = np.linspace(-1.0, 1.0, n)
    t = half * np.sinh(k * s) / math.sinh(k)
    w = np.gradient(t)
    return t, w


# ======================================================================
# 2D
# ======================================================================
class Wake2D:
    def __init__(self, case):
        from scipy.interpolate import LinearNDInterpolator
        self.case = case
        m = case.meta
        Ui = m.get("U_inf")
        if Ui is None:
            a = math.radians(float(m.get("alpha_deg", 0.0)))
            Ui = [math.cos(a), math.sin(a)]
        self.q = float(np.hypot(*Ui[:2]))
        self.alpha = math.degrees(math.atan2(Ui[1], Ui[0]))
        es, en = _axes(self.alpha)
        self.es, self.en = es[:2], en[:2]
        walls = case.walls()
        if walls:
            w = next(iter(walls.values()))
            i = int(np.argmax(w["x"]))
            self.te = np.array([w["x"][i], w["y"][i]])
            self.chord = float(w["x"].max() - w["x"].min())
        else:
            lo, hi = case.extent()
            self.te = np.array([hi[0], 0.5 * (lo[1] + hi[1])])
            self.chord = float(hi[0] - lo[0])
        self.Cd_surface = (m.get("coefficients") or m.get("coefficients_final") or {}).get("Cd")
        # interpolate only in the wake region (a Delaunay of the whole far field is slow)
        c, x = case.centres, case.centres - self.te
        box = (x[:, 0] > -0.15 * self.chord) & (x[:, 0] < 9.0 * self.chord) & (np.abs(x[:, 1]) < 4.0 * self.chord)
        U = case.data["U"][box, :2]
        self._f = LinearNDInterpolator(c[box], np.c_[U, case.data["p"][box]])

    def rake(self, d, half=None, n=801):
        """profile on a line normal to the free stream, d chords behind the trailing edge"""
        c = self.chord
        half = half or c * (0.6 + 0.25 * d)
        t, wt = _stretched(half, n)
        pts = self.te + d * c * self.es + t[:, None] * self.en
        v = self._f(pts)
        ok = np.isfinite(v).all(1)
        t, wt, v, pts = t[ok], wt[ok], v[ok], pts[ok]
        u_s = v[:, :2] @ self.es
        v_n = v[:, :2] @ self.en
        pt = v[:, 2] + 0.5 * (v[:, 0] ** 2 + v[:, 1] ** 2)
        k = max(len(t) // 25, 3)
        pt_inf = 0.5 * (np.median(pt[:k]) + np.median(pt[-k:]))
        q2 = 0.5 * self.q ** 2
        loss = (pt_inf - pt) / q2
        return {"d": d, "t": t / c, "w": wt, "pts": pts, "u": u_s / self.q, "v": v_n / self.q, "loss": loss}

    @staticmethod
    def _wake_region(loss, frac=0.005):
        """the contiguous band around the peak loss where loss > frac * peak"""
        i = int(np.argmax(loss))
        th = frac * loss[i]
        a = i
        while a > 0 and loss[a - 1] > th:
            a -= 1
        b = i
        while b < len(loss) - 1 and loss[b + 1] > th:
            b += 1
        m = np.zeros(len(loss), bool)
        m[a:b + 1] = True
        return m

    def survey(self, ds=(0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0)):
        rows = []
        for d in ds:
            r = self.rake(d)
            if len(r["t"]) < 20:
                continue
            w = r["w"] / self.chord
            # Betz: only across the wake (where total pressure is lost), so the flow the
            # lift induces outside it does not enter; u* is the velocity the wake would
            # have without its total-pressure loss
            wake = self._wake_region(r["loss"])
            loss, u = r["loss"][wake], r["u"][wake]
            us = np.sqrt(u ** 2 + np.maximum(loss, 0.0))
            cd_pt = float(np.sum(loss * w[wake]))
            cd_betz = float(np.sum((loss + (us - u) * (us + u - 2.0)) * w[wake]))
            j = int(np.argmin(r["u"]))
            above = r["loss"] > 0.05 * r["loss"].max()
            rows.append({"d": d, "Cd_loss": cd_pt, "Cd_wake": cd_betz, "u_min": float(r["u"][j]),
                         "centre": float(r["t"][j]),
                         "width": float(np.ptp(r["t"][above])) if above.any() else 0.0})
        return rows


# ======================================================================
# 3D
# ======================================================================
class Wake3D:
    def __init__(self, case):
        from scipy.spatial import cKDTree
        self.case = case
        self.info = case.info
        self.alpha = float(case.info["alpha"])
        self.es, self.en = _axes(self.alpha)
        self.s = float(case.info["semi_span"])
        self.S = float(case.info.get("S_half") or self.s)
        cen = case.d["centres"].astype(float)
        self.tree = getattr(case, "_tree", None) or cKDTree(cen)
        case._tree = self.tree
        w = case.d["wall_centres"].astype(float)
        i = int(np.argmax(w[:, 0]))
        self.te = w[i].copy()                       # the most aft trailing-edge point
        self.te[2] = 0.0
        root = w[w[:, 2] < 0.1 * self.s]
        self.te[1] = float(root[np.argmax(root[:, 0]), 1]) if len(root) else 0.0
        self.vort = case.d["vorticity"].astype(float) if "vorticity" in case.d else None

    # -- sampling: inverse-distance average of the 4 nearest cells -----------------------
    def sample(self, q):
        dist, j = self.tree.query(q, k=4)
        w = 1.0 / np.maximum(dist, 1e-9)
        w /= w.sum(1, keepdims=True)
        U = (self.case.U[j] * w[..., None]).sum(1)
        p = (self.case.p[j] * w).sum(1)
        nut = (self.case.nut[j] * w).sum(1) / self.case.nu
        om = (self.vort[j] * w[..., None]).sum(1) if self.vort is not None else np.zeros_like(U)
        u = U @ self.es
        return {"u": u, "v": U @ self.en, "w": U[:, 2], "p": p,
                "loss": 1.0 - (2 * p + (U ** 2).sum(1)),          # (pt_inf - pt) / q_inf
                "deficit": 1.0 - u, "xi": om @ self.es, "nut_ratio": nut, "Cp": 2 * p}

    def origin(self, d):
        return self.te + d * self.es

    # -- a plane normal to the free stream ------------------------------------------------
    def plane(self, d, nt=181, nz=241, half=None, zmax=None):
        half = half or (0.5 + 0.12 * d)
        zmax = zmax or 1.5 * self.s
        t, wt = _stretched(half, nt, k=2.0)
        t = t + self._centre(d)
        z = np.linspace(0.0, zmax, nz)
        wz = np.gradient(z)
        wz[0] *= 0.5
        T, Z = np.meshgrid(t, z, indexing="ij")
        q = self.origin(d) + T.ravel()[:, None] * self.en + Z.ravel()[:, None] * np.array([0, 0, 1.0])
        f = {k: v.reshape(T.shape) for k, v in self.sample(q).items()}
        f.update(t=T, z=Z, dA=np.outer(wt, wz), xyz=q.reshape(T.shape + (3,)))
        # streamwise vorticity from the in-plane velocities (its area integral is the
        # circulation around the plane, so it conserves the shed circulation; the cell
        # gradients are noisy in the coarse far-wake cells)
        f["xi_p"] = np.gradient(f["w"], t, axis=0) - np.gradient(f["v"], z, axis=1)
        return f

    def _centre(self, d):
        """the wake sheet drifts down with the downwash: follow it at mid span"""
        t, _ = _stretched(0.6 + 0.12 * d, 241, k=2.0)
        q = self.origin(d) + t[:, None] * self.en + np.array([0, 0, 0.5 * self.s])
        loss = self.sample(q)["loss"]
        return float(t[np.argmax(loss)]) if np.isfinite(loss).all() and loss.max() > 1e-3 else 0.0

    # -- induced drag from the vorticity in the plane (Maskell) ---------------------------
    @staticmethod
    def _maskell(T, Z, xi, dA, nmax=6000):
        """1/2 int psi xi dA over the half plane, psi from xi with a mirror at z = 0"""
        a = np.abs(xi * dA).ravel()
        o = np.argsort(-a)
        c = np.cumsum(a[o])
        if c[-1] <= 0:
            return 0.0
        n = min(int(np.searchsorted(c, 0.995 * c[-1])) + 1, nmax)
        idx = o[:n]
        t, z = T.ravel()[idx], Z.ravel()[idx]
        g = (xi * dA).ravel()[idx]                          # circulation of each sample
        r_self = np.sqrt(dA.ravel()[idx] / math.pi) * math.exp(-0.5)
        psi = np.zeros(n)
        for s0 in range(0, n, 1500):
            sl = slice(s0, s0 + 1500)
            dt = t[sl, None] - t[None, :]
            r2 = dt ** 2 + (z[sl, None] - z[None, :]) ** 2
            ri2 = dt ** 2 + (z[sl, None] + z[None, :]) ** 2
            k = np.arange(s0, min(s0 + 1500, n))
            r2[k - s0, k] = r_self[k] ** 2
            psi[sl] = -(1 / (4 * math.pi)) * ((np.log(r2) - np.log(ri2)) * g[None, :]).sum(1)
        return 0.5 * float(np.sum(psi * g))

    def survey(self, ds=(0.25, 0.5, 1.0, 2.0, 4.0)):
        """wake drag split, circulation and tip vortex at each station"""
        rows = []
        for d in ds:
            f = self.plane(d)
            dA = f["dA"]
            Dp = float(np.sum(f["loss"] * dA) * 0.5)              # int (pt_inf - pt) dS (q_inf = 1/2)
            Dx = float(-0.5 * np.sum(f["deficit"] ** 2 * dA))
            Di = self._maskell(f["t"], f["z"], f["xi_p"], dA)
            G = float(np.sum(f["xi_p"] * dA))                     # shed circulation (half wing)
            core = self.core(f, G, key="xi")
            loss_z = (f["loss"] * dA).sum(0) / np.gradient(f["z"][0])   # int loss dt, per z
            rows.append({"d": d, "x": float(self.origin(d)[0]), "CDp": (Dp + Dx) / (0.5 * self.S),
                         "CDi": Di / (0.5 * self.S), "CD": (Dp + Dx + Di) / (0.5 * self.S),
                         "Gamma": G, **core, "z": f["z"][0].tolist(), "cd_c": loss_z.tolist()})
        return rows

    def core(self, f, G=None, key="xi"):
        """tip-vortex core: the extreme of the streamwise vorticity with the sign of the
        shed circulation; its centroid, peak and radius (second moment)"""
        xi = f[key]
        sg = -1.0 if (G if G is not None else np.sum(xi * f["dA"])) < 0 else 1.0
        x = sg * xi
        i = np.unravel_index(int(np.nanargmax(x)), x.shape)
        pk = x[i]
        m = x > 0.3 * pk
        tc = float(np.sum((x * f["t"] * f["dA"])[m]) / np.sum((x * f["dA"])[m]))
        zc = float(np.sum((x * f["z"] * f["dA"])[m]) / np.sum((x * f["dA"])[m]))
        r2 = (f["t"] - tc) ** 2 + (f["z"] - zc) ** 2
        rc = float(np.sqrt(np.sum((x * r2 * f["dA"])[m]) / np.sum((x * f["dA"])[m])))
        o = f["xyz"][0, 0] - f["t"][0, 0] * self.en          # the plane's origin (t = 0, z = 0)
        c = o + tc * self.en
        return {"core_t": tc, "core_z": zc, "core_x": float(c[0]), "core_y": float(c[1]),
                "peak_xi": float(pk), "core_r": rc, "core_Gamma": float(np.sum((x * f["dA"])[m]))}

    # -- plan view ------------------------------------------------------------------------
    def plan(self, field="loss", x1=6.0, offset=0.0, mode="max", ns=241, nz=161, nt=25, thick=None):
        """top view on the surface through the trailing edge along the free stream.
        mode 'slice': the field at a normal offset; 'max': the largest value through the
        wake thickness (+-thick), i.e. where the wake is seen from above"""
        x0 = float(self.case.d["wall_centres"][:, 0].min()) - 0.3
        s = np.linspace(x0 - self.te[0], x1 - self.te[0], ns) / self.es[0]
        z = np.linspace(0.0, 1.5 * self.s, nz)
        S, Z = np.meshgrid(s, z, indexing="ij")
        base = self.te + S.ravel()[:, None] * self.es + Z.ravel()[:, None] * np.array([0, 0, 1.0])
        if mode == "slice":
            q = base + offset * self.en
            v = self.sample(q)[field]
            solid = self.case._solid_mask(q)
            v = np.where(solid, np.nan, v).reshape(S.shape)
        else:
            thick = thick or 0.35
            ts = np.linspace(-thick, thick, nt) + offset
            vals = []
            for tt in ts:
                q = base + tt * self.en
                vv = self.sample(q)[field]
                vals.append(np.where(self.case._solid_mask(q), np.nan, vv))
            V = np.array(vals)
            if field in ("xi",):
                k = np.nanargmax(np.abs(V), axis=0)
                v = np.take_along_axis(V, k[None], 0)[0]
            else:
                v = np.nanmax(V, axis=0)
            v = v.reshape(S.shape)
        X = (self.te + S.ravel()[:, None] * self.es)[:, 0].reshape(S.shape)
        zp, le, te = planform(self.case)                    # the wing itself is not the wake
        over = (Z <= self.s) & (X >= np.interp(Z, zp, le) - 0.01) & (X <= np.interp(Z, zp, te) + 0.005)
        return X, Z, np.where(over, np.nan, v)


# ======================================================================
# figures
# ======================================================================
LABELS = {"loss": "total-pressure loss (pt∞ − pt)/q∞", "deficit": "velocity deficit 1 − u/U∞",
          "xi": "streamwise vorticity ξ c/U∞", "nut_ratio": "ν_t / ν", "Cp": "Cp"}


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def planform(case, nb=40):
    """leading/trailing edge lines (x vs z) from the wall points"""
    w = case.d["wall_centres"].astype(float)
    zb = np.linspace(0, w[:, 2].max(), nb + 1)
    z, le, te = [], [], []
    for a, b in zip(zb[:-1], zb[1:]):
        m = (w[:, 2] >= a) & (w[:, 2] <= b)
        if m.any():
            z.append(0.5 * (a + b))
            le.append(w[m, 0].min())
            te.append(w[m, 0].max())
    z = np.array(z)
    z[0], z[-1] = 0.0, zb[-1]                   # the outline runs from the root to the tip
    return z, np.array(le), np.array(te)


def plan_figure(wk, field="loss", x1=6.0, offset=0.0, mode="max"):
    plt = _plt()
    X, Z, v = wk.plan(field, x1, offset, mode)
    fig, ax = plt.subplots(figsize=(12, 5))
    if field == "xi":
        m = np.nanpercentile(np.abs(v), 99.5) or 1.0
        im = ax.pcolormesh(X, Z, v, cmap="RdBu_r", vmin=-m, vmax=m, shading="auto")
    else:
        lo, hi = np.nanpercentile(v, [1, 99.5])
        lo = 0.0 if field in ("loss", "deficit", "nut_ratio") else lo
        im = ax.pcolormesh(X, Z, v, cmap="magma_r" if field in ("loss", "deficit") else "turbo",
                           vmin=lo, vmax=max(hi, lo + 1e-9), shading="auto")
    fig.colorbar(im, ax=ax, label=LABELS.get(field, field), shrink=0.85, pad=0.015)
    z, le, te = planform(wk.case)
    ax.fill(np.r_[le, te[::-1]], np.r_[z, z[::-1]], color="#b9bec6", ec="k", lw=1.0)
    ax.axhline(wk.s, color="k", lw=0.5, ls=":")
    ax.set_xlabel("x/c (stream)")
    ax.set_ylabel("z/c (span)")
    ax.set_aspect("equal")
    ttl = "maximum through the wake thickness" if mode == "max" else f"slice {offset:+.2f} c from the trailing-edge surface"
    ax.set_title(f"Wake seen from above: {LABELS.get(field, field)} ({ttl})")
    fig.tight_layout()
    return fig


def planes_figure(wk, ds=(0.5, 1.5, 3.0, 5.0), field="loss", height=640, cut=0.06):
    """the wing and planes normal to the free stream behind it, wake only (values above
    a threshold), coloured by field"""
    import postview as V
    fig = V.surface_figure(wk.case, "Cp", height=height)
    fig.data[0].update(showscale=False, opacity=1.0)
    planes = [wk.plane(d, nt=121, nz=161, half=0.6 + 0.15 * d) for d in ds]
    vals = [p[field] for p in planes]
    if field == "xi":
        m = max(np.nanpercentile(np.abs(v), 99.5) for v in vals) or 1.0
        lo, hi, cs = -m, m, "RdBu_r"
    else:
        hi = max(np.nanpercentile(v, 99.5) for v in vals) or 1.0
        lo, cs = 0.0, "Magma_r" if field in ("loss", "deficit") else "Turbo"
    for k, (p, v) in enumerate(zip(planes, vals)):
        hole = ~(np.abs(v) > cut * max(abs(lo), abs(hi)))
        v = np.where(hole, np.nan, v)
        P = np.where(hole[..., None], np.nan, p["xyz"])          # holes where there is no wake
        fig.add_surface(x=P[..., 0], y=P[..., 2], z=P[..., 1], surfacecolor=v, colorscale=cs, cmin=lo, cmax=hi,
                        showscale=k == 0, colorbar=dict(title=field, x=1.02), opacity=0.95,
                        hovertemplate=f"x {p['xyz'][0, 0, 0]:.2f}<br>{field} %{{surfacecolor:.3g}}<extra></extra>")
    fig.update_layout(title=f"Wake planes behind the wing: {LABELS.get(field, field)}")
    return fig
