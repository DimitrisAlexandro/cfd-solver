// AeroLab 3D native kernels (C ABI, loaded with ctypes by fv3d/native.py).
//
//   mom_assemble   momentum matrix + sources in two parallel passes (faces, then cells):
//                  upwind convection + deferred correction to (limited) linear upwind,
//                  over-relaxed diffusion with explicit non-orthogonal correction, the
//                  transpose viscous stress. Same arithmetic as the 2D numba kernel.
//   venkat         Venkatakrishnan limiter per cell
//   csr_fill       values of the fixed face-based CSR pattern
//   sgs            hybrid symmetric Gauss-Seidel (OpenMP row blocks; exact SGS on 1 thread),
//                  several right-hand sides, residual-based stop
//   amg_*          algebraic multigrid hierarchy (built by pyamg in Python, copied here) and
//                  conjugate gradients preconditioned with one V-cycle per iteration
//   sa_source      Spalart-Allmaras source and its cell-local derivative
//
// Build: g++ -O3 -march=native -fopenmp -shared -fPIC -std=c++17 aero3d.cpp -o libaero3d.so
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#else
static inline int omp_get_max_threads() { return 1; }
static inline int omp_get_thread_num() { return 0; }
static inline void omp_set_num_threads(int) {}
#endif

typedef int64_t idx;

extern "C" {

int aero_threads(int n) {
    if (n > 0) omp_set_num_threads(n);
    return omp_get_max_threads();
}

// ---------------------------------------------------------------------------
// momentum assembly
// gU, gk: (3 components, n cells, 3 directions) contiguous
// cptr/cface/csign: internal faces of each cell (+1 owner, -1 neighbour)
// scratch: nf * 5
// ---------------------------------------------------------------------------
void mom_assemble(idx n, idx nf, const idx* own, const idx* nei, const idx* cptr, const idx* cface,
                  const int8_t* csign, const double* F, const double* nuf, const double* coef,
                  const double* w, const double* Sf, const double* Delta, const double* xc,
                  const double* xf, const double* gU, const double* gk, int second_order,
                  int transpose, double* diag, double* a_on, double* a_no, double* src,
                  double* scratch) {
    const idx n3 = n * 3;
#pragma omp parallel for schedule(static)
    for (idx f = 0; f < nf; ++f) {
        const idx a = own[f], b = nei[f];
        const double D = nuf[f] * coef[f];
        const double Fp = F[f] > 0.0 ? F[f] : 0.0;
        const double Fm = F[f] < 0.0 ? -F[f] : 0.0;
        a_on[f] = -D - Fm;
        a_no[f] = -D - Fp;
        const double wf = w[f];
        double s[3], S[3], gf[3][3];
        for (int j = 0; j < 3; ++j) {
            S[j] = Sf[f * 3 + j];
            s[j] = S[j] - Delta[f * 3 + j];
        }
        for (int k = 0; k < 3; ++k)
            for (int j = 0; j < 3; ++j)
                gf[k][j] = wf * gU[k * n3 + a * 3 + j] + (1.0 - wf) * gU[k * n3 + b * 3 + j];
        const idx u = F[f] >= 0.0 ? a : b;
        double dx[3];
        for (int j = 0; j < 3; ++j) dx[j] = xf[f * 3 + j] - xc[u * 3 + j];
        for (int k = 0; k < 3; ++k) {
            double c = nuf[f] * (gf[k][0] * s[0] + gf[k][1] * s[1] + gf[k][2] * s[2]);
            if (transpose) c += nuf[f] * (gf[0][k] * S[0] + gf[1][k] * S[1] + gf[2][k] * S[2]);
            if (second_order)
                c -= F[f] * (gk[k * n3 + u * 3 + 0] * dx[0] + gk[k * n3 + u * 3 + 1] * dx[1] +
                             gk[k * n3 + u * 3 + 2] * dx[2]);
            scratch[f * 5 + k] = c;
        }
        scratch[f * 5 + 3] = D + Fp;
        scratch[f * 5 + 4] = D + Fm;
    }
#pragma omp parallel for schedule(static)
    for (idx c = 0; c < n; ++c) {
        double d = 0.0, s0 = 0.0, s1 = 0.0, s2 = 0.0;
        for (idx jj = cptr[c]; jj < cptr[c + 1]; ++jj) {
            const idx f = cface[jj];
            const double* q = scratch + f * 5;
            if (csign[jj] > 0) {
                d += q[3]; s0 += q[0]; s1 += q[1]; s2 += q[2];
            } else {
                d += q[4]; s0 -= q[0]; s1 -= q[1]; s2 -= q[2];
            }
        }
        diag[c] = d;
        src[c * 3 + 0] = s0;
        src[c * 3 + 1] = s1;
        src[c * 3 + 2] = s2;
    }
}

// ---------------------------------------------------------------------------
// Venkatakrishnan limiter (bptr/bface: boundary faces of each cell)
// ---------------------------------------------------------------------------
void venkat(idx n, const idx* own, const idx* nei, const idx* cptr, const idx* cface,
            const int8_t* csign, const idx* bptr, const idx* bface, const double* xc,
            const double* xf, const double* bxf, const double* phi, const double* g,
            const double* phib, double eps2, const uint8_t* mask, double* psi) {
#pragma omp parallel for schedule(static)
    for (idx c = 0; c < n; ++c) {
        if (mask && !mask[c]) {
            psi[c] = 1.0;
            continue;
        }
        double pmax = phi[c], pmin = phi[c];
        for (idx jj = cptr[c]; jj < cptr[c + 1]; ++jj) {
            const idx f = cface[jj];
            const double v = phi[csign[jj] > 0 ? nei[f] : own[f]];
            pmax = std::max(pmax, v);
            pmin = std::min(pmin, v);
        }
        for (idx jj = bptr[c]; jj < bptr[c + 1]; ++jj) {
            const double v = phib[bface[jj]];
            pmax = std::max(pmax, v);
            pmin = std::min(pmin, v);
        }
        double ps = 1.0;
        auto lim = [&](const double* x) {
            const double d2 = g[c * 3] * (x[0] - xc[c * 3]) + g[c * 3 + 1] * (x[1] - xc[c * 3 + 1]) +
                              g[c * 3 + 2] * (x[2] - xc[c * 3 + 2]);
            if (std::fabs(d2) <= 1e-30) return;
            const double d1 = d2 > 0 ? pmax - phi[c] : pmin - phi[c];
            const double num = (d1 * d1 + eps2) * d2 + 2.0 * d2 * d2 * d1;
            const double den = d2 * (d1 * d1 + 2.0 * d2 * d2 + d1 * d2 + eps2);
            double v = num / den;
            v = v < 0.0 ? 0.0 : (v > 1.0 ? 1.0 : v);
            ps = std::min(ps, v);
        };
        for (idx jj = cptr[c]; jj < cptr[c + 1]; ++jj) lim(xf + cface[jj] * 3);
        for (idx jj = bptr[c]; jj < bptr[c + 1]; ++jj) lim(bxf + bface[jj] * 3);
        psi[c] = ps;
    }
}

// ---------------------------------------------------------------------------
void csr_fill(idx n, idx nf, const idx* pd, const idx* pon, const idx* pno, const double* diag,
              const double* a_on, const double* a_no, double* data) {
#pragma omp parallel for schedule(static)
    for (idx i = 0; i < n; ++i) data[pd[i]] = diag[i];
#pragma omp parallel for schedule(static)
    for (idx f = 0; f < nf; ++f) {
        data[pon[f]] = a_on[f];
        data[pno[f]] = a_no[f];
    }
}

void spmv(idx n, const idx* ip, const idx* ix, const double* a, const double* x, double* y) {
#pragma omp parallel for schedule(static)
    for (idx i = 0; i < n; ++i) {
        double s = 0.0;
        for (idx jj = ip[i]; jj < ip[i + 1]; ++jj) s += a[jj] * x[ix[jj]];
        y[i] = s;
    }
}

}  // extern "C"

// ---------------------------------------------------------------------------
// hybrid Gauss-Seidel sweep: rows split in T contiguous blocks; inside a block the
// latest values, across blocks the values from the start of the sweep (deterministic)
// ---------------------------------------------------------------------------
static void gs_sweep(idx n, idx k, const idx* ip, const idx* ix, const double* a, const double* dinv,
                     double* X, const double* B, std::vector<double>& Xold, bool forward, int T) {
    if (T <= 1) {
        if (forward) {
            for (idx i = 0; i < n; ++i)
                for (idx c = 0; c < k; ++c) {
                    double s = B[i * k + c];
                    for (idx jj = ip[i]; jj < ip[i + 1]; ++jj)
                        if (ix[jj] != i) s -= a[jj] * X[ix[jj] * k + c];
                    X[i * k + c] = s * dinv[i];
                }
        } else {
            for (idx i = n - 1; i >= 0; --i)
                for (idx c = 0; c < k; ++c) {
                    double s = B[i * k + c];
                    for (idx jj = ip[i]; jj < ip[i + 1]; ++jj)
                        if (ix[jj] != i) s -= a[jj] * X[ix[jj] * k + c];
                    X[i * k + c] = s * dinv[i];
                }
        }
        return;
    }
    Xold.assign(X, X + n * k);
#pragma omp parallel num_threads(T)
    {
        const int t = omp_get_thread_num();
        const idx lo = n * t / T, hi = n * (t + 1) / T;
        if (forward) {
            for (idx i = lo; i < hi; ++i)
                for (idx c = 0; c < k; ++c) {
                    double s = B[i * k + c];
                    for (idx jj = ip[i]; jj < ip[i + 1]; ++jj) {
                        const idx j = ix[jj];
                        if (j == i) continue;
                        s -= a[jj] * ((j >= lo && j < hi) ? X[j * k + c] : Xold[j * k + c]);
                    }
                    X[i * k + c] = s * dinv[i];
                }
        } else {
            for (idx i = hi - 1; i >= lo; --i)
                for (idx c = 0; c < k; ++c) {
                    double s = B[i * k + c];
                    for (idx jj = ip[i]; jj < ip[i + 1]; ++jj) {
                        const idx j = ix[jj];
                        if (j == i) continue;
                        s -= a[jj] * ((j >= lo && j < hi) ? X[j * k + c] : Xold[j * k + c]);
                    }
                    X[i * k + c] = s * dinv[i];
                }
        }
    }
}

static void resnorm(idx n, idx k, const idx* ip, const idx* ix, const double* a, const double* X,
                    const double* B, double* out) {
    std::vector<double> acc(k, 0.0);
    for (idx c = 0; c < k; ++c) {
        double s2 = 0.0;
#pragma omp parallel for reduction(+ : s2) schedule(static)
        for (idx i = 0; i < n; ++i) {
            double s = B[i * k + c];
            for (idx jj = ip[i]; jj < ip[i + 1]; ++jj) s -= a[jj] * X[ix[jj] * k + c];
            s2 += s * s;
        }
        out[c] = std::sqrt(s2);
    }
}

static int threads_for(idx n) {
    const int T = omp_get_max_threads();
    if (n < 20000) return 1;
    return std::max(1, std::min<int>(T, (int)(n / 10000)));
}

extern "C" idx sgs(idx n, idx k, const idx* ip, const idx* ix, const double* a, double* X,
                   const double* B, double rtol, idx max_sweeps) {
    std::vector<double> dinv(n), Xold;
    for (idx i = 0; i < n; ++i) {
        double d = 0.0;
        for (idx jj = ip[i]; jj < ip[i + 1]; ++jj)
            if (ix[jj] == i) d = a[jj];
        dinv[i] = 1.0 / d;
    }
    std::vector<double> r0(k), r(k);
    resnorm(n, k, ip, ix, a, X, B, r0.data());
    bool done = true;
    for (idx c = 0; c < k; ++c)
        if (r0[c] > 0.0) done = false;
    if (done) return 0;
    const int T = threads_for(n);
    idx sweeps = 0;
    for (idx it = 1; it <= max_sweeps; ++it) {
        gs_sweep(n, k, ip, ix, a, dinv.data(), X, B, Xold, true, T);
        gs_sweep(n, k, ip, ix, a, dinv.data(), X, B, Xold, false, T);
        sweeps = it;
        if (it % 2 == 0 || it == 1) {
            resnorm(n, k, ip, ix, a, X, B, r.data());
            bool ok = true;
            for (idx c = 0; c < k; ++c)
                if (r[c] > rtol * r0[c]) ok = false;
            if (ok) break;
        }
    }
    return sweeps;
}

// ---------------------------------------------------------------------------
// algebraic multigrid: hierarchy storage + V-cycle preconditioned CG
// ---------------------------------------------------------------------------
struct Csr {
    idx n = 0, m = 0;
    std::vector<idx> ip, ix;
    std::vector<double> a;
    void set(idx n_, idx m_, const idx* ip_, const idx* ix_, const double* a_) {
        n = n_; m = m_;
        ip.assign(ip_, ip_ + n + 1);
        ix.assign(ix_, ix_ + ip[n]);
        a.assign(a_, a_ + ip[n]);
    }
    void mv(const double* x, double* y) const { spmv(n, ip.data(), ix.data(), a.data(), x, y); }
};

struct Level {
    Csr A, P, R;
    std::vector<double> dinv, x, b, r, Xold;
};

struct Amg {
    std::vector<Level> lev;
    idx nc = 0;
    std::vector<double> Cinv, cx, cb;   // dense inverse of the coarsest matrix
};

static void vcycle(Amg& h, size_t l, const double* b, double* x) {
    if (l == h.lev.size()) {            // coarsest: dense solve
        const idx m = h.nc;
        for (idx i = 0; i < m; ++i) {
            double s = 0.0;
            for (idx j = 0; j < m; ++j) s += h.Cinv[i * m + j] * b[j];
            x[i] = s;
        }
        return;
    }
    Level& L = h.lev[l];
    const idx n = L.A.n;
    const int T = threads_for(n);
    std::fill(x, x + n, 0.0);
    gs_sweep(n, 1, L.A.ip.data(), L.A.ix.data(), L.A.a.data(), L.dinv.data(), x, b, L.Xold, true, T);
    L.A.mv(x, L.r.data());
#pragma omp parallel for schedule(static)
    for (idx i = 0; i < n; ++i) L.r[i] = b[i] - L.r[i];
    const idx nn = L.R.n;
    double* bc = (l + 1 < h.lev.size()) ? h.lev[l + 1].b.data() : h.cb.data();
    double* xc = (l + 1 < h.lev.size()) ? h.lev[l + 1].x.data() : h.cx.data();
    L.R.mv(L.r.data(), bc);
    vcycle(h, l + 1, bc, xc);
    L.P.mv(xc, L.r.data());
#pragma omp parallel for schedule(static)
    for (idx i = 0; i < n; ++i) x[i] += L.r[i];
    (void)nn;
    gs_sweep(n, 1, L.A.ip.data(), L.A.ix.data(), L.A.a.data(), L.dinv.data(), x, b, L.Xold, false, T);
}

extern "C" {

void* amg_new() { return new Amg(); }
void amg_free(void* h) { delete static_cast<Amg*>(h); }

// one level: A (n x n), P (n x nc), R (nc x n)
void amg_add_level(void* hp, idx n, const idx* aip, const idx* aix, const double* aa, idx nc,
                   const idx* pip, const idx* pix, const double* pa, const idx* rip, const idx* rix,
                   const double* ra) {
    Amg& h = *static_cast<Amg*>(hp);
    h.lev.emplace_back();
    Level& L = h.lev.back();
    L.A.set(n, n, aip, aix, aa);
    L.P.set(n, nc, pip, pix, pa);
    L.R.set(nc, n, rip, rix, ra);
    L.dinv.resize(n);
    for (idx i = 0; i < n; ++i) {
        double d = 0.0;
        for (idx jj = L.A.ip[i]; jj < L.A.ip[i + 1]; ++jj)
            if (L.A.ix[jj] == i) d = L.A.a[jj];
        L.dinv[i] = d != 0.0 ? 1.0 / d : 0.0;
    }
    L.x.resize(n); L.b.resize(n); L.r.resize(n);
}

void amg_set_coarse(void* hp, idx m, const double* inv) {
    Amg& h = *static_cast<Amg*>(hp);
    h.nc = m;
    h.Cinv.assign(inv, inv + m * m);
    h.cx.resize(m); h.cb.resize(m);
}

// CG on the CURRENT matrix (ip, ix, a) preconditioned with the stored hierarchy.
// x: initial guess in, solution out. Returns iterations (negative if not converged).
idx amg_pcg(void* hp, idx n, const idx* ip, const idx* ix, const double* a, const double* b, double* x,
            double rtol, idx maxit) {
    Amg& h = *static_cast<Amg*>(hp);
    std::vector<double> r(n), z(n), p(n), q(n);
    spmv(n, ip, ix, a, x, q.data());
    double bn = 0.0;
#pragma omp parallel for reduction(+ : bn) schedule(static)
    for (idx i = 0; i < n; ++i) {
        r[i] = b[i] - q[i];
        bn += b[i] * b[i];
    }
    bn = std::sqrt(bn);
    if (bn == 0.0) {
        std::fill(x, x + n, 0.0);
        return 0;
    }
    vcycle(h, 0, r.data(), z.data());
    double rz = 0.0;
#pragma omp parallel for reduction(+ : rz) schedule(static)
    for (idx i = 0; i < n; ++i) {
        p[i] = z[i];
        rz += r[i] * z[i];
    }
    for (idx it = 1; it <= maxit; ++it) {
        spmv(n, ip, ix, a, p.data(), q.data());
        double pq = 0.0;
#pragma omp parallel for reduction(+ : pq) schedule(static)
        for (idx i = 0; i < n; ++i) pq += p[i] * q[i];
        const double al = rz / pq;
        double rn = 0.0;
#pragma omp parallel for reduction(+ : rn) schedule(static)
        for (idx i = 0; i < n; ++i) {
            x[i] += al * p[i];
            r[i] -= al * q[i];
            rn += r[i] * r[i];
        }
        if (std::sqrt(rn) <= rtol * bn) return it;
        vcycle(h, 0, r.data(), z.data());
        double rz2 = 0.0;
#pragma omp parallel for reduction(+ : rz2) schedule(static)
        for (idx i = 0; i < n; ++i) rz2 += r[i] * z[i];
        const double be = rz2 / rz;
        rz = rz2;
#pragma omp parallel for schedule(static)
        for (idx i = 0; i < n; ++i) p[i] = z[i] + be * p[i];
    }
    return -maxit;
}

// ---------------------------------------------------------------------------
// Spalart-Allmaras (noft2, Allmaras 2012 negative-S~ limiter): source per unit volume
// and its derivative by the same one-sided difference as the 2D solver
// ---------------------------------------------------------------------------
static inline double sa_S(double nT, double Om, double d2, double nu, double g, const double* C) {
    const double CB1 = C[0], CW1 = C[1], CW2 = C[2], CW3 = C[3], CV1 = C[4], CV2 = C[5], CV3 = C[6],
                 K = C[7];
    const double chi = nT / nu;
    const double fv1 = chi * chi * chi / (chi * chi * chi + CV1 * CV1 * CV1);
    const double fv2 = 1.0 - chi / (1.0 + chi * fv1);
    const double Sbar = nT * fv2 / (K * K * d2);
    double St;
    if (Sbar >= -CV2 * Om) St = Om + Sbar;
    else St = Om + Om * (CV2 * CV2 * Om + CV3 * Sbar) / ((CV3 - 2 * CV2) * Om - Sbar);
    double r = 10.0;
    if (St > 0) {
        r = nT / (St * K * K * d2);
        if (!std::isfinite(r)) r = 10.0;
    }
    if (r > 10.0) r = 10.0;
    const double gg = r + CW2 * (std::pow(r, 6) - r);
    const double cw36 = std::pow(CW3, 6);
    const double fw = gg * std::pow((1 + cw36) / (std::pow(gg, 6) + cw36), 1.0 / 6.0);
    return CB1 * St * nT * g - CW1 * fw * nT * nT / d2;
}

void sa_source(idx n, const double* nuT, const double* Om, const double* d, double nu,
               const double* gamma, const double* C, double* S0, double* J) {
#pragma omp parallel for schedule(static)
    for (idx i = 0; i < n; ++i) {
        const double d2 = d[i] * d[i];
        const double g = gamma ? gamma[i] : 1.0;
        const double s0 = sa_S(nuT[i], Om[i], d2, nu, g, C);
        const double eps = 1e-6 * std::max(nuT[i], nu);
        const double s1 = sa_S(nuT[i] + eps, Om[i], d2, nu, g, C);
        S0[i] = s0;
        J[i] = (s1 - s0) / eps;
    }
}

}  // extern "C"
