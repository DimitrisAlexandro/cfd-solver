# Low-Reynolds-number mode: Eppler 387

RANS (Spalart–Allmaras) with e^N transition (N_crit 9) and separation-induced transition: transition at laminar separation + Mayle's long-bubble length Re_l = 1000 Re_θs^0.7; leading-edge bubbles (separation ahead of 0.15 c): not shorter than Horton's Re_l = 4×10⁴ and no trip; mid-chord bubbles: a turbulence trip behind the transition point (ν̃ ≥ 100 ν over 0.02 c × 0.05 c) so the shear layer reattaches. Every angle from a cold start. Reference: NASA Langley LTPT (TM-4062) and XFOIL. Reproduce with `python src/validate_lowre.py`; `--report-only` rebuilds this page from the saved results.

| Re | α | Cl | LTPT | XFOIL | Cd | LTPT | XFOIL | ΔCd vs LTPT | x_tr upper | XFOIL x_tr | status |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1e+05 | -2 | 0.160 | 0.209 | 0.200 | 0.01711 | 0.0163 | 0.0145 | +5 % | 0.86 | 0.87 | unsteady |
| 1e+05 | 0 | 0.390 | 0.390 | 0.419 | 0.01448 | 0.0167 | 0.0155 | -13 % | 0.80 | 0.80 | converged (mean of a small oscillation) |
| 1e+05 | 2 | 0.610 | 0.588 | 0.634 | 0.01644 | 0.0211 | 0.0180 | -22 % | 0.74 | 0.74 | converged (mean of a small oscillation) |
| 1e+05 | 4 | 0.684 | 0.778 | 0.844 | 0.03916 | 0.0230 | 0.0209 | +70 % | 0.69 | 0.68 | unsteady |
| 1e+05 | 6 | 0.797 | 0.974 | 1.052 | 0.05596 | 0.0224 | 0.0213 | +150 % | 0.63 | 0.59 | unsteady |
| 1e+05 | 8 | 1.213 | 1.172 | 1.221 | 0.02572 | 0.0210 | 0.0215 | +22 % | 0.36 | 0.27 | unsteady |
| 1e+05 | 10 | 1.123 | 1.199 | 1.286 | 0.07845 | 0.0407 | 0.0412 | +93 % | 0.19 | 0.05 | unsteady |
| 2e+05 | -2 | 0.167 | 0.155 | 0.183 | 0.01510 | 0.0133 | 0.0113 | +13 % | 0.76 | 0.78 | unsteady |
| 2e+05 | 0 | 0.375 | 0.351 | 0.395 | 0.01060 | 0.0105 | 0.0099 | +1 % | 0.70 | 0.72 | converged |
| 2e+05 | 2 | 0.603 | 0.570 | 0.622 | 0.01159 | 0.0118 | 0.0111 | -2 % | 0.66 | 0.67 | converged (mean of a small oscillation) |
| 2e+05 | 4 | 0.827 | 0.786 | 0.846 | 0.01285 | 0.0133 | 0.0123 | -3 % | 0.61 | 0.61 | unsteady |
| 2e+05 | 6 | 1.042 | 0.995 | 1.062 | 0.01872 | 0.0144 | 0.0130 | +30 % | 0.63 | 0.52 | unsteady |
| 2e+05 | 8 | 1.215 | 1.179 | 1.225 | 0.01831 | 0.0174 | 0.0191 | +6 % | 0.14 | 0.12 | converged (mean of a small oscillation) |
| 2e+05 | 10 | 1.251 | 1.231 | 1.285 | 0.04075 | 0.0355 | 0.0344 | +15 % | 0.07 | 0.02 | unsteady |
| 3e+05 | -2 | 0.165 | 0.147 | 0.176 | 0.01172 | 0.0118 | 0.0102 | -1 % | 0.71 | 0.74 | unsteady |
| 3e+05 | 0 | 0.377 | 0.351 | 0.391 | 0.00910 | 0.0089 | 0.0081 | +2 % | 0.67 | 0.68 | converged |
| 3e+05 | 2 | 0.604 | 0.573 | 0.621 | 0.01001 | 0.0099 | 0.0090 | +1 % | 0.62 | 0.64 | converged |
| 3e+05 | 4 | 0.829 | 0.792 | 0.847 | 0.01099 | 0.0109 | 0.0099 | +1 % | 0.57 | 0.58 | converged |
| 3e+05 | 6 | 1.050 | 1.008 | 1.066 | 0.01516 | 0.0118 | 0.0108 | +29 % | 0.59 | 0.48 | unsteady |
| 3e+05 | 8 | 1.225 | 1.178 | 1.232 | 0.01690 | 0.0168 | 0.0181 | +1 % | 0.09 | 0.06 | converged (mean of a small oscillation) |
| 3e+05 | 10 | 1.267 | 1.250 | 1.316 | 0.03866 | 0.0315 | 0.0303 | +23 % | 0.05 | 0.02 | unsteady |

Re 1e+05, α −2…8: mean |ΔCd| 47 %, mean |ΔCl| 0.064 vs LTPT.

Re 2e+05, α −2…8: mean |ΔCd| 9 %, mean |ΔCl| 0.032 vs LTPT.

Re 3e+05, α −2…8: mean |ΔCd| 6 %, mean |ΔCl| 0.033 vs LTPT.
