# Learned laminar-separation-bubble length (Eppler 387, LTPT)

K* = the Mayle constant (Re_l = K Re_theta,s^0.7) at which the RANS drag equals the measured drag; a Gaussian process of log K* on the state of the laminar layer at separation (log Re_theta,s, H_s, x_sep, log Re). Leave-one-out: K predicted from the other cases, the drag at that K interpolated from the case's own K scan. Reproduce: `python src/bubble_ml.py scan` then `fit`.

| Re | α | K* | K predicted (LOO) | Cd LTPT | Cd, K = 1000 | Cd, learned K | error K=1000 | error learned |
|---|---|---|---|---|---|---|---|---|
| 1e+05 | 0 | 156 (extrap.) | 797 (x/÷ 1.79) | 0.0167 | 0.01448 | 0.01470 | -13 % | -12 % |
| 1e+05 | 2 | 1600 (extrap.) | 185 (x/÷ 1.79) | 0.0211 | 0.01644 | 0.01624 | -22 % | -23 % |
| 2e+05 | 0 | 1111 (extrap.) | 469 (x/÷ 1.47) | 0.0105 | 0.01060 | 0.01178 | +1 % | +12 % |
| 2e+05 | 2 | 820 | 1291 (x/÷ 1.21) | 0.0118 | 0.01159 | 0.01159 | -2 % | -2 % |
| 3e+05 | 0 | 732 (extrap.) | 1109 (x/÷ 3.47) | 0.0089 | 0.00910 | 0.00917 | +2 % | +3 % |
| 3e+05 | 2 | 1123 (extrap.) | 981 (x/÷ 2.82) | 0.0099 | 0.01001 | 0.01003 | +1 % | +1 % |
| 3e+05 | 4 | 1116 (extrap.) | 1185 (x/÷ 4.04) | 0.0109 | 0.01099 | 0.01099 | +1 % | +1 % |
| 3e+05 | 8 | 2500 (extrap.) | 800 (x/÷ 5.38) | 0.0168 | 0.01690 | 0.01690 | +1 % | +1 % |

Mean |ΔCd| vs LTPT: K = 1000 5.3 %, learned (leave-one-out) 6.8 %.

## Findings

* Only 1 of 8 cases bracket the measured drag on the steady branch. The steady drag responds weakly to the bubble constant: d ln Cd / d ln K = -0.063 (median; range -0.095 to +0.001) - a factor 2 in the bubble length changes Cd by a few per cent at most before the bubble bursts.
* At Re 3e5 the fixed K = 1000 is already within 1-2 % of LTPT; at Re 1e5 no bubble length on the steady branch reaches the measured drag (-13 to -22 %).
* So the learned K does not beat the constant: the transition location inside the bubble is not what limits the accuracy. What is left is how the separated shear layer reattaches and recovers under SA (and the unsteadiness of the bubble that steady RANS averages away) - a correction of the turbulence model in the shear layer (the FIML route, fiml.py) is the indicated next step, not a better transition-length correlation.
* The learned model is therefore not installed by default (bubble_ml.install() opts in).

The band is the GP's 2-sigma factor on K. Only steady RANS solutions enter the inversion: beyond a critical K the bubble bursts and the solution turns unsteady (time-averaged drag 50-150 % high), a branch on which matching the measured drag would be meaningless. (extrap.): the measured drag is not bracketed by the steady runs; K* is extrapolated by at most one ladder step.
