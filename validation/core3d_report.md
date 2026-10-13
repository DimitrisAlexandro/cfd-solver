# 3D finite-volume core validation (3D-1)

Native kernels: {'native': True, 'threads': 2, 'note': None}

## cavity: PASS
- cells: 1024
- iterations_3d: 218
- iterations_2d: 218
- time_3d_s: 0.9
- max_dU_3d_vs_2d: 8.99203e-12
- max_Uz: 0
- wall_time_s: 2.1

## airfoil: PASS
- cells: 15004
- Cl_3d: 1.0904
- Cd_3d: 0.0129695
- Cl_2d: 1.09039
- Cd_2d: 0.0129688
- Cl_CFL3D: 1.0909
- Cd_CFL3D: 0.01231
- iterations_3d: 670
- time_3d_s: 46.8
- wall_time_s: 73.9

## duct: PASS
- cells: 15680
- iterations: 133
- time_s: 4.1
- umax_over_umean: 2.04015
- ref_umax_over_umean: 2.0962
- fRe: 56.1261
- ref_fRe: 56.91
- wall_time_s: 4.3

## sphere: PASS
- cells: 61440
- iterations: 68
- time_s: 8.3
- Cd: 1.09884
- Cd_pressure: 0.521333
- Cd_friction: 0.57751
- ref_Cd: [1.087, 1.09]
- Ls_over_D: 0.875
- ref_Ls_over_D: [0.87, 0.88]
- wall_time_s: 9.5
