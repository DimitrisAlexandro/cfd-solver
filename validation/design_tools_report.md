# Design tools: validation

Reproduce with `python src/validate_design_tools.py [--verify]`.

## Drag buildup

- Cessna 172-like geometry (strut-braced high wing, fixed gear with wheel fairings, riveted metal, 8° upsweep): **CD0 0.0273** against 0.027–0.031 quoted for the type (UIUC/FlightGear model from Roskam: 0.031). Equivalent-skin-friction check (Cfe 0.0055): 0.0214. Drag area f = 0.449 m².
- NACA 0012 perimeter (wetted area per chord): 2.0393 from the section, Raymer's rule 2.0394.
- Fully turbulent, smooth: the fuselage term equals the aircraft model's fuselage drag to machine precision (test).

## Propeller

- Adkins–Liebeck design re-analysed with BEMT (2 blades, 60 m/s, 2500 rpm, 60 kW): thrust 899.1 / 900.3 N, power 60.00 / 60.16 kW, efficiency 0.899 / 0.898 (actuator disc 0.963).
- Without profile drag the design efficiency rises with the blade count towards the actuator-disc value: 2: 0.841, 3: 0.848, 4: 0.851, 6: 0.855, 8: 0.857, 12: 0.859, 20: 0.860 (disc 0.874; the rest is swirl and hub).

## Conceptual sizing against real aircraft

| case | quantity | reference | sized | difference |
|---|---|---|---|---|
| C172 | MTOW (kg) | 1111 | 956.4 | -13.9 % |
| C172 | empty mass (kg) | 757 | 636.8 | -15.9 % |
| C172 | fuel (kg) | 144 | 121.7 | -15.5 % |
| C172 | wing area (m²) | 16.2 | 13.9 | -13.9 % |
| C172 | power (kW) | 119 | 112.5 | -5.4 % |
| Velis | MTOW (kg) | 600 | 659.5 | +9.9 % |
| Velis | empty mass without battery (kg) | 288 | 322.3 | +11.9 % |
| Velis | battery (kg) | 140 | 164.8 | +17.7 % |
| Velis | wing area (m²) | 9.51 | 9.9 | +4.0 % |
| Velis | span (m) | 10.71 | 10.9 | +1.9 % |
| Velis | power (kW) | 49.2 | 49.7 | +1.0 % |

C172: requirements from the 172S handbook class (2 occupants + 25 kg, 1100 km at 65 % power, 8000 ft, 45 min reserve; stall 48 kt with flaps, take-off 497 m, landing 407 m over 50 ft), drag details as above. The largest difference is the empty mass: Raymer's general-aviation Class II equations put the C172 structure and systems 16 % lighter, and the lighter aircraft then needs less wing (the stall line fixes the wing loading) and less fuel.
Velis Electro: 2 × 86 kg, 50 min at 170 km/h + 20 min reserve, 177 Wh/kg pack (24.8 kWh / 140 kg), 85 % usable, composite.

Conceptual methods are usually quoted at ±10–15 %. Outside ±15 % here: C172 empty mass (-16 %), C172 fuel (-15 %), Velis battery (+18 %).

## Sizing checked by the performance tool (default two-seat trainer)

| quantity | sizing / required | performance tool |
|---|---|---|
| stall speed, landing flap (m/s) | 24 | 24.1 |
| take-off over 15 m (m) | 450 | 526 |
| landing over 15 m (m) | 450 | 456 |
| rate of climb (m/s) | 3.5 | 3.87 |
| service ceiling (m) | 4e+03 | 5.6e+03 |
| range at cruise speed, cruise fuel (km) | 900 | 1.17e+03 |
| landing CL max | 2 | 1.98 |
| Oswald e | 0.842 | 0.846 |

The performance tool uses the trimmed nonlinear polar (vortex lattice + CFD sections), the BEMT propeller designed for cruise and a time-integrated take-off. The take-off line is the one to watch: a fixed-pitch propeller pitched for cruise gives less static thrust than Roskam's statistics assume.

## Weight and balance

- Demo trainer: c.g. range 14.7–23.3 % MAC (estimated component list), tail S_h/S 0.198; smallest tail for this range at the current wing position 0.116.
- Tests: neutral point of the scissor plot = the stability page's at the same tail; wing placement puts the loaded c.g. at the target to 0.05 % MAC; suggested gear meets tip-back, turnover, rotation and propeller clearance.
