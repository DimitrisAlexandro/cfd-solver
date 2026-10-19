"""AeroLab: CFD and aircraft-design toolkit (the code behind the app).

Packages:
    core       Finite-volume solvers and numerics: mesh geometry, SIMPLEC, coupled and compressible solvers, turbulence, linear solvers.
    mesh       Meshing and geometry: hybrid mesher, triangulation, mesh checks, case runner, viewers and result readers.
    physics    Transition and boundary-layer models: e^N, AFT, swept-wing criteria, integral boundary layer.
    airfoil    Airfoil sections: geometry, CST, panel method, polars, multi-element, XFOIL reference.
    design     Adjoint, shape optimisation, inverse design, sizing and the polar surrogate.
    ml         Field inversion and machine-learned turbulence corrections.
    aircraft   Wings and aircraft: models, loads, performance, structures, aeroelasticity, databases.
    workflow   Projects, pre-run checks, confidence, reports and the job queue.
    validation Validation and benchmark scripts.
    fv3d       3D finite-volume RANS solver (wings): mesh, solver, native C++ kernel, validation.

Command line: python src/run.py <tool> [arguments]   (e.g. python src/run.py airfoil polar naca0012 ...)
"""
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent

# tool (module) name -> package
MODULES = {
    'compressible': 'core',
    'coupled': 'core',
    'fv1d_advection': 'core',
    'fv1d_muscl': 'core',
    'fvmesh': 'core',
    'io_vtk': 'core',
    'kernels': 'core',
    'linsolve': 'core',
    'postproc': 'core',
    'simple_solver': 'core',
    'turbulence': 'core',
    'case': 'mesh',
    'cdt2d': 'mesh',
    'delaunay2d': 'mesh',
    'geometry': 'mesh',
    'hybrid_mesher': 'mesh',
    'mesh_check': 'mesh',
    'predicates': 'mesh',
    'view_mesh': 'mesh',
    'viewer': 'mesh',
    'aft': 'physics',
    'boundary_layer': 'physics',
    'fsc': 'physics',
    'swept_transition': 'physics',
    'transition': 'physics',
    'airfoil': 'airfoil',
    'airfoil_geom': 'airfoil',
    'cst': 'airfoil',
    'multi_element': 'airfoil',
    'panel': 'airfoil',
    'polar': 'airfoil',
    'xfoil_ref': 'airfoil',
    'adjoint': 'design',
    'adjoint_error': 'design',
    'design_adjoint': 'design',
    'design_drag_rise': 'design',
    'design_inverse': 'design',
    'design_optimize': 'design',
    'design_wing_sizing': 'design',
    'surrogate': 'design',
    'bubble_ml': 'ml',
    'fiml': 'ml',
    'fiml_multi': 'ml',
    'aero_database': 'aircraft',
    'aeroelastic': 'aircraft',
    'aircraft': 'aircraft',
    'aircraft_import': 'aircraft',
    'drag_buildup': 'aircraft',
    'loads': 'aircraft',
    'performance': 'aircraft',
    'propeller': 'aircraft',
    'sizing': 'aircraft',
    'stl_aircraft': 'aircraft',
    'weight_balance': 'aircraft',
    'wing': 'aircraft',
    'wing_structure': 'aircraft',
    'jobqueue': 'workflow',
    'preflight': 'workflow',
    'project': 'workflow',
    'report': 'workflow',
    'report_aircraft': 'workflow',
    'templates': 'workflow',
    'trust': 'workflow',
    'bench_adjoint': 'validation',
    'benchmark': 'validation',
    'benchmark_coupled': 'validation',
    'validate_aeroelastic': 'validation',
    'validate_aircraft': 'validation',
    'validate_airfoil': 'validation',
    'validate_bump': 'validation',
    'validate_cavity': 'validation',
    'validate_channel': 'validation',
    'validate_cylinder': 'validation',
    'validate_design_tools': 'validation',
    'validate_flatplate': 'validation',
    'validate_lowre': 'validation',
    'validate_multi': 'validation',
    'validate_naca0012_sa': 'validation',
    'validate_phase3': 'validation',
    'validate_research': 'validation',
    'validate_transition': 'validation',
    'validate_transonic': 'validation',
    'validate_unsteady': 'validation',
    'validate_wing': 'validation',
    'postview': 'mesh',
    'wakeview': 'mesh',
    'mesh3d': 'fv3d',
    'meshgen3d': 'fv3d',
    'native': 'fv3d',
    'solver3d': 'fv3d',
    'validate_core3d': 'fv3d',
    'validate_wing3d': 'fv3d',
    'wing3d': 'fv3d',
    'wingmesh': 'fv3d',
}


def module(name):
    """'airfoil' or 'airfoil.py' -> 'aerolab.airfoil.airfoil'"""
    name = Path(str(name)).stem
    return f"aerolab.{MODULES[name]}.{name}"


def script_path(name):
    """the file of a tool: 'airfoil.py' -> src/aerolab/airfoil/airfoil.py"""
    name = Path(str(name)).stem
    return SRC / "aerolab" / MODULES[name] / f"{name}.py"


def script_argv(args):
    """python arguments for a run: ['airfoil.py', ...] (the tool names used before the
    package layout, still in saved jobs and projects) -> ['-m', 'aerolab.airfoil.airfoil', ...];
    run with cwd = src. Anything else is returned unchanged."""
    args = [str(a) for a in args]
    if args and args[0].endswith(".py") and "/" not in args[0] and Path(args[0]).stem in MODULES:
        return ["-m", module(args[0])] + args[1:]
    return args
