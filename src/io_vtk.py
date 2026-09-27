"""
VTK XML writers (no dependencies beyond numpy).

Files written here open in ParaView and in our own PyVista viewer:
  .vtu  unstructured grid      -> the cells (triangles, quads, polygons)
  .vtp  poly data (lines)      -> boundary patches, one line per boundary face
  .vtm  multiblock             -> ties them together: open this one

Data is stored as base64-encoded binary (compact, exact float64).
2D meshes are written with z = 0.
"""

import base64
from pathlib import Path

import numpy as np

VTK_LINE, VTK_TRIANGLE, VTK_POLYGON, VTK_QUAD = 3, 5, 7, 9

_TYPES = {np.dtype(np.float64): "Float64", np.dtype(np.float32): "Float32",
          np.dtype(np.int64): "Int64", np.dtype(np.int32): "Int32",
          np.dtype(np.uint8): "UInt8"}


def _b64(arr):
    a = np.ascontiguousarray(arr)
    raw = a.tobytes()
    return base64.b64encode(np.uint64(len(raw)).tobytes() + raw).decode("ascii")


def _data_array(name, arr, ncomp=None):
    arr = np.asarray(arr)
    if arr.dtype == np.bool_:
        arr = arr.astype(np.uint8)
    if arr.dtype not in _TYPES:
        arr = arr.astype(np.float64 if arr.dtype.kind == "f" else np.int64)
    if ncomp is None:
        ncomp = 1 if arr.ndim == 1 else arr.shape[1]
    nm = f' Name="{name}"' if name else ""
    return (f'<DataArray type="{_TYPES[arr.dtype]}"{nm} NumberOfComponents="{ncomp}" '
            f'format="binary">{_b64(arr.ravel())}</DataArray>\n')


def _vec3(v):
    """(n,2) vectors -> (n,3) with z = 0; (n,) scalars unchanged."""
    v = np.asarray(v, float)
    if v.ndim == 2 and v.shape[1] == 2:
        return np.c_[v, np.zeros(len(v))]
    return v


def _field_block(tag, data):
    if not data:
        return ""
    s = f"<{tag}>\n"
    for name, arr in data.items():
        s += _data_array(name, _vec3(arr) if np.asarray(arr).dtype.kind == "f" else arr)
    return s + f"</{tag}>\n"


def _points_xyz(points):
    P = np.asarray(points, float)
    return np.c_[P, np.zeros(len(P))] if P.shape[1] == 2 else P


def write_vtu(path, points, cells, cell_data=None, point_data=None):
    """cells: list of vertex-index lists (any polygon)."""
    P = _points_xyz(points)
    conn = np.concatenate([np.asarray(c, np.int64) for c in cells])
    offs = np.cumsum([len(c) for c in cells]).astype(np.int64)
    types = np.array([{3: VTK_TRIANGLE, 4: VTK_QUAD}.get(len(c), VTK_POLYGON) for c in cells],
                     np.uint8)
    xml = ('<?xml version="1.0"?>\n'
           '<VTKFile type="UnstructuredGrid" version="1.0" byte_order="LittleEndian" '
           'header_type="UInt64">\n<UnstructuredGrid>\n'
           f'<Piece NumberOfPoints="{len(P)}" NumberOfCells="{len(cells)}">\n'
           + _field_block("PointData", point_data)
           + _field_block("CellData", cell_data)
           + "<Points>\n" + _data_array("", P, 3) + "</Points>\n"
           "<Cells>\n" + _data_array("connectivity", conn) + _data_array("offsets", offs)
           + _data_array("types", types) + "</Cells>\n"
           "</Piece>\n</UnstructuredGrid>\n</VTKFile>\n")
    Path(path).write_text(xml)


def write_vtp_lines(path, points, lines, cell_data=None):
    """lines: (m,2) point-index pairs (one boundary face each)."""
    L = np.asarray(lines, np.int64)
    used = np.unique(L)
    ren = -np.ones(len(points), np.int64)
    ren[used] = np.arange(len(used))
    P = _points_xyz(np.asarray(points)[used])
    conn = ren[L].ravel()
    offs = np.arange(2, 2 * len(L) + 1, 2, dtype=np.int64)
    xml = ('<?xml version="1.0"?>\n'
           '<VTKFile type="PolyData" version="1.0" byte_order="LittleEndian" '
           'header_type="UInt64">\n<PolyData>\n'
           f'<Piece NumberOfPoints="{len(P)}" NumberOfVerts="0" NumberOfLines="{len(L)}" '
           'NumberOfStrips="0" NumberOfPolys="0">\n'
           + _field_block("CellData", cell_data)
           + "<Points>\n" + _data_array("", P, 3) + "</Points>\n"
           "<Lines>\n" + _data_array("connectivity", conn) + _data_array("offsets", offs)
           + "</Lines>\n</Piece>\n</PolyData>\n</VTKFile>\n")
    Path(path).write_text(xml)


def write_vtm(path, blocks):
    """blocks: {name: relative file name}."""
    s = ('<?xml version="1.0"?>\n<VTKFile type="vtkMultiBlockDataSet" version="1.0">\n'
         '<vtkMultiBlockDataSet>\n')
    for i, (name, f) in enumerate(blocks.items()):
        s += f'<DataSet index="{i}" name="{name}" file="{f}"/>\n'
    s += "</vtkMultiBlockDataSet>\n</VTKFile>\n"
    Path(path).write_text(s)


def write_pvd(path, entries):
    """ParaView time-series collection. entries: [(time, relative file), ...].
    Open the .pvd in ParaView and press play to animate."""
    s = ('<?xml version="1.0"?>\n<VTKFile type="Collection" version="0.1">\n<Collection>\n')
    for t, f in entries:
        s += f'<DataSet timestep="{t:.6g}" group="" part="0" file="{f}"/>\n'
    s += "</Collection>\n</VTKFile>\n"
    Path(path).write_text(s)
