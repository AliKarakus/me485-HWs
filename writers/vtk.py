# -*- coding: utf-8 -*-

import numpy as np

from writers.base import BaseWriter


class VTKWriter(BaseWriter):
    name = 'vtu'
    _vtk_types = dict(tri=5, quad=9, tet=10, pyr=14, pri=13, hex=12)
    _is_cstyle = True
    _var_names = {'rho': 'Density', 'u': 'Velocity', 'p': 'Pressure'}

    def _raw_write(self):
        # Data
        nodes = np.array(self._nodes, dtype=np.float32)
        cons = self._vtu_con()
        voff = self._vtu_off()
        vtyp = self._vtu_typ()
        soln, aux = self._soln

        primevars = self._elms.primevars

        if aux is not None:
            auxvars = self._elms.auxvars

        # Write
        with open(self._outf, 'wb') as fp:
            head = '''<?xml version="1.0" ?>
            <VTKFile byte_order="LittleEndian" type="UnstructuredGrid" version="0.1">
            <UnstructuredGrid>
            <Piece NumberOfPoints="{0}" NumberOfCells="{1}">'''.format(len(nodes), len(voff))
            self._write_str(head, fp)

            # Header for node
            off = 0
            self._write_str('\n<Points>', fp)
            off = self._write_arr_header(fp, '', 3, off, nodes.nbytes)
            self._write_str('\n</Points>', fp)
            self._write_str('\n<Cells>', fp)

            # Header for cell
            off = self._write_arr_header(
                fp, 'connectivity', '', off, voff[-1]*4, dtype='Int32')
            off = self._write_arr_header(
                fp, 'offsets', '', off, voff.nbytes, dtype='Int32')
            off = self._write_arr_header(
                fp, 'types', '', off, vtyp.nbytes, dtype='UInt8')

            self._write_str('\n</Cells>', fp)

            # Header for cell data
            self._write_str('\n<CellData>', fp)
            for pv in primevars:
                if pv == 'u':
                    off = self._write_arr_header(
                        fp, 'Velocity', self.ndims, off, self.ndims*voff.nbytes)
                elif pv in ['v', 'w']:
                    pass
                else:
                    if pv in self._var_names:
                        vname = self._var_names[pv]
                    else:
                        vname = pv

                    off = self._write_arr_header(
                        fp, vname, 1, off, voff.nbytes)

            if aux is not None:
                for av in auxvars:
                    off = self._write_arr_header(fp, av, 1, off, voff.nbytes)

            self._write_str('\n</CellData>', fp)

            self._write_str('\n</Piece>\n</UnstructuredGrid>', fp)

            self._write_str('\n<AppendedData encoding="raw">\n_', fp)
            self._write_darray(nodes, np.float32, fp)
            self._write_darray(cons, np.int32, fp)
            self._write_darray(voff, np.int32, fp)
            self._write_darray(vtyp, np.uint8, fp)

            for i, pv in enumerate(primevars):
                if pv == 'u':
                    self._write_darray(
                        soln[i:i+self.ndims].swapaxes(0, 1), np.float32, fp)
                elif pv in ['v', 'w']:
                    pass
                else:
                    self._write_darray(soln[i], np.float32, fp)

            if aux is not None:
                for i in range(len(auxvars)):
                    self._write_darray(aux[i], np.float32, fp)

            self._write_str('\n</AppendedData>\n</VTKFile>', fp)

    def _vtu_typ(self):
        ele = []
        for k, v in self._cells.items():
            n = len(v)
            ele.append(self._vtk_types[k]*np.ones(n, dtype='i1'))

        return np.concatenate(ele)

    def _vtu_off(self):
        off = []
        for k, v in self._cells.items():
            n = len(v)
            m = len(v[0])
            off.append(m*np.ones(n, dtype=np.int32))

        off = np.concatenate(off)
        return np.cumsum(off, dtype=np.int32)

    def _vtu_con(self):
        cons = []
        for k, v in self._cells.items():
            cons.append(v.reshape(-1))

        return np.concatenate(cons)

    def _write_str(self, s, fp):
        # Write string
        fp.write(s.encode('utf-8'))

    def _write_darray(self, arr, dtype, fp):
        # Write array
        arr = np.array(arr, dtype=dtype)
        np.uint32(arr.nbytes).tofile(fp)
        arr.tofile(fp)

    def _write_arr_header(self, fp, name, nvars=1, offset=0, nbytes=0, dtype='Float32'):
        _dtxt = '\n<DataArray Name="{0}" type="{1}"' \
            ' NumberOfComponents="{2}" format="appended" offset="{3}"/>'

        self._write_str(_dtxt.format(name, dtype, nvars, offset), fp)

        return offset + 4 + nbytes
