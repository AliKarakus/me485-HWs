# -*- coding: utf-8 -*-
from argparse import ArgumentParser, FileType

from api.io import import_mesh, partition_mesh, export_soln
from api.simulation import run, restart
from inifile import INIFile
from readers.native import NativeReader


def process_import(args):
    import_mesh(args.inmesh, args.outmesh, args.scale)


def process_part(args):
    partition_mesh(args.mesh, args.out, args.npart)


def process_export(args):
    export_soln(args.mesh, args.soln, args.out)


def process_run(args):
    mesh = NativeReader(args.mesh)
    cfg = INIFile(args.ini)

    run(mesh, cfg)


def process_restart(args):
    mesh = NativeReader(args.mesh)
    soln = NativeReader(args.soln)
    
    # Config file
    if args.ini:
        cfg = INIFile(args.ini)
    else:
        cfg = INIFile()
        cfg.fromstr(soln['config'])

    restart(mesh, soln, cfg)


def main():
    ap = ArgumentParser(prog='mefvm')
    sp = ap.add_subparsers(dest='cmd', help='sub-command help')

    # Common options
    ap.add_argument('--verbose', '-v', action='count')

    # Import command
    ap_import = sp.add_parser('import', help='import --help')
    ap_import.add_argument('inmesh', help='input mesh file')
    ap_import.add_argument('outmesh', help='output mesh file')
    ap_import.add_argument('-s', '--scale', type=float, default=1,
                           help='scale mesh')
    ap_import.set_defaults(process=process_import)

    # Partition command
    ap_part = sp.add_parser('partition', help='partition --help')
    ap_part.add_argument('npart', help='number of partition')
    ap_part.add_argument('mesh', help='mesh file')
    ap_part.add_argument('out', help='partitioned mesh file')
    ap_part.set_defaults(process=process_part)

    # Run command
    ap_run = sp.add_parser('run', help='run --help')
    ap_run.add_argument('mesh', type=str, help='mesh file')
    ap_run.add_argument('ini', type=str, help='config file')
    ap_run.set_defaults(process=process_run)

    # Run restart
    ap_restart = sp.add_parser('restart', help='run --help')
    ap_restart.add_argument('mesh', type=str, help='mesh file')
    ap_restart.add_argument('soln', type=str, help='solution file')
    ap_restart.add_argument('ini', nargs='?', type=str, help='config file')
    ap_restart.set_defaults(process=process_restart)

    # Export command
    ap_export = sp.add_parser('export', help='export --help')
    ap_export.add_argument('mesh', help='mesh file')
    ap_export.add_argument('soln', help='solution file')
    ap_export.add_argument('out', help='output file')
    ap_export.set_defaults(process=process_export)

    # Parse the arguments
    args = ap.parse_args()

    # Invoke the process method
    if hasattr(args, 'process'):
        args.process(args)
    else:
        ap.print_help()


if __name__ == '__main__':
    main()