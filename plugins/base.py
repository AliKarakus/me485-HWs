# -*- coding: utf-8 -*-
import os


def csv_write(fname, header):
    # Write data as CSV file
    outf = open(fname, 'a')

    if os.path.getsize(fname) == 0:
        print(','.join(header), file=outf)

    return outf


class BasePlugin:
    # Abstract class of Plugin
    name = None
