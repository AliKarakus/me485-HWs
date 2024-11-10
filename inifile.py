# -*- coding: utf-8 -*-
from configparser import ConfigParser, NoOptionError, NoSectionError

import io
import re


class INIFile:
    def __init__(self, file=None):
        self._cfg = cfg = ConfigParser()

        if file:
            cfg.read(file)

    def set(self, sect, opt, value):
        try:
            self._cfg.set(sect, opt, value)
        except NoSectionError:
            self._cfg.add_section(sect)
            self._cfg.set(sect, opt, value)

    def get(self, sect, opt, default=None):
        cfg = self._cfg
        try:
            item = cfg.get(sect, opt)
        except (NoSectionError, NoOptionError):
            self.set(sect, opt, str(default))
            item = self.get(sect, opt)

        return item

    def getint(self, sect, item, default=None):
        return int(self.get(sect, item, default))

    def getfloat(self, sect, item, default=None):
        return float(self.get(sect, item, default))

    def getlist(self, sect, item, default=''):
        txt = self.get(sect, item, default)
        return [eval(e) for e in txt.split(',')]

    def geteval(self, sect, item, default=None):
        return eval(self.get(sect, item, default))

    def getexpr(self, sect, item, subs={}, default=None):
        expr = self.get(sect, item, default).lower()
        for k, v in subs.items():
            expr = re.sub(r'\b{}\b'.format(k), str(v), expr)

        return expr

    def items(self, sect):
        return {k: eval(v) for (k, v) in self._cfg.items(sect)}

    def sections(self):
        return self._cfg.sections()

    def tostr(self):
        buf = io.StringIO()
        self._cfg.write(buf)
        return buf.getvalue()

    def fromstr(self, str):
        # h5py compatability 
        # Older version (2.X) gives string but new one (3.X) gives bytes
        if isinstance(str, bytes):
            str = str.decode('utf-8')

        self._cfg.read_string(str)

    def has_section(self, sect):
        return self._cfg.has_section(sect)
