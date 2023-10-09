# -*- coding: utf-8 -*-
# functions are originated from
# https://github.com/PyFR/PyFR/blob/develop/pyfr/util.py
# modified by jspark
#
def subclasses(cls, just_leaf=False):
    sc = cls.__subclasses__()
    ssc = [g for s in sc for g in subclasses(s, just_leaf)]

    return [s for s in sc if not just_leaf or not s.__subclasses__()] + ssc


def subclass_by_name(cls, name):
    for s in subclasses(cls):
        if s.name == name:
            return s


def subclass_dict(cls, attr):
    sc = {getattr(sc, attr): sc for sc in cls.__subclasses__()}
    return sc


class ProxyList(list):
    def __init__(self, *args):
        super().__init__(*args)

    def __getattr__(self, attr):
        return ProxyList([getattr(item, attr) for item in self])

    def __setattr__(self, key, value):
        for item in self:
            setattr(item, key, value)

    def __getitem__(self, key):
        return ProxyList([item[key] for item in self])

    def __call__(self, *args, **kwargs):
        return ProxyList([item(*args, **kwargs) for item in self])

    def apply(self, fn, *args, **kwargs):
        return ProxyList([fn(item, *args, **kwargs) for item in self])

    def apply_at(self, name, fn, *args, **kwargs):
        return ProxyList([fn(*getattr(item, name), args, **kwargs) for item in self])
