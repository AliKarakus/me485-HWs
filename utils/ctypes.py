import ctypes
import os


def platform_libname(name):
    # Linux DLL
    return 'lib{}.so'.format(name)


def platform_dirs():
    # Find environment
    libpaths = os.environ.get('PYBARAM_LIB_PATH', '')

    # Add path for virtualenv
    virtpath = os.environ.get('VIRTUAL_ENV', '')
    if virtpath:
        libpaths += ':{}/lib'.format(virtpath)

    return libpaths.split(':')


def load_lib(name):
    # Load library via ctypes
    libname = platform_libname(name)

    try:
        return ctypes.CDLL(libname)
    except OSError:
        for path in platform_dirs():
            try:
                return ctypes.CDLL(os.path.join(path, libname))
            except OSError:
                pass
        else:
            raise(OSError('Cannot find {} library'.format(name)))
