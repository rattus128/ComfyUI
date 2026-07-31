import functools
import sys

import psutil


MIB = 1024 ** 2
GIB = 1024 ** 3
WINDOWS_EPOCH_OFFSET = 11644473600


def _fixed_pagefile_quota(settings):
    if not settings:
        return 0

    paths = set()
    quota = 0
    for setting in settings:
        try:
            path, initial, maximum = setting.rsplit(None, 2)
            initial = int(initial)
            maximum = int(maximum)
        except (AttributeError, TypeError, ValueError):
            return None

        path = path.casefold()
        if not path or path in paths or initial <= 0 or initial != maximum:
            return None
        paths.add(path)
        quota += maximum * MIB
    return quota


@functools.cache
def fixed_pagefile_quota():
    if sys.platform != "win32":
        return None

    import winreg

    try:
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"SYSTEM\CurrentControlSet\Control\Session Manager\Memory Management") as key:
            settings, value_type = winreg.QueryValueEx(key, "PagingFiles")
            modified = winreg.QueryInfoKey(key)[2] / 10000000 - WINDOWS_EPOCH_OFFSET
    except OSError:
        return None

    if value_type != winreg.REG_MULTI_SZ or not isinstance(settings, (list, tuple)):
        return None
    if modified > psutil.boot_time():
        return None
    return _fixed_pagefile_quota(settings)


def recommended_pagefile_size(vram):
    quota = fixed_pagefile_quota()
    minimum = vram * 5 // 4
    if quota is None or quota >= minimum:
        return None
    return (minimum + GIB - 1) // GIB * GIB
