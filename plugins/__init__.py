# -*- coding: utf-8 -*-
from plugins.base import BasePlugin
from plugins.force import ForcePlugin
from plugins.stats import StatsPlugin
from plugins.writer import WriterPlugin
from plugins.surfint import SurfIntPlugin
from utils.misc import subclass_by_name


def get_plugin(name, *args, **kwargs):
    return subclass_by_name(BasePlugin, name)(*args, **kwargs)