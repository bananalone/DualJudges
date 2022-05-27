import os
from configparser import RawConfigParser
from typing import Any

def parseConfig(path: str, config: Any):
    '''
    读取path路径下的ini配置文件并保存在config里
    @params: path: 配置文件路径
    @params: config: 配置的对象
    @returns: None
    '''
    cp = RawConfigParser()
    if not os.path.exists(path):
        raise FileNotFoundError("{} is not found".format(path))
    cp.read(path, encoding="utf-8")
    for section in cp.sections():
        if not hasattr(config, section):
            continue
        attr = getattr(config, section)
        for item in cp.items(section):
            name, value = item
            if not hasattr(attr, name):
                continue
            try:
                value = int(value)
            except ValueError:
                try:
                    value = float(value)
                except ValueError:
                    pass
            setattr(attr, name, value)
