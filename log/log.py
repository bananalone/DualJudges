import logging
import time
from functools import wraps

from log.utils import prettyDict

def initLogging(filepath: str, format: str, datefmt: str):
    logging.basicConfig(
        format = format,
        datefmt = datefmt,
        stream = open(filepath, "w", encoding="utf-8"),
        level = logging.INFO
    )

def log(level: str, stdout: bool = False):
    """
    在需要打印日志的函数上添加 @log(level, stdout), 函数需返回一个字典
    @level must in ["debug", "info", "warning", "error", "critical"]
    @if stdout == True, 打印到控制台
    """
    printLog = {
        "debug": logging.debug, 
        "info": logging.info,
        "warning": logging.warning,
        "error": logging.error,
        "critical": logging.critical
    }
    def outter(func):
        @wraps(func)
        def inner(*args, **kwargs):
            msg = func(*args, **kwargs)
            if msg is None:
                return None
            pMsg = prettyDict(msg)
            if stdout:
                print(pMsg)
            printLog[level](pMsg)
            return msg
        return inner
    return outter


if __name__ == "__main__":

    initLogging(
        filepath = r"out/logs/segnet-loc-pfeedback.log",
        format =  "%(asctime)s[%(levelname)s] %(filename)s[line:%(lineno)d]:\n%(message)s",
        datefmt = r"%Y/%m/%d %H:%M:%S"
    )

    # import sys, os
    # sys.path.append(os.path.abspath(os.path.dirname(".")))
    # print(sys.path)
    # from utils import pretty
    
    @log("info", stdout=True)
    def outMsg(msg: str):
        time.sleep(1)
        return {
            "python file": __name__,
            "time": time.strftime(r'%Y-%m-%d %H:%M:%S', time.localtime()),
            "function": outMsg.__name__,
            "msg": msg
        }

    outMsg("测试log1")
    outMsg("测试log2")
    outMsg("测试log3")
