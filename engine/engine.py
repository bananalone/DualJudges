import bisect


class Engine:
    """
    训练引擎
    @ step标识执行函数 def step(priority: int) -> Any
    @ run运行每一个step的函数
    """
    def __init__(self) -> None:
        self.__funcList = []

    def step(self, priority: int):
        """
        step标识执行函数 def step(priority: int) -> Any
        @ 函数需满足 func(step: int) -> Any
        @ 按照priority由小到大依次执行
        """
        def outer(func):
            bisect.insort(self.__funcList, (priority, func))
            return func
        return outer

    def run(self, start: int,  steps: int):
        """
        开启训练
        @run运行每一个step的函数
        @start 起始步
        @steps 总步数
        """
        for step in range(start, start + steps):
            for _, func in self.__funcList:
                func(step)


if __name__ == "__main__":
    import time

    engine = Engine()

    @engine.step(2)
    def func1(step):
        time.sleep(0.2)
        msg = {
            "step": step,
            "massage": "func1"
        }
        print(msg)
        return msg
    
    @engine.step(1)
    def func2(step):
        time.sleep(0.2)
        msg = {
            "step": step,
            "massage": "func2"
        }
        print(msg)
        return msg

    engine.run(3, 10)

    func1(111)
