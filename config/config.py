class Config:
    def __init__(self) -> None:
        self.log = LogConf()
        self.data = DataConf()
        self.train = TrainConf()
        self.test = TestConf()
        self.model = ModelConf()

class LogConf:
    def __init__(self) -> None:
        self.root = ""
        self.log_fmt = ""
        self.date_fmt = ""

class DataConf:
    def __init__(self) -> None:
        self.root = ""
        self.image_size = 0
        self.num_classes = 0
        self.usage = 0

class TrainConf:
    def __init__(self) -> None:
        self.lr = 0
        self.batch_size = 0
        self.device = ""
        self.save_path = ""
        self.model_name = ""

class TestConf:
    def __init__(self) -> None:
        self.save_path = ""

class ModelConf:
    def __init__(self) -> None:
        self.name = ""


if __name__ == "__main__":
    config = Config()
    from utils import parseConfig
    parseConfig("conf\config.ini", config)
    print(config)