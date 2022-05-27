from torch.utils import data

class DeviceDataloader:
    def __init__(self, dataloader: data.DataLoader, device: str) -> None:
        self.dataloader = dataloader
        self.device = device

    def __iter__(self):
        self.it = iter(self.dataloader)
        return self

    def __next__(self):
        datas = next(self.it)
        devDatas = []
        for data in datas:
            devData = data.to(self.device)
            devDatas.append(devData)
        return devDatas

    def __len__(self):
        return len(self.dataloader)