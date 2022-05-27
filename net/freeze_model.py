from typing import Iterable, List, Union

import torch.nn as nn

class FreezeModel(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model


    def forward(self, *args):
        return self.model.forward(*args)


    def freezeByLayers(self, layerNames: Union[str, List[str]], freeze=True):
        """冻结指定层的参数

        Args:
            layerNames (List[str] or str): 层名
            freeze (bool, optional): 冻结为True，解冻为False. Defaults to True.
        """
        if not isinstance(layerNames, Iterable):
            layerNames = [layerNames]
            
        for name, child in self.model.named_children():
            if name in layerNames:
                for param in child.parameters():
                    param.requires_grad = not freeze
        
    
    def freeze(self, freeze=True):
        """冻结所有层的参数

        Args:
            freeze (bool, optional): 冻结为True，解冻为False. Defaults to True.
        """
        for param in self.model.parameters():
            param.requires_grad = not freeze
    
    
    def listNames(self):
        """列出所有层

        Returns:
            List: 层名
        """
        names = []
        for name, _ in self.model.named_children():
            names += [name]
        return names
    
    
    def freezeBeforeLayer(self, layerName: str, freeze=True):
        """冻结layerName之前的所有层的参数，包括该层

        Args:
            layerName (str): 层名
            freeze (bool, optional): 冻结为True，解冻为False. Defaults to True.
        """
        names = self.listNames()
        if layerName not in names:
            raise Exception("layerName does not exist")
        layerNames =  names[:names.index(layerName)+1]
        self.freezeByLayers(layerNames, freeze)