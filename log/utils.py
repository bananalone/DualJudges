import json

def prettyDict(msgDict: dict):
    """
    美化dict输出
    """
    return json.dumps(msgDict, indent = 4, ensure_ascii = False)