_model_entrypoints = {}

def register_model(fn):
    name = fn.__name__
    if name in _model_entrypoints.keys():
        raise Exception("%s method has already exists" % name)
    _model_entrypoints[name] = fn
    return fn

def create_model(model_fn_name: str, **kwargs):
    if model_fn_name not in _model_entrypoints.keys():
        raise Exception("%s does not exist" % model_fn_name)
    model_fn = _model_entrypoints[model_fn_name]
    model = model_fn(**kwargs)
    return model