# Load the benchmarks from the subfiles
from .nine_rooms import NineRooms

def get_model_fun(model_name):
    if model_name == 'NineRooms':
        envfun = NineRooms
    else:
        assert False, f"Unknown model name: {model_name}"
    return envfun
