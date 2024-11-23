from .bikefusion import *
import os

current_dir = os.path.dirname(os.path.abspath(__file__))

def load_bikefusion():
    return load_bikefusion_and_data(current_dir)

__all__ = [name for name in dir() if not name.startswith("_") and name not in {"os"}]