import json
import numpy as np

class NumpyEncoder(json.JSONEncoder):
  def default(self, obj):
    if isinstance(obj, np.ndarray):
      return obj.tolist()
    if np.issubdtype(obj, np.integer):
      return int(obj)
    return super().default(obj)

def RMS(x, **kwargs):
  return np.sqrt(np.mean(np.square(x), **kwargs))
