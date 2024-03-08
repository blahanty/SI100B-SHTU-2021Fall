import pickle

def save_obj(path, obj):
    with open(path, "wb") as f:
        pickle.dump(obj, f)

def load_obj(path):
    ret = None
    with open(path, "rb") as f:
        ret = pickle.load(f)
    return ret
