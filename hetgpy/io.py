import pickle

def load(filename,rebuild=True):
    with open(filename,'rb') as stream:
        model = pickle.load(stream)
        if rebuild:
            model.rebuild()
    return model

def save(model,filename,strip=True):
    model = model.copy()
    if strip:
        model.strip()
    with open(filename,'wb') as stream:
        pickle.dump(model,stream)