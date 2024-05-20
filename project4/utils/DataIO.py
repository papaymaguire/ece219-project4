import joblib

class DataIO ():
    def __init__(self, data_path) -> None:
        self.data_path = data_path

    def save (self, object, name):
        path = f'{self.data_path}/{name}'
        joblib.dump(object, path)
        return path
    
    def load (self, name):
        path = f'{self.data_path}/{name}'
        return joblib.load(path)