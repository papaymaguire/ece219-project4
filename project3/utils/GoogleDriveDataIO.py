import joblib

class GoogleDriveDataIO ():
    def __init__(self, mount_path, drive_path) -> None:
        self.mount_path = mount_path
        self.drive_path = drive_path

    def save (self, name, object):
        path = f'{self.mount_path}/{self.drive_path}/{name}'
        joblib.dump(object, path)
        return path
    
    def load (self, name):
        path = f'{self.mount_path}/{self.drive_path}/{name}'
        return joblib.load(path)