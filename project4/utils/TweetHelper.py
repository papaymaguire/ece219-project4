class TweetHelper:
    def __init__(self, X_train, X_test, y_train, y_test) -> None:
        self.data = {
            "train": {
                "features": {
                    "original": X_train,
                },
                "target": y_train
            },
            "test": {
                "features": {
                    "original": X_test,
                },
                "target": y_test
            },
        }