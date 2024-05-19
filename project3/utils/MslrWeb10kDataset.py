from sklearn.datasets import load_svmlight_file
from sklearn.metrics import ndcg_score
import numpy as np

class MslrWeb10kDataset ():
    def __init__(self) -> None:
        pass
    
    # Load the dataset for one fold
    def load_one_fold(self, data_path):
        X_train, y_train, qid_train = load_svmlight_file(str(data_path + 'train.txt'), query_id=True)
        X_test, y_test, qid_test = load_svmlight_file(str(data_path + 'test.txt'), query_id=True)
        y_train = y_train.astype(int)
        y_test = y_test.astype(int)
        _, group_train = np.unique(qid_train, return_counts=True)
        _, group_test = np.unique(qid_test, return_counts=True)
        return X_train, y_train, qid_train, group_train, X_test, y_test, qid_test, group_test

    def ndcg_single_query(self, y_score, y_true, k):
        order = np.argsort(y_score)[::-1]
        y_true = np.take(y_true, order[:k])

        gain = 2 ** y_true - 1

        discounts = np.log2(np.arange(len(y_true)) + 2)
        return np.sum(gain / discounts)

    # calculate NDCG score given a trained model 
    def compute_ndcg_all(self, model, X_test, y_test, qids_test, k=10):
        unique_qids = np.unique(qids_test)
        ndcg_ = list()
        for i, qid in enumerate(unique_qids):
            y = y_test[qids_test == qid]

            if np.sum(y) == 0:
                continue

            p = model.predict(X_test[qids_test == qid])

            idcg = self.ndcg_single_query(y, y, k=k)
            ndcg_.append(self.ndcg_single_query(p, y, k=k) / idcg)
        return np.mean(ndcg_)

    # get importance of features
    def get_feature_importance(self, model, importance_type='gain'):
        return model.booster_.feature_importance(importance_type=importance_type)