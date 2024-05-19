import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
from surprise.model_selection import train_test_split
from project3.utils.ExperimentBase import ExperimentBase

class SweepExperiment ():
    def __init__(self, experiment: ExperimentBase) -> None:
        self.base_experiment = experiment

    def sweep(self, dataset, parameter: str, param_range, refit=True):
        self.rmses = []
        self.maes = []
        self.parameter = parameter
        self.param_range = param_range
        for param_value in param_range:
            params = {
                parameter: param_value
            }
            experiment = self.base_experiment(**params)
            exp_result = experiment.run(dataset)
            self.rmses.append(exp_result['rmse'])
            self.maes.append(exp_result['mae'])
        
        if refit:
            self.min_k = self.find_min_best_param(0.001, ['rmse', 'mae'])
            best_params = {
                parameter: self.min_k
            }
            self.best_experiment = self.base_experiment(**best_params)
            trainset, testset = train_test_split(dataset, test_size=0.1)
            self.best_predictions = self.best_experiment.fit_and_predict(trainset, testset)

    def find_min_best_param(self, eps, metrics):
        '''
        should return the value of the parameter sweeped across that produces the best results (min errors)
        ties are broken by taking the lowest value of the parameter
        eps is the epsilon move in the errors to be considered a better result
        '''
        best_ks = []
        for metric in metrics:
            search_list = None
            if metric == 'rmse':
                search_list = self.rmses
            elif metric == 'mae':
                search_list = self.maes
            else:
                raise ValueError('Unsupported metric')
            
            best_idx = np.argmin(search_list)
            min_met = search_list[best_idx]
            i = best_idx
            while True:
                if i == 0:
                    best_ks.append(self.param_range[i])
                    break
                met = search_list[i]
                if np.abs(min_met - met) < eps:
                    i -= 1
                else:
                    best_ks.append(self.param_range[i+1])
                    break
            # past = np.inf
            # for idx in range(len(self.param_range)):
            #     curr = search_list[idx]
            #     if curr <= past - eps:
            #         past = curr
            #     else:
            #         best_ks.append(self.param_range[idx-1])
            #         break
        if len(best_ks) == 0:
            return max(self.param_range)
        return max(best_ks)
    
    def find_min_metric(self, min_k, metric):
        idx = list(self.param_range).index(min_k)
        if metric == 'rmse':
            return self.rmses[idx]
        elif metric == 'mae':
            return self.maes[idx]
        else:
            raise ValueError('Unsupported metric')
        
    def plot_metrics(self, metrics, title_add):
        for metric in metrics:
            if metric == 'rmse':
                plt.plot(self.param_range, self.rmses, color='blue', label="RMSE")
                min_k_rmse = self.find_min_best_param(0.001, ['rmse'])
                min_rmse = self.find_min_metric(min_k_rmse, 'rmse')
                plt.axhline(min_rmse, color='blue', linestyle='dashed')
                plt.plot([min_k_rmse], [min_rmse], marker="o", color='blue')
                annotation = f"({min_k_rmse}, {min_rmse:.3f})"
                plt.annotate(text=annotation, xy=(min_k_rmse, min_rmse))
            elif metric == 'mae':
                plt.plot(self.param_range, self.maes, color='orange', label="MAE")
                min_k_mae = self.find_min_best_param(0.001, ['mae'])
                min_mae = self.find_min_metric(min_k_mae, 'mae')
                plt.axhline(min_mae, color='orange', linestyle='dashed')
                plt.plot([min_k_mae], [min_mae], marker="o", color='orange')
                annotation = f"({min_k_mae}, {min_mae:.3f})"
                plt.annotate(text=annotation, xy=(min_k_mae, min_mae))
            else:
                raise ValueError("Unsupported metric")
        
        plt.legend()
        plt.suptitle(str(self.base_experiment.__name__))
        plt.title(f"Average Metric for 10-Fold CV " + title_add)
        plt.xlabel(self.parameter)
        plt.ylabel('Metric Value')
        plt.show()

    def plot_rocs(self, thresholds, title_add):
        for threshold in thresholds:
            y_true = []
            y_score = []
            for pred in self.best_predictions:
                binary_class = 1 if pred.r_ui >= threshold else 0
                y_true.append(binary_class)
                y_score.append(pred.est)

            fpr, tpr, _ = roc_curve(y_true, y_score)
            roc_auc = roc_auc_score(y_true, y_score)

            plt.plot(fpr, tpr, label=f"ROC Curve Threshold={threshold} (AUC={roc_auc:.3f})")

        plt.suptitle(str(self.base_experiment.__name__))
        plt.title(f"ROC Plots (k={self.min_k}) " + title_add)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend()
        plt.show()

            