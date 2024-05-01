import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from sklearn.base import TransformerMixin
class MLP(torch.nn.Module, TransformerMixin):
    def __init__(self):
        # Most of the init is done in fit so they input dimension can be inferred from data
        super().__init__()
    
    def forward(self, X):
        return self.model(X)
    
    def train(self, X, y):
        X = torch.tensor(X, dtype=torch.float32, device='cuda')
        y = torch.tensor(y, dtype=torch.int64, device='cuda')

        self.model.train()
        
        criterion = nn.NLLLoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3, weight_decay=1e-5)

        dataset = TensorDataset(X, y)
        dataloader = DataLoader(dataset, batch_size=128, shuffle=True)

        for epoch in tqdm(range(100)):
            for (X_, y_) in dataloader:
                X_ = X_.cuda()
                y_ = y_.cuda()
                # ===================forward=====================
                output = self(X_)
                loss = criterion(output, y_)
                # ===================backward====================
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        return self
    
    def eval(self, X_test, y_test):
        X_test = torch.tensor(X_test, dtype=torch.float32, device='cuda')
        y_test = torch.tensor(y_test, dtype=torch.int64, device='cuda')
        self.model.eval()
        loss = -1
        with torch.no_grad():
            criterion = nn.NLLLoss()

            output = self(X_test)
            loss = criterion(output, y_test)

        return loss
    
    def set_params():
        pass
    
    def fit(self, X, Y):
        X = torch.tensor(X, dtype=torch.float32, device='cuda')
        Y = torch.tensor(Y, dtype=torch.int64, device='cuda')
        num_features = X.shape[1]
        self.model = nn.Sequential(
            nn.Linear(num_features, 1280),
            nn.ReLU(True),
            nn.Linear(1280, 640),
            nn.ReLU(True), 
            nn.Linear(640, 5),
            nn.LogSoftmax(dim=1)
        )
        self.cuda()
        self.train(X, Y)
        self.labels_ = self(X).cpu().numpy()
        return self

    def transform(self, X):
        X = torch.tensor(X, dtype=torch.float32, device='cuda')
        return self(X).cpu().numpy()