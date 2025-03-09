import numpy as np
import lightgbm as lgb

def get_weights(rho, losses, min_weight=0.01):
    losses = losses - np.mean(losses)
    std = np.sqrt(np.mean(losses ** 2))
    return np.maximum(1 + np.sqrt(2 * rho) * losses / std, min_weight)

class DRO():

    def __init__(self, num_boost_round=100, params=None, categorical_feature=None, rho=1, k=2):
        self.num_boost_round = num_boost_round
        self.params = params
        self.categorical_feature = categorical_feature
        self.rho = rho
        self.k = k

    def fit(self, X, y):
        data = lgb.Dataset(
            data=X,
            label=y,
            weight=np.ones_like(y),
            categorical_feature=self.categorical_feature
        )
        self.booster = lgb.Booster(
            params=self.params, train_set=data
        )
        residuals = y.copy()

        for idx in range(self.num_boost_round):
            self.booster.update(
                train_set = data,
            )
            residuals -= self.booster.predict(X, start_iteration=idx, num_iteration=1)
            losses = residuals ** 2
            data.weight = get_weights(rho = self.rho, losses=losses)
        
        return self
    
    def predict(self, X):
        return self.booster.predict(X)


