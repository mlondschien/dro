import numpy as np
import lightgbm as lgb


def get_weights(rho, losses, min_weight=0.01):
    losses = losses - np.mean(losses)
    std = np.sqrt(np.mean(losses**2))
    return np.maximum(1 + np.sqrt(2 * rho) * losses / std, min_weight)


class DRO:
    def __init__(
        self, num_boost_round=100, params=None, categorical_feature=None, rho=1, k=2
    ):
        self.num_boost_round = num_boost_round
        self.params = params
        self.categorical_feature = categorical_feature
        self.rho = rho
        self.k = k

    def fit(self, X, y):
        X_arrow = X.to_arrow()
        data = lgb.Dataset(
            data=X_arrow,
            label=y,
            weight=np.ones_like(y, dtype="float"),
            categorical_feature=self.categorical_feature,
            feature_name=X.columns,
        )
        self.booster = lgb.Booster(params=self.params, train_set=data)
        scores = np.zeros_like(y, dtype="float")

        if "objective" in self.params and self.params["objective"] == "binary":
            y_tilde = 2 * y - 1  # y_tilde in {-1, 1}

        for idx in range(self.num_boost_round):
            self.booster.update(train_set=data)  # this fits one additional tree
            # raw_score=True returns the raw scores of the trees if the objective is
            # classification. Probabilities are sigmoid(raw_score). The loss is
            # (y - raw_score)**2 for regression and log(1 + exp(-y_tilde * raw_score))
            # for binary classification.
            scores += self.booster.predict(X_arrow, start_iteration=idx, num_iteration=1, raw_score=True)

            if "objective" in self.params and self.params["objective"] == "binary":
                losses = np.log1p(np.exp(-y_tilde * scores))
            else:
                losses = np.square(y - scores)

            data.weight = get_weights(rho=self.rho, losses=losses)

        return self

    def predict(self, X):
        return self.booster.predict(X)
