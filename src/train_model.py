from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV

def train_xgb_model(X_train, y_train, param_dist, n_iter=10, scoring="f1"):
    """
    Train an XGBoost model with RandomizedSearchCV.

    Parameters:
    - X_train: Training features
    - y_train: Training target
    - param_dist: Dictionary of hyperparameters to search
    - n_iter: Number of parameter combinations to try
    - scoring: Evaluation metric for model selection

    Returns:
    - best_model: Trained model with best parameters
    - best_params: Dictionary of best parameters
    """
    model = XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42)
    search = RandomizedSearchCV(
        model,
        param_distributions=param_dist,
        n_iter=n_iter,
        scoring=scoring,
        cv=3,
        verbose=1,
        random_state=42,
        n_jobs=-1
    )
    search.fit(X_train, y_train)
    return search.best_estimator_, search.best_params_