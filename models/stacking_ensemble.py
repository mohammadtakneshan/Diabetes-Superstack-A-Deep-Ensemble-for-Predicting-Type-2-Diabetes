# models/stacking_ensemble.py

import numpy as np
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score, roc_auc_score
from imblearn.over_sampling import SMOTE
import lightgbm as lgb
from keras.models import load_model
from features.domain_features import load_with_features


def run_stacking_with_nn(nn_model_path):
    X, y = load_with_features()
    X = PowerTransformer().fit_transform(X)
    X = StandardScaler().fit_transform(X)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    meta_features, meta_labels = [], []

    rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    gbm = GradientBoostingClassifier(n_estimators=100, learning_rate=0.01, max_depth=3, random_state=42)
    lgbm = lgb.LGBMClassifier(n_estimators=100, learning_rate=0.01, max_depth=3, random_state=42)
    nn_model = load_model(nn_model_path)

    for train_idx, test_idx in skf.split(X, y):
        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]

        X_train, y_train = SMOTE(random_state=42).fit_resample(X_train, y_train)

        rf.fit(X_train, y_train)
        gbm.fit(X_train, y_train)
        lgbm.fit(X_train, y_train)
        nn_pred = nn_model.predict(X_test).flatten()

        rf_pred = rf.predict_proba(X_test)[:, 1]
        gbm_pred = gbm.predict_proba(X_test)[:, 1]
        lgbm_pred = lgbm.predict_proba(X_test)[:, 1]

        stacked = np.vstack((nn_pred, rf_pred, gbm_pred, lgbm_pred)).T
        meta_features.append(stacked)
        meta_labels.append(y_test)

    meta_X = np.vstack(meta_features)
    meta_y = np.hstack(meta_labels)

    meta_clf = LogisticRegression()
    meta_clf.fit(meta_X, meta_y)
    meta_probs = meta_clf.predict_proba(meta_X)[:, 1]

    best_thresh, best_f1 = 0.5, 0
    for t in np.arange(0.3, 0.7, 0.01):
        preds = (meta_probs > t).astype(int)
        f1 = f1_score(meta_y, preds)
        if f1 > best_f1:
            best_thresh = t
            best_f1 = f1

    final_preds = (meta_probs > best_thresh).astype(int)
    print(f"\n🎯 Best Threshold: {best_thresh:.2f} with F1-score: {best_f1:.4f}")
    print("\n📊 Classification Report:")
    print(classification_report(meta_y, final_preds))
    print("\n🧾 Confusion Matrix:")
    print(confusion_matrix(meta_y, final_preds))
    print(f"\n🔢 ROC AUC Score: {roc_auc_score(meta_y, meta_probs):.4f}")

if __name__ == "__main__":
    run_stacking_with_nn("models/optuna_best_nn.h5")
