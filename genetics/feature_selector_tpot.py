# genetics/feature_selector_tpot.py

from tpot import TPOTClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler, PowerTransformer
from features.domain_features import load_with_features


def run_tpot_feature_selection():
    X, y = load_with_features()

    X = PowerTransformer().fit_transform(X)
    X = StandardScaler().fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

    tpot = TPOTClassifier(generations=5, population_size=20,
                          verbosity=2, scoring='accuracy', random_state=42,
                          disable_update_check=True)

    tpot.fit(X_train, y_train)

    print("\n📊 Best Pipeline:")
    print(tpot.fitted_pipeline_)

    y_pred = tpot.predict(X_test)
    print("\n📈 Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))

    tpot.export("genetics/tpot_selected_pipeline.py")

if __name__ == "__main__":
    run_tpot_feature_selection()
