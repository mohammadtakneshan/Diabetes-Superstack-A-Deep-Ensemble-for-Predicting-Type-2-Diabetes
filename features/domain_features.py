# features/domain_features.py

import pandas as pd
import numpy as np

def add_domain_features(df):
    df['Glucose_BMI'] = df['Glucose'] * df['BMI']
    df['Age_BMI'] = df['Age'] * df['BMI']
    df['Glucose_Insulin'] = df['Glucose'] * df['Insulin']
    df['SkinThickness_BMI'] = df['SkinThickness'] / (df['BMI'] + 1e-5)
    df['Glucose_squared'] = df['Glucose'] ** 2
    df['BMI_squared'] = df['BMI'] ** 2
    df['Age_squared'] = df['Age'] ** 2
    df['Glucose_to_Insulin'] = df['Glucose'] / (df['Insulin'] + 1e-5)
    df['BMI_by_Age'] = df['BMI'] / (df['Age'] + 1e-5)

    # Clinical flags
    df['High_BMI'] = (df['BMI'] > 30).astype(int)
    df['High_Glucose'] = (df['Glucose'] > 140).astype(int)
    df['Older'] = (df['Age'] > 50).astype(int)

    return df

def load_with_features(filepath="data/pima-indians-diabetes.csv"):
    df = pd.read_csv(filepath, header=None)
    df.columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
                  'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']

    for col in ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']:
        df[col] = df[col].replace(0, np.nan)
        df[col] = df[col].fillna(df[col].median())

    df = add_domain_features(df)
    return df.drop('Outcome', axis=1), df['Outcome'].values

if __name__ == "__main__":
    X, y = load_with_features()
    print(X.head())
