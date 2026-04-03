#%pip install crunch-cli --upgrade --quiet --progress-bar off
#!crunch setup-notebook datacrunch-2 q4TWLiFxUssUY3SAvYiAp8i1


import os 
import joblib
import pandas as pd
import crunch
import numpy as np
import sklearn  # == 1.7.2
from sklearn.linear_model import Ridge, LogisticRegression
import seaborn as sns
import matplotlib.pyplot as plt


#plt.style.use('seaborn-v0_8-whitegrid')


#crunch_tools = crunch.load_notebook()


def get_feature_columns(X: pd.DataFrame):
    return [column for column in X.columns if column.startswith("Feature_")]

# This function just return a list of colomns in the dataframe which names start with "Feature_", 


# Load the data
#X_train, y_train, X_test = crunch_tools.load_data()


#X_train.head()
# no leakage risk: the id changes every moon


#feature_cols = get_feature_columns(X_train)

# Confirm features are bounded in [0, 1]
#print(f"Global min: {X_train[feature_cols].min().min():.2f}")
#print(f"Global max: {X_train[feature_cols].max().max():.2f}")


#y_train.head()


#sns.countplot(x="target", data=y_train, order=[-1.0, 0.0, 1.0])
#plt.title("Target distribution")
#plt.xlabel("Target")
#plt.ylabel("Count")
#plt.show()


# Check for missing values
#cols_with_missing = [col for col in X_train.columns if X_train[col].isnull().any()]
#print(f"{len(cols_with_missing)} columns with missing values — {'imputation needed' if cols_with_missing else 'no imputation needed'}")


#X_train.dtypes.value_counts()
# Only int and float — no categorical variables, no encoding needed


def train(X_train: pd.DataFrame, y_train: pd.DataFrame, model_directory_path: str) -> None:
    feature_columns = get_feature_columns(X_train)
    data = X_train[["moon", "id"] + feature_columns].merge(y_train[["moon", "id", "target"]], on=["moon", "id"])
    top_features_path = os.path.join(model_directory_path, "top_features.joblib")

    if os.path.exists(top_features_path):
        top_features = joblib.load(top_features_path)
    else:
        # Spearman correlation — rapide et efficace pour données financières
        spearman_scores = data[feature_columns].corrwith(data["target"], method="spearman").abs()
        top_features = spearman_scores.nlargest(500).index.tolist()
        joblib.dump(top_features, top_features_path)

    # Étape 1 : classifier non-zero vs zero
    data["is_extreme"] = (data["target"] != 0).astype(int)
    clf = LogisticRegression(C=0.5, max_iter=1000, n_jobs=-1)
    clf.fit(data[top_features], data["is_extreme"])

    # Étape 2 : Ridge sur les non-zero uniquement pour prédire +1 ou -1
    data_extreme = data[data["target"] != 0]
    reg = Ridge(alpha=100)
    reg.fit(data_extreme[top_features], data_extreme["target"])

    joblib.dump(clf, os.path.join(model_directory_path, "clf.joblib"))
    joblib.dump(reg, os.path.join(model_directory_path, "reg.joblib"))
    


def infer(X_test: pd.DataFrame, model_directory_path: str) -> pd.DataFrame:
    prediction = X_test[["id", "moon"]].copy()
    clf = joblib.load(os.path.join(model_directory_path, "clf.joblib"))
    reg = joblib.load(os.path.join(model_directory_path, "reg.joblib"))
    top_features = joblib.load(os.path.join(model_directory_path, "top_features.joblib"))

    # Probabilité d'être un stock extrême (+1 ou -1)
    proba_extreme = clf.predict_proba(X_test[top_features])[:, 1]

    # Direction prédite par Ridge (+1 ou -1)
    direction = reg.predict(X_test[top_features])

    # Prédiction finale : proba_extreme * direction
    # → proche de 0 si le stock est probablement médian
    # → proche de ±1 si le stock est probablement extrême
    prediction["prediction"] = proba_extreme * direction
    prediction["prediction"] = prediction["prediction"].clip(-1, 1)

    return prediction


#crunch_tools.test(force_first_train=True,train_frequency=0)


#prediction = pd.read_parquet("prediction/prediction.parquet") #inder() retourne un fihcier de prédiction, il faut donc le lire 
#prediction


# Load the targets
#y_test = pd.read_parquet("data/y.reduced.parquet",filters=[("moon", "in", prediction["moon"].unique())])
#y_test 


# Define the scoring function (la corrélation de Pearson utilisée par CrunchDAO)
def score(group: pd.DataFrame):
    return group["prediction"].corr(group["target"], method="pearson")

# Merge the prediction with the target y (with moon and id)
#merged = y_test.merge(prediction,on=["moon", "id"])

# Compute the pearson for each moon
#pearson_values = merged.groupby("moon").apply(score, include_groups=False).fillna(0)  # map constants to zero

#print(pearson_values) # voir si certains moons sont très différents des autres
#print(pearson_values.mean())


import seaborn as sns
import matplotlib.pyplot as plt

#fig, ax = plt.subplots(figsize=(10, 4))
#sns.barplot(x=pearson_values.index, y=pearson_values.values, ax=ax)
#ax.axhline(pearson_values.mean(), color="green", linewidth=0.8, linestyle="--", label=f"mean = {pearson_values.mean():.4f}")
#ax.set_title("Pearson correlation per moon")
#ax.set_xlabel("Moon")
#ax.set_ylabel("Pearson")
#ax.legend()
#plt.tight_layout()
#plt.show()
