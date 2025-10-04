import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.plotting import scatter_matrix


import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

#STEP 1
file_path = "Project 1 Data.csv"
data = pd.read_csv(file_path)
X = data[["X","Y","Z"]].to_numpy()
y = data["Step"].to_numpy()

print("Shape of the dataset:", data.shape)
print("\nColumn names:", data.columns.tolist())
print("\nFirst 5 rows of the dataset:")
print(data.head())

#STEP 2
#STEP 2.1
print("\n[2.1] Overall descriptive statistics")
print(data.describe().T)

#STEP 2.2
print("\n[2.2] Per-step stats for X, Y, Z")
per_step = (
    data.groupby("Step")[["X","Y","Z"]]
        .agg(["count","mean","std","min","max"])
        .round(4)
)
print("\nX stats by Step:\n", per_step["X"])
print("\nY stats by Step:\n", per_step["Y"])
print("\nZ stats by Step:\n", per_step["Z"])

#STEP 2.3
(unique, counts) = np.unique(y, return_counts=True)
plt.bar(unique, counts, color="purple")
plt.title("Class Distribution by Step")
plt.xlabel("Step")
plt.ylabel("Count")
plt.show()



#STEP 2.4
print("\nSTEP 2.4 – Plots: Boxplots of X, Y, Z by Step")

plt.figure()
data.boxplot(column="X", by="Step")
plt.title("X by Step")
plt.suptitle("")
plt.xlabel("Step")
plt.ylabel("X")
plt.tight_layout()
plt.show()

plt.figure()
data.boxplot(column="Y", by="Step")
plt.title("Y by Step")
plt.suptitle("")
plt.xlabel("Step")
plt.ylabel("Y")
plt.tight_layout()
plt.show()

plt.figure()
data.boxplot(column="Z", by="Step")
plt.title("Z by Step")
plt.suptitle("")
plt.xlabel("Step")
plt.ylabel("Z")
plt.tight_layout()
plt.show()

scatter_matrix(data[["X","Y","Z"]], figsize=(7,7))
plt.suptitle("Scatter matrix")
plt.show()

#STEP3
#STEP 3.1
corr = data[["X","Y","Z","Step"]].corr(method="pearson")
print(corr)

plt.figure(figsize=(6,5))
sns.heatmap(corr, annot=True, fmt=".2f", vmin=-1, vmax=1, cmap="coolwarm")
plt.title("Pearson Correlation Matrix (X, Y, Z, Step)")
plt.tight_layout()
plt.show()

#STEP 3.2
ft = corr["Step"].drop("Step").sort_values(ascending=False)
print("\nCorrelation with Step:\n", ft)

plt.figure(figsize=(5,3))
sns.barplot(x=ft.index, y=ft.values, color="purple")
plt.title("Feature–Target Correlation (Pearson)")
plt.ylabel("corr(feature, Step)")
plt.tight_layout()
plt.show()

#STEP 4
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from scipy.stats import loguniform, randint
from sklearn.linear_model import LogisticRegression


# 4.1 STRATIFIED SPLIT

Xtr, Xte, ytr, yte = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# 4.2 DEFINING MODELS+ GRID
# A) SVM 
svm_pipe = Pipeline([("scaler", StandardScaler()), ("clf", SVC())])
svm_grid = {
    "clf__kernel": ["rbf"],
    "clf__C": [0.1, 1, 10, 100],
    "clf__gamma": ["scale", 0.1, 0.01],
    "clf__class_weight": [None, "balanced"],
}

# B) RANDOM FOREST
rf = RandomForestClassifier(random_state=42)
rf_grid = {
    "n_estimators": [200, 400, 800],
    "max_depth": [None, 4, 8, 12],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 5],
    "class_weight": [None, "balanced"],
}

# C) LOGISITIC REGRESSION
log_pipe = Pipeline([("scaler", StandardScaler()),
                     ("clf", LogisticRegression(max_iter=5000, multi_class="ovr", random_state=42))])
log_grid = {
    "clf__penalty": ["l1", "l2"],
    "clf__C": [0.01, 0.1, 1, 10, 100],
    "clf__solver": ["saga"],
    "clf__class_weight": [None, "balanced"],
}


# 4.3 GRID SEARCHES
svm_cv = GridSearchCV(svm_pipe, svm_grid, cv=cv, n_jobs=-1, scoring="accuracy")
rf_cv  = GridSearchCV(rf, rf_grid, cv=cv, n_jobs=-1, scoring="accuracy")
log_cv = GridSearchCV(log_pipe, log_grid, cv=cv, n_jobs=-1, scoring="accuracy")

for name, gs in [("SVM", svm_cv), ("RF", rf_cv), ("LOG", log_cv)]:
    gs.fit(Xtr, ytr)
    yhat = gs.predict(Xte)
    print(f"\n{name} — best params: {gs.best_params_}")
    print(f"{name} — CV best score: {gs.best_score_:.3f}")
    print(f"{name} — Test accuracy: {accuracy_score(yte, yhat):.3f}")
    print("Confusion matrix:\n", confusion_matrix(yte, yhat))

print(classification_report(yte, yhat, digits=3))

# 4.4 RANDOMIZED SEARCHCV 
svm_rand = RandomizedSearchCV(
    Pipeline([("scaler", StandardScaler()), ("clf", SVC())]),
    {
        "clf__kernel": ["rbf"],
        "clf__C": loguniform(1e-2, 1e3),
        "clf__gamma": loguniform(1e-4, 1),
        "clf__class_weight": [None, "balanced"],
    },
    n_iter=30, cv=cv, n_jobs=-1, random_state=42, scoring="accuracy"
)
svm_rand.fit(Xtr, ytr)
yhat_r = svm_rand.predict(Xte)
print("\nSVM (RandomizedSearchCV) — best params:", svm_rand.best_params_)
print("SVM (Rand) — CV best score:", f"{svm_rand.best_score_:.3f}")
print("SVM (Rand) — Test accuracy:", f"{accuracy_score(yte, yhat_r):.3f}")
print("Confusion matrix:\n", confusion_matrix(yte, yhat_r))
    
#STEP 5
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np

models = {
    "SVM": svm_cv.best_estimator_,
    "RF":  rf_cv.best_estimator_,
    "LOG": log_cv.best_estimator_,
}

def eval_and_plot(name, model, Xte, yte):
    yhat = model.predict(Xte)

    acc = accuracy_score(yte, yhat)
    p, r, f1, support = precision_recall_fscore_support(yte, yhat, average=None, zero_division=0)
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(yte, yhat, average="macro", zero_division=0)
    w_p, w_r, w_f1, _ = precision_recall_fscore_support(yte, yhat, average="weighted", zero_division=0)

    print(f"\n{name}")
    print(f"Accuracy: {acc:.3f}")
    print(f"Macro  —  Precision: {macro_p:.3f}  Recall: {macro_r:.3f}  F1: {macro_f1:.3f}")
    print(f"Weighted— Precision: {w_p:.3f}  Recall: {w_r:.3f}  F1: {w_f1:.3f}")
    print(classification_report(yte, yhat, digits=3, zero_division=0))

    fig, ax = plt.subplots(figsize=(5,4))
    ConfusionMatrixDisplay.from_predictions(yte, yhat, cmap="Blues", ax=ax, colorbar=False)
    ax.set_title(f"{name} — Confusion Matrix")
    plt.tight_layout(); plt.show()

for name, model in models.items():
    eval_and_plot(name, model, Xte, yte)


eval_and_plot("SVM (Rand)", svm_rand.best_estimator_, Xte, yte)

#STEP 6
from sklearn.ensemble import StackingClassifier
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt


svm_best = svm_cv.best_estimator_       
rf_best  = rf_cv.best_estimator_         
log_best = log_cv.best_estimator_        


stack = StackingClassifier(
    estimators=[("svm", svm_best), ("rf", rf_best)],
    final_estimator=log_best.named_steps["clf"],   
    passthrough=False,                              
    n_jobs=-1
)

stack.fit(Xtr, ytr)
yhat_stack = stack.predict(Xte)

print("\nSTACK — Test accuracy:", f"{accuracy_score(yte, yhat_stack):.3f}")
print(classification_report(yte, yhat_stack, digits=3, zero_division=0))

fig, ax = plt.subplots(figsize=(5,4))
ConfusionMatrixDisplay.from_predictions(yte, yhat_stack, cmap="Blues", ax=ax, colorbar=False)
ax.set_title("Stacked Model — Confusion Matrix")
plt.tight_layout(); plt.show()

#STEP 7
from joblib import dump, load
import numpy as np

final_model = svm_rand.best_estimator_      

dump(final_model, "final_model.joblib")

clf = load("final_model.joblib")

X_new = np.array([
    [9.375, 3.0625, 1.51],
    [6.995, 5.125 , 0.3875],
    [0.0  , 3.0625, 1.93],
    [9.4  , 3.0   , 1.8 ],
    [9.4  , 3.0   , 1.3 ]
])

pred = clf.predict(X_new)
print("Predicted Steps:", pred.tolist())