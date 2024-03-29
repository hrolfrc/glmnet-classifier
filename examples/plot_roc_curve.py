"""
=============================
Plot the ROC curve
=============================

An example plot of the Receiver Operating Characteristic (ROC)
curve for :class:`GlmnetClassifier` on the breast cancer dataset.  We
want an area under the curve (AUC) that is near 1.

"""
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import RocCurveDisplay
from sklearn.model_selection import train_test_split

from glmnet_classifier import GlmnetClassifier

X, y = load_breast_cancer(return_X_y=True)
classifier = GlmnetClassifier()
X_train, X_test, y_train, y_test = train_test_split(X, y)
y_score = classifier.fit(X_train, y_train).predict_proba(X_test)

RocCurveDisplay.from_predictions(
    y_test,
    y_score[:, 1],
    name="Has breast cancer",
    color="darkorange"
)

plt.axis("square")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC curves")
plt.legend()
plt.show()
