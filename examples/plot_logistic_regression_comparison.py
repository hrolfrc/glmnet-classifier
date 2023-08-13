"""

======================================
Compare SPPAM with LogisticRegression
======================================

A comparison of LogisticRegression and :class:`SPPAM` over 20
synthetic classification problems.

"""

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from sppam import SPPAM

methods = [
    ('Logit', LogisticRegressionCV()),
    ('SPPAM', SPPAM())
]

score = {}
for desc, _ in methods:
    score[desc] = {}
    score[desc]['AUC'] = []
    score[desc]['Accuracy'] = []

rng = np.random.RandomState(11)
for _ in range(20):
    # Make a classification problem
    X, y_d = make_classification(
        n_samples=1000,
        n_features=100,
        n_informative=50,
        n_redundant=20,
        n_classes=2,
        hypercube=True,
        random_state=rng
    )
    scaler = StandardScaler()
    X_d = scaler.fit_transform(X)

    for desc, clf in methods:
        lp = clf.fit(X_d, y_d).predict_proba(X_d)
        auc = roc_auc_score(y_true=y_d, y_score=clf.fit(X_d, y_d).predict_proba(X_d)[:, 1])
        acc = accuracy_score(y_true=y_d, y_pred=clf.fit(X_d, y_d).predict(X_d))
        score[desc]['AUC'].append(auc)
        score[desc]['Accuracy'].append(acc)
        print('finished: ', desc)

# compare the mean of the differences of auc
diff = np.subtract(score['Logit']['AUC'], score['SPPAM']['AUC'])
df_describe = pd.DataFrame(diff)

# plot the results
fig, axs = plt.subplots(3, 1, layout='constrained')
xdata = np.arange(len(score['Logit']['AUC']))
axs[0].plot(xdata, score['Logit']['AUC'], label='LogisticRegressionCV')
axs[0].plot(xdata, score['SPPAM']['AUC'], label='SPPAM')

axs[0].set_title('Comparison of Calf and LogisticRegressionCV')
axs[0].set_ylabel('AUC')
axs[0].legend()

axs[1].plot(xdata, score['Logit']['Accuracy'], label='LogisticRegressionCV')
axs[1].plot(xdata, score['SPPAM']['Accuracy'], label='SPPAM')
axs[1].set_ylabel('Accuracy')
axs[1].legend()

axs[2].hist(diff)
axs[2].set_ylabel('AUC difference')
stats = pd.DataFrame(diff).describe().loc[['mean', 'std']].to_string(header=False)
axs[2].text(.1, 2, stats)
fig.set_size_inches(18.5, 20)
plt.show()

