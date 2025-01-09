# Data Processing
import pandas as pd
import numpy as np

# Modeling
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from scipy.stats import randint

#this dataset contains CRC samples with diagnoses ranging from 0 (normal) to 4 (cancer), with 3 intermediate stages
#first we will attempt training a RF to predict all of the diagnoses
data = pd.read_csv('~/Desktop/mbVAE/PRJNA318004_disease_countsNorm.tsv',sep='\t')
meta = pd.read_csv('~/Desktop/mbVAE/PRJNA318004_disease_meta.tsv',sep='\t')
metaNP = meta['status'].values
dataNP = data.to_numpy()

x_train, x_test, y_train, y_test = train_test_split(dataNP,metaNP,test_size=0.2)

rf = RandomForestClassifier()
rf.fit(x_train,y_train)

y_pred = rf.predict(x_test)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

param_dist = {'n_estimators': randint(50,500),
              'max_depth': randint(1,20)}

# Create a random forest classifier
rf = RandomForestClassifier()

# Use random search to find the best hyperparameters
rand_search = RandomizedSearchCV(rf,
                                 param_distributions = param_dist,
                                 n_iter=5,
                                 cv=5)

# Fit the random search object to the data
rand_search.fit(x_train, y_train)

best_rf = rand_search.best_estimator_

# Print the best hyperparameters
print('Best hyperparameters:', rand_search.best_params_)

y_pred = best_rf.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print("HP tuned accuracy:", accuracy)

# now we want to see how the prediction accuracy differs if we use only the cancer and normal samples
# so we'll filter the dataset for status == 0 (normal) or status == 4 (cancer)

filtered_meta = meta.query('status == 0 or status == 4')
samples = list(filtered_meta.index)
filtered_metaNP = filtered_meta['status'].values
filtered_data = data.loc[samples]
filtered_dataNP = filtered_data.to_numpy()

# once filtering is complete, we repeat the RF training on the new filtered data
x_train, x_test, y_train, y_test = train_test_split(filtered_dataNP,filtered_metaNP,test_size=0.2)

rf = RandomForestClassifier()
rf.fit(x_train,y_train)

y_pred = rf.predict(x_test)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

param_dist = {'n_estimators': randint(50,500),
              'max_depth': randint(1,20)}

# Create a random forest classifier
rf = RandomForestClassifier()

# Use random search to find the best hyperparameters
rand_search = RandomizedSearchCV(rf,
                                 param_distributions = param_dist,
                                 n_iter=5,
                                 cv=5)

# Fit the random search object to the data
rand_search.fit(x_train, y_train)

best_rf = rand_search.best_estimator_

# Print the best hyperparameters
print('Best hyperparameters:', rand_search.best_params_)

y_pred = best_rf.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print("HP tuned accuracy:", accuracy)