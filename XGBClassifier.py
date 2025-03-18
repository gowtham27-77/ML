#!/usr/bin/env python
# coding: utf-8

# In[ ]:


pip install xgboost


# In[ ]:


import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


# In[ ]:


df = pd.read_csv('diabetes.csv')
df


# In[ ]:


# split features and target
X = df.drop('class', axis=1)
y = df['class']
# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[ ]:


# feature scaling
scaler = StandardScaler()
X_train_scaled = Scaler.fit_transform(X_train)
X_test_scaled = scaler.transfrom(X_test)
print(X_train_scaled)
print("----------------------------------")
print((X_test_scaled)


# In[ ]:


# XGBoost classifier Instantiation with hyper parameter grid

xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)

# Hyperparameter grid
param_grid = {
    n_estimators': [100, 150, 200, 300],
    learning_rate': [0.01, 0.1,0.15],
    'max_depth': [2, 3, 4, 5],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}

# Stratified K-Fold
skf = StratifiedKFold(n_splts=5, shuffle=True, random_state=42)

# GridSearchCV with scoring = recalL
grid_search = GridSearchCV(estimator=xgb,
                           param_grid=param_grid,
                           scoring='recall',
                           cv=skf,
                           verbose=1,
                           n_jobs =- 1)


# In[ ]:


# fit the model with train data
grid_search.fit(X_train_scaled, y_train)
best_model = grid_search.best_estimator_
print("Best Parameters:", grid_search.best_params_)
print("Best Cross-Validated Recall:", grid_search.best_score_)
Y_pred = best_model.predict(X_test_scaled)


# In[5]:


# Evaluation 
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("nXClassification Report:\n", classification_report(y_test, y_pred))


# #### * Identify Feature importance scores using XGBClassifier*

# In[ ]:


best_model.feature_importances_


# In[ ]:


features = pd.Dataframe(best_model.feature_importances_, index = df.iloc[:,:-1].columns, columns=["Importances"])
df1 = features.sort_values(by = "Importances")
df1


# In[10]:


import seaborn as sns
sns.barplot(data = df1, x= features.index, y= "Importances", hue = features.index,palette= "Set1")


# In[ ]:




