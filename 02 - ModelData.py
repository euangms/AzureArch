import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import OrderedDict
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

titanic_df = pd.read_csv('C:\\dev\Demos\\AzureArch\\Exercise\\clean_titanic.csv')

target_df = titanic_df['survived']
features_df = titanic_df.drop(['survived'], axis = 1)
                                 
# Create training and test sets
X_train, X_test, y_train, y_test = train_test_split(features_df, target_df, test_size = 0.3)

###############################################################################
# Logistic Regression : 
lr = LogisticRegression()

lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)

lr_accuracy = lr.score(X_test, y_test)
prob = lr.predict_proba(X_test)[:,1]

###############################################################################
# Random Forest
rfc = RandomForestClassifier(n_estimators = 100)

rfc.fit(X_train, y_train)
y_pred = rfc.predict(X_test)

rfc_accuracy = rfc.score(X_test, y_test)

###############################################################################
#Decision Tree
dtree = DecisionTreeClassifier(max_depth=10)

dtree.fit(X_train,  y_train)
y_pred = dtree.predict(X_test)

dtree_accuracy = dtree.score(X_test,  y_test)

#Compare Results
print("Logistic Regression: " + str(round(lr_accuracy,2)))
print("Random Forest: " + str(round(rfc_accuracy,2)))
print("Decision Tree: " + str(round(dtree_accuracy,2)))
    
#Features importance
FeaturesImportance(features_df,rfc)
FeaturesImportance(features_df,dtree)


def FeaturesImportance(data,model):
    features = data.columns.tolist()
    fi = model.feature_importances_
    sorted_features = {}
    for feature, imp in zip(features, fi):
        sorted_features[feature] = round(imp,3)

    sorted_features = OrderedDict(sorted(sorted_features.items(),reverse=True, key=lambda t: t[1]))

    dfvi = pd.DataFrame(list(sorted_features.items()), columns=['Features', 'Importance'])
    plt.figure(figsize=(15, 5))
    sns.barplot(x='Features', y='Importance', data=dfvi);
    plt.xticks(rotation=90) 
    plt.show()
