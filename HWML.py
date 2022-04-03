# %%
import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import plot_confusion_matrix
from sklearn.tree import DecisionTreeClassifier 
from sklearn import metrics 
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_regression

from imblearn.over_sampling import SMOTE, ADASYN

from xgboost import XGBClassifier,plot_importance

from catboost import Pool, CatBoostClassifier
from catboost.utils import get_confusion_matrix,eval_metric


# %% [markdown]
# ### Veri Tanımı
# - 10 Farklı Kategoriye sahip, 11 Tane Farklı Feature Sahip olan bir datasete sahibiz. Veri imbalance ve outlier değerler bulunuyor.

# %% [markdown]
# ### Veri tipleri ve nullable durumu

# %%
data = pd.read_csv('WineQT.csv').drop(columns=['Id'])
data.info()
quality = data['quality']

# %% [markdown]
# ### Her bir kategorinin yüzdelik dilimleri

# %%
weight = pd.DataFrame(data.groupby('quality').count()['fixed acidity'])
total = pd.DataFrame(data.groupby('quality').count()['fixed acidity']).sum()
classWeights = []
for index, row in weight.iterrows():
    classWeights.append((row['fixed acidity']/total).values[0])
    print(f"Quality: {index}, Count: {row['fixed acidity']}, Percent of Dataset: {(row['fixed acidity']/total).values[0]}")

# %% [markdown]
# ### Toplam datanın yüzde kaçı null 

# %%
def missing_values_table(df):
    mis_val = df.isnull().sum()
    mis_val_percent = 100 * df.isnull().sum() / len(df)
    mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
    mis_val_table_ren_columns = mis_val_table.rename(
    columns = {0 : 'Missing Values', 1 : '% of Total Values'})
    mis_val_table_ren_columns = mis_val_table_ren_columns[
        mis_val_table_ren_columns.iloc[:,1] != 0].sort_values(
    '% of Total Values', ascending=False).round(1)
    print ("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"      
        "There are " + str(mis_val_table_ren_columns.shape[0]) +
            " columns that have missing values.")
    return mis_val_table_ren_columns

missing_values_table(data)

# %% [markdown]
# ### Basit istatistiksel değerler

# %%
data.describe()

# %% [markdown]
# ### Her bir feature için box plat
# ### IQR, QR, Outlier

# %%
fig, axes = plt.subplots(4, 3, figsize=(54, 36))

sns.boxplot(ax=axes[0, 0], data=data, x='quality', y='fixed acidity')
sns.boxplot(ax=axes[0, 1], data=data, x='quality', y='volatile acidity')
sns.boxplot(ax=axes[0, 2], data=data, x='quality', y='citric acid')
sns.boxplot(ax=axes[1, 0], data=data, x='quality', y='residual sugar')
sns.boxplot(ax=axes[1, 1], data=data, x='quality', y='chlorides')
sns.boxplot(ax=axes[1, 2], data=data, x='quality', y='free sulfur dioxide')
sns.boxplot(ax=axes[2, 0], data=data, x='quality', y='total sulfur dioxide')
sns.boxplot(ax=axes[2, 1], data=data, x='quality', y='density')
sns.boxplot(ax=axes[2, 2], data=data, x='quality', y='pH')
sns.boxplot(ax=axes[3, 0], data=data, x='quality', y='sulphates')
sns.boxplot(ax=axes[3, 1], data=data, x='quality', y='alcohol')

# %% [markdown]
# ### Baskılama İşlemi

# %%
for i in data.columns:
    if i != 'quality':
        Q1 = data[i].quantile(0.25)
        Q3 = data[i].quantile(0.75)
        IQR = Q3-Q1
        data[i][data[i] < (Q1 - 1.5 * IQR)] = Q1 - 2.5 * IQR #np.median(data[i])#Q1 - 1.5 * IQR
        data[i][data[i] > (Q3 + 1.5 * IQR)] = Q1 - 2.5 * IQR #np.median(data[i])#Q3 + 1.5 * IQR

# %% [markdown]
# ### Baskılandıktan sonra verinin durumu 

# %%
fig, axes = plt.subplots(4, 3, figsize=(54, 36))
sns.boxplot(ax=axes[0, 0], data=data, x='quality', y='fixed acidity')
sns.boxplot(ax=axes[0, 1], data=data, x='quality', y='volatile acidity')
sns.boxplot(ax=axes[0, 2], data=data, x='quality', y='citric acid')
sns.boxplot(ax=axes[1, 0], data=data, x='quality', y='residual sugar')
sns.boxplot(ax=axes[1, 1], data=data, x='quality', y='chlorides')
sns.boxplot(ax=axes[1, 2], data=data, x='quality', y='free sulfur dioxide')
sns.boxplot(ax=axes[2, 0], data=data, x='quality', y='total sulfur dioxide')
sns.boxplot(ax=axes[2, 1], data=data, x='quality', y='density')
sns.boxplot(ax=axes[2, 2], data=data, x='quality', y='pH')
sns.boxplot(ax=axes[3, 0], data=data, x='quality', y='sulphates')
sns.boxplot(ax=axes[3, 1], data=data, x='quality', y='alcohol')

# %% [markdown]
# ### Genel datanın dağılımı

# %%
sns.pairplot(data,hue='quality',corner=True)

# %% [markdown]
# ### Her bir feature ın ne kadar önemli olduğunu belirliyoruz 
# ***Burada extract ettiğim featureları model için kullanmayacağım***

# %% [markdown]
# ***XGBoostClassifier Feature Importance***

# %%
model = XGBClassifier()
model.fit(data, quality)

plot_importance(model)
plt.show()

# %% [markdown]
# ***Her bir feature'ın birbiri ile  açıklanabilirliği-mutual information-***

# %%
def make_mi_scores(X, y):
    mi_scores = mutual_info_regression(X, y)
    mi_scores = pd.Series(mi_scores, name="Mutual Information Scores", index=X.columns)
    mi_scores = mi_scores.sort_values(ascending=False)
    return mi_scores

mi_scores = make_mi_scores(data, quality)
mi_scores

def plot_mi_scores(scores):
    scores = scores.sort_values(ascending=True)
    width = np.arange(len(scores))
    ticks = list(scores.index)
    plt.barh(width, scores)
    plt.yticks(width, ticks)
    plt.title("Mutual Information Scores")


plt.figure(dpi=100, figsize=(8, 5))
plot_mi_scores(mi_scores)

# %% [markdown]
# ### Train-Test dataset ayrımı

# %%
scaler=StandardScaler()
data=scaler.fit_transform(data)
X_train, X_test, y_train, y_test = train_test_split(data, quality, test_size=0.10, random_state=13)

# %% [markdown]
# ### Oversample yapmadan önceki DecisionTree

# %%
# Create Decision Tree classifer object
clf = DecisionTreeClassifier()

# Train Decision Tree Classifer
clf = clf.fit(X_train,y_train)

#Predict the response for test dataset
y_pred = clf.predict(X_test)

# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

# %% [markdown]
# ### Oversample yapmadan XGBoost classifier ve confusion matrix

# %%

model = XGBClassifier()
model.fit(X_train, y_train,compute_sample_weight("balanced", y_train))
# make predictions for test data
y_pred = model.predict(X_test)
# evaluate predictions
accuracy = accuracy_score(y_pred, y_test)
print("Accuracy: %.2f%%" % (accuracy * 100.0))
plot_confusion_matrix(model, X_test, y_test) 

# %% [markdown]
# ### Oversample yapmadan CatBoost classifier ve confusion matrix

# %%
train_dataset = Pool(data=X_train,
                     label=y_train)

eval_dataset = Pool(data=X_test,
                    label=y_test)
model = CatBoostClassifier(iterations=10,
                           learning_rate=1,
                           depth=1,
                           loss_function='MultiClass')
# Fit model
model.fit(train_dataset)
# Get predicted classes
preds_class = model.predict(eval_dataset)
# Get predicted probabilities for each class
preds_proba = model.predict_proba(eval_dataset)

# Get Acc
accuracy = accuracy_score(preds_class, y_test)
print("Accuracy: %.2f%%" % (accuracy * 100.0))
#Get Confusion Matrix
plot_confusion_matrix(model, X_test, y_test) 

# %% [markdown]
# ### Normalizasyon, SMOTE  ve Train-Test Split

# %%
X_resampled_smote, y_resampled_smote = SMOTE(sampling_strategy='not majority').fit_resample(data, quality)
#X_resampled_adasyn, y_resampled_adasyn = ADASYN(sampling_strategy='not majority').fit_resample(data, quality)

scaler=StandardScaler()
X_resampled_smote = scaler.fit_transform(X_resampled_smote)
#scaler=StandardScaler()
#X_resampled_adasyn = scaler.fit_transform(X_resampled_adasyn)

X_train_smote, X_test_smote, y_train_smote, y_test_smote = train_test_split(X_resampled_smote, y_resampled_smote, test_size=0.10, random_state=13)
#X_train_adasyn, X_test_adasyn, y_train_adasyn, y_test_adasyn = train_test_split(X_resampled_adasyn, y_resampled_adasyn, test_size=0.10, random_state=13)

# %% [markdown]
# ### Oversample  XGBoost classifier 'Balanced olarak ayarlı' ve confusion matrix

# %%
model = XGBClassifier()
model.fit(X_train_smote, y_train_smote,compute_sample_weight("balanced", y_train_smote))
# make predictions for test data
y_pred = model.predict(X_test_smote)
# evaluate predictions
accuracy = accuracy_score(y_pred, y_test_smote)
print("Accuracy: %.2f%%" % (accuracy * 100.0))
plot_confusion_matrix(model, X_test_smote, y_test_smote) 

# %% [markdown]
# ### Oversample DecisionTree  ve confusion matrix

# %%
# Create Decision Tree classifer object
clf = DecisionTreeClassifier()

# Train Decision Tree Classifer
clf = clf.fit(X_train_smote,y_train_smote)

#Predict the response for test dataset
y_pred = clf.predict(X_test_smote)

# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test_smote, y_pred))
plot_confusion_matrix(clf, X_test_smote, y_test_smote) 

# %%




# %%
