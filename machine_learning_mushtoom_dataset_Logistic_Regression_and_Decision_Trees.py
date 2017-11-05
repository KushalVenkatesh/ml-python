
# coding: utf-8

# In[1]:

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns


# Relevant information:
#     
# This data set includes descriptions of hypothetical samples
#     corresponding to 23 species of gilled mushrooms in the Agaricus and
#     Lepiota Family (pp. 500-525).  Each species is identified as
#     definitely edible, definitely poisonous, or of unknown edibility and
#     not recommended.  This latter class was combined with the poisonous
#     one.  The Guide clearly states that there is no simple rule for
#     determining the edibility of a mushroom; no rule like ``leaflets
#     three, let it be'' for Poisonous Oak and Ivy.
# 
#  Number of Instances: 8124
# 
#  Number of Attributes: 22 (all nominally valued)
# 
#  Attribute Information: (classes: edible=e, poisonous=p)
#      1. cap-shape:                bell=b,conical=c,convex=x,flat=f,
#                                   knobbed=k,sunken=s
#      2. cap-surface:              fibrous=f,grooves=g,scaly=y,smooth=s
#      3. cap-color:                brown=n,buff=b,cinnamon=c,gray=g,green=r,
#                                   pink=p,purple=u,red=e,white=w,yellow=y
#      4. bruises?:                 bruises=t,no=f
#      5. odor:                     almond=a,anise=l,creosote=c,fishy=y,foul=f,
#                                   musty=m,none=n,pungent=p,spicy=s
#      6. gill-attachment:          attached=a,descending=d,free=f,notched=n
#      7. gill-spacing:             close=c,crowded=w,distant=d
#      8. gill-size:                broad=b,narrow=n
#      9. gill-color:               black=k,brown=n,buff=b,chocolate=h,gray=g,
#                                   green=r,orange=o,pink=p,purple=u,red=e,
#                                   white=w,yellow=y
#     10. stalk-shape:              enlarging=e,tapering=t
#     11. stalk-root:               bulbous=b,club=c,cup=u,equal=e,
#                                   rhizomorphs=z,rooted=r,missing=?
#     12. stalk-surface-above-ring: fibrous=f,scaly=y,silky=k,smooth=s
#     13. stalk-surface-below-ring: fibrous=f,scaly=y,silky=k,smooth=s
#     14. stalk-color-above-ring:   brown=n,buff=b,cinnamon=c,gray=g,orange=o,
#                                   pink=p,red=e,white=w,yellow=y
#     15. stalk-color-below-ring:   brown=n,buff=b,cinnamon=c,gray=g,orange=o,
#                                   pink=p,red=e,white=w,yellow=y
#     16. veil-type:                partial=p,universal=u
#     17. veil-color:               brown=n,orange=o,white=w,yellow=y
#     18. ring-number:              none=n,one=o,two=t
#     19. ring-type:                cobwebby=c,evanescent=e,flaring=f,large=l,
#                                   none=n,pendant=p,sheathing=s,zone=z
#     20. spore-print-color:        black=k,brown=n,buff=b,chocolate=h,green=r,
#                                   orange=o,purple=u,white=w,yellow=y
#     21. population:               abundant=a,clustered=c,numerous=n,
#                                   scattered=s,several=v,solitary=y
#     22. habitat:                  grasses=g,leaves=l,meadows=m,paths=p,
#                                   urban=u,waste=w,woods=d   
#                                   
#      Missing Attribute Values: 2480 of them (denoted by "?"), all for
#    attribute #11.                             

# In[64]:

#Applying data transformation techniques in excel - performing data formatting and adding header to the mushroom input data

data = pd.read_csv("mushroom.csv")
data.head(6)



# In[3]:

data.isnull().sum()


# In[4]:

data['class'].unique()


# In[5]:

data.shape


# Here is the machine learning application that won’t let people eat poisonous mushrooms

# In[6]:

from sklearn.preprocessing import LabelEncoder
labelencoder=LabelEncoder()
for col in data.columns:
    data[col] = labelencoder.fit_transform(data[col])
 
data.head()


# In[7]:

data['stalk-color-above-ring'].unique()


# In[8]:



print(data.groupby('class').size())



# Class Distribution: 
#     --    edible: 4208 (51.8%)
#     -- poisonous: 3916 (48.2%)
#     --     total: 8124 instances

# Class 0 represents total number of edible mushrooms, which is 51.8% of total records in the dataset 
# and calss 1 represents toal number of poisonous mushrooms, which is 48.2% of the total records in the dataset

# boxplot to see distribution of the data

# In[9]:

ax = sns.boxplot(x='class', y='stalk-color-above-ring', 
                data=data)
ax = sns.stripplot(x="class", y='stalk-color-above-ring',
                   data=data, jitter=True,
                   edgecolor="gray")
sns.plt.title("Class w.r.t stalkcolor above ring",fontsize=12)


# Seperating features and label
# 

# All rows, all the features and no labels

# In[10]:



X = data.iloc[:,1:23]  




# All rows and label only

# In[ ]:

y = data.iloc[:, 0] 
X.head()
y.head()


# In[11]:

X.describe()


# In[12]:

y.head()


# 
# 
# L1 and L2 are regularization parameters.
# 
# They're used to avoid overfiting.Both L1 and L2 regularization prevents overfitting by shrinking i.e. by imposing a penalty on the coefficients.
# 
# L1 is the first moment norm |x1-x2| (|w| for regularization case) that is simply the absolute dıstance between two points where L2 is second moment norm corresponding to Eucledian Distance that is |x1-x2|^2 (|w|^2 for regularization case).
# 
# In simple words,L2 (Ridge) shrinks all the coefficients by the same proportions but eliminates none, while L1 (Lasso) can shrink some coefficients to zero, performing variable selection. 
# 
# If all the features are correlated with the label, ridge outperforms lasso, as the coefficients are never zero in ridge. 
# 
# If only a subset of features are correlated with the label, lasso outperforms ridge as in lasso model some coefficients can be shrunken to zero.
# 

# Looking at the correlation:

# In[13]:

data.corr()


# Standardizing data

# Scaling the data between -1 and 1

# In[14]:



from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X=scaler.fit_transform(X)
X



# 
# Splitting the data into training and testing dataset

# In[16]:



from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=4)



# Logistic Regression - Default Model

# In[17]:

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn import metrics

model_LR= LogisticRegression()


# In[18]:



model_LR.fit(X_train,y_train)



# This will give the positive class prediction probabilities  

# In[19]:

y_prob = model_LR.predict_proba(X_test)[:,1] 



# This will threshold the probabilities to give class predictions.

# In[63]:

y_pred = np.where(y_prob > 0.5, 1, 0) 
model_LR.score(X_test, y_pred)


# In[20]:

confusion_matrix=metrics.confusion_matrix(y_test,y_pred)
confusion_matrix


# In[21]:

auc_roc=metrics.roc_auc_score(y_test,y_pred)
auc_roc


# In[22]:

from sklearn.metrics import roc_curve, auc
false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(false_positive_rate, true_positive_rate)
roc_auc


# In[23]:



import matplotlib.pyplot as plt
plt.figure(figsize=(10,10))
plt.title('Receiver Operating Characteristic')
plt.plot(false_positive_rate,true_positive_rate, color='red',label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],linestyle='--')
plt.axis('tight')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')



# Logistic Regression - Tuned model

# In[24]:



from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn import metrics

LR_model= LogisticRegression()

tuned_parameters = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000] ,
              'penalty':['l1','l2']
                   }



# looking at the correlation

# In[25]:

data.corr()


# In[26]:

from sklearn.model_selection import GridSearchCV

LR= GridSearchCV(LR_model, tuned_parameters,cv=10)


# In[27]:

LR.fit(X_train,y_train)


# In[28]:

print(LR.best_params_)


# In[29]:

y_prob = LR.predict_proba(X_test)[:,1] # This will give the positive class prediction probabilities  
y_pred = np.where(y_prob > 0.5, 1, 0) # This will threshold the probabilities to give class predictions.
LR.score(X_test, y_pred)


# In[30]:



confusion_matrix=metrics.confusion_matrix(y_test,y_pred)
confusion_matrix



# In[31]:



auc_roc=metrics.classification_report(y_test,y_pred)
auc_roc



# In[32]:

auc_roc=metrics.roc_auc_score(y_test,y_pred)
auc_roc


# In[33]:

from sklearn.metrics import roc_curve, auc
false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(false_positive_rate, true_positive_rate)
roc_auc


# In[34]:



import matplotlib.pyplot as plt
plt.figure(figsize=(10,10))
plt.title('Receiver Operating Characteristic')
plt.plot(false_positive_rate,true_positive_rate, color='red',label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],linestyle='--')
plt.axis('tight')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')



# In[35]:

LR_ridge= LogisticRegression(penalty='l2')
LR_ridge.fit(X_train,y_train)


# In[ ]:

y_prob = LR_ridge.predict_proba(X_test)[:,1] # This will give the positive class prediction probabilities  
y_pred = np.where(y_prob > 0.5, 1, 0) # This will threshold the probabilities to give the class predictions.
LR_ridge.score(X_test, y_pred)


# In[37]:



confusion_matrix=metrics.confusion_matrix(y_test,y_pred)
confusion_matrix



# In[38]:



auc_roc=metrics.classification_report(y_test,y_pred)
auc_roc



# In[39]:

auc_roc=metrics.roc_auc_score(y_test,y_pred)
auc_roc


# In[40]:

from sklearn.metrics import roc_curve, auc
false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(false_positive_rate, true_positive_rate)
roc_auc


# In[41]:



import matplotlib.pyplot as plt
plt.figure(figsize=(10,10))
plt.title('Receiver Operating Characteristic')
plt.plot(false_positive_rate,true_positive_rate, color='red',label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],linestyle='--')
plt.axis('tight')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')



# Default Decision Tree model

# In[42]:

from sklearn.tree import DecisionTreeClassifier

model_tree = DecisionTreeClassifier()


# In[43]:

model_tree.fit(X_train, y_train)


# In[44]:

y_prob = model_tree.predict_proba(X_test)[:,1] # This will give the positive class prediction probabilities  
y_pred = np.where(y_prob > 0.5, 1, 0) # This will threshold the probabilities to give class predictions.
model_tree.score(X_test, y_pred)


# In[45]:



confusion_matrix=metrics.confusion_matrix(y_test,y_pred)
confusion_matrix



# In[46]:



auc_roc=metrics.classification_report(y_test,y_pred)
auc_roc



# 
# 
# auc_roc=metrics.roc_auc_score(y_test,y_pred)
# auc_roc
# 
# 

# In[48]:

from sklearn.metrics import roc_curve, auc
false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(false_positive_rate, true_positive_rate)


# In[49]:



import matplotlib.pyplot as plt
plt.figure(figsize=(10,10))
plt.title('Receiver Operating Characteristic')
plt.plot(false_positive_rate,true_positive_rate, color='red',label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],linestyle='--')
plt.axis('tight')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')



#  The default decision tree model is giving the best accuracy score 

# Tuning the hyperparameters of the Decision tree model

# There are a few criterions to consider here:
# 
# 1 Decision trees use multiple algorithms to decide while splitting a node in two or more sub-nodes.
# Decision tree splits the nodes on all available variables and then it selects the split which results in most homogeneous sub-nodes. 
# 
# 2 max_depth i.e. the Maximum depth of tree, which is the vertical depth: Is used to control over-fitting as higher depth will allow model to learn relations very specific to a particular sample.
# 
# 3 max_features and min_samples_leaf is same as Random Forest classifier
# 

# In[51]:



from sklearn.tree import DecisionTreeClassifier

model_DD = DecisionTreeClassifier()


tuned_parameters= {'criterion': ['gini','entropy'], 'max_features': ["auto","sqrt","log2"],
                   'min_samples_leaf': range(1,100,1) , 'max_depth': range(1,50,1)
                  }
           



# In[52]:



from sklearn.grid_search import RandomizedSearchCV
DD_model= RandomizedSearchCV(model_DD, tuned_parameters,cv=10,scoring='accuracy',n_iter=20,n_jobs= -1,random_state=5)



# In[53]:



DD_model.fit(X_train, y_train)



# In[54]:

print(DD_model.grid_scores_)


# In[55]:



print(DD_model.best_score_)



# In[56]:

print(DD_model.best_params_)


# In[57]:

y_prob = DD_model.predict_proba(X_test)[:,1] # This will give the positive class prediction probabilities  
y_pred = np.where(y_prob > 0.5, 1, 0) # This will threshold the probabilities to give the class predictions.
DD_model.score(X_test, y_pred)


# In[58]:

confusion_matrix=metrics.confusion_matrix(y_test,y_pred)
confusion_matrix


# In[59]:



auc_roc=metrics.classification_report(y_test,y_pred)
auc_roc



# In[60]:

auc_roc=metrics.roc_auc_score(y_test,y_pred)
auc_roc


# In[61]:



from sklearn.metrics import roc_curve, auc
false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(false_positive_rate, true_positive_rate)
roc_auc



# In[62]:



import matplotlib.pyplot as plt
plt.figure(figsize=(10,10))
plt.title('Receiver Operating Characteristic')
plt.plot(false_positive_rate,true_positive_rate, color='red',label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],linestyle='--')
plt.axis('tight')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')



# 
# 
# Conclusion - all the models applied above were able to succesfully predict the correct class with 99% accuracy
# 
