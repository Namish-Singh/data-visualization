# -*- coding: utf-8 -*-
"""
Created on Thu Dec  6 10:07:58 2018

@author: Namish Kaushik
"""

import pandas as pd
import numpy as np
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.cross_validation import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier 
import seaborn as sns
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB
from scipy.stats import ttest_1samp, ttest_ind, mannwhitneyu, levene, shapiro,wilcoxon,bartlett
from statsmodels.stats.power import ttest_power
df = pd.read_csv("Dataset_spine.csv")
df.head()
df.shape
df.columns
df.drop(columns = ['Unnamed: 13', 'Unnamed: 14', 'Unnamed: 15'],axis =1, inplace = True)
df.shape
# COUNTING THE NULL VALUES

def null_count(df):
    total_miss = df.isnull().sum()
    perc_miss = (df.isnull().sum())*100/len(df)
    mis_val_table=pd.concat([total_miss,perc_miss],axis =1)
    #renaming cols
    mis_val_table_ren_columns = mis_val_table.rename(
        columns = {0 : 'Missing Values', 1 : '% of Total Values'})
    mis_val_table_ren_columns = mis_val_table_ren_columns[ mis_val_table_ren_columns.iloc[:,1] != 0].sort_values(
        '% of Total Values', ascending=False).round(1)
    print ("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"      
            "There are " + str(mis_val_table_ren_columns.shape[0]) +
              " columns that have missing values.")
    return mis_val_table_ren_columns
miss_val_res = null_count(df)    
# NO NULL VALUES ARE PRESENT

# scaling the input so that everyting gets normalized bacause  some inputs are numerically
# high and some are very low . Hence we need to scale it
X= df.iloc[:,:12]
scaler = StandardScaler()
scaler.fit(X)
x= scaler.fit_transform(X)

# since our dependent variable is string hence we need to encde it throurgh label encoder

Y= df.iloc[:,-1]
le= LabelEncoder()
y=le.fit_transform(Y)
y1= pd.Series(y,name = 'dependent_variable')
x1= pd.DataFrame(x, columns =['pelvic_incidence', 'pelvic_tilt', 'lumbar_lordosis_angle',
       'sacral_slope', 'pelvic_radius', 'degree_spondylolisthesis',
       'pelvic_slope(numeric)', 'Direct_tilt(numeric)',
       'thoracic_slope(numeric)', 'cervical_tilt(numeric)',
       'sacrum_angle(numeric)', 'scoliosis_slope(numeric)'] )
df2 = pd.concat([x1,y1],axis =1)
df2['dependent_variable'].replace({0:1, 1:0},inplace = True)
df2.describe
gr = df2.groupby(['dependent_variable'])
gr1= gr.get_group(0)
gr2= gr.get_group(1)
shapiro(gr1['pelvic_incidence'])# not normal
shapiro(gr2['pelvic_incidence'])# not normal
shapiro(gr1['pelvic_tilt'])# normal
shapiro(gr2['pelvic_tilt'])# not normal
shapiro(gr1['lumbar_lordosis_angle'])# not normal
shapiro(gr2['lumbar_lordosis_angle'])# not normal
shapiro(gr1['sacral_slope'])#  normal
shapiro(gr2['sacral_slope'])# not normal
shapiro(gr1['pelvic_radius'])#  normal
shapiro(gr2['pelvic_radius'])# not normal
shapiro(gr1['degree_spondylolisthesis'])# not normal
shapiro(gr2['degree_spondylolisthesis'])# not normal
shapiro(gr1['pelvic_slope(numeric)'])# not normal
shapiro(gr2['pelvic_slope(numeric)'])# not normal
shapiro(gr1['Direct_tilt(numeric)'])# not normal
shapiro(gr2['Direct_tilt(numeric)'])# not normal
shapiro(gr1['cervical_tilt(numeric)'])# not normal
shapiro(gr2['cervical_tilt(numeric)'])# not normal
shapiro(gr1['thoracic_slope(numeric)'])# not normal
shapiro(gr2['thoracic_slope(numeric)'])# not normal
shapiro(gr1['sacrum_angle(numeric)'])# not normal
shapiro(gr2['sacrum_angle(numeric)'])# not normal
shapiro(gr1['scoliosis_slope(numeric)'])# not normal
shapiro(gr2['scoliosis_slope(numeric)'])# not normal
# since approximately in  all the cases disribution is not normal so we wiil apply manhhwhitneyu
mannwhitneyu(gr1['pelvic_incidence'],gr2['pelvic_incidence'])# pval< 0.05
mannwhitneyu(gr1['pelvic_tilt'],gr2['pelvic_tilt'])# pval< 0.05
mannwhitneyu(gr1['lumbar_lordosis_angle'],gr2['lumbar_lordosis_angle'])# pval< 0.05
mannwhitneyu(gr1['sacral_slope'],gr2['sacral_slope'])# pval< 0.05
mannwhitneyu(gr1['pelvic_radius'],gr2['pelvic_radius'])# pval< 0.05
mannwhitneyu(gr1['degree_spondylolisthesis'],gr2['degree_spondylolisthesis'])# pval< 0.05
mannwhitneyu(gr1['pelvic_slope(numeric)'],gr2['pelvic_slope(numeric)'])# pval> 0.05
mannwhitneyu(gr1['Direct_tilt(numeric)'],gr2['Direct_tilt(numeric)'])# pval> 0.05
mannwhitneyu(gr1['cervical_tilt(numeric)'],gr2['cervical_tilt(numeric)'])# pval< 0.05
mannwhitneyu(gr1['sacrum_angle(numeric)'],gr2['sacrum_angle(numeric)'])# pval> 0.05
mannwhitneyu(gr1['scoliosis_slope(numeric)'],gr2['scoliosis_slope(numeric)'])# pval> 0.05

x_f = df2[['pelvic_incidence','pelvic_tilt','lumbar_lordosis_angle','sacral_slope',
           'pelvic_radius','degree_spondylolisthesis','cervical_tilt(numeric)','scoliosis_slope(numeric)']]
y_f= df2['dependent_variable']

x_train,x_test,y_train,y_test = train_test_split(x_f,y_f,test_size = 0.3,random_state=1)
depth = []
depth1= []
for i in range(3,20):
    clf = tree.DecisionTreeClassifier(max_depth = i)
    scores = cross_val_score(estimator = clf, X = x_train,y=y_train , cv =5)
    depth1.append(scores.mean())
    depth.append((i,scores.mean()))
print(depth)
# we get the maximum accuracy at depth =6
# fitting the model at depth =6
clf= tree.DecisionTreeClassifier(max_depth = 6)
clf.fit(x_train,y_train)
y_pred= clf.predict(x_test)
ct_d=pd.crosstab(y_test,y_pred)
print((ct_d.iloc[0,0]+ct_d.iloc[1,1])/(ct_d.iloc[0,0]+ct_d.iloc[0,1]+ct_d.iloc[1,0]+ct_d.iloc[1,1]))
ct_d
# now applying the logistic regression

m1= LogisticRegression()
scores = cross_val_score(estimator = m1, X = x_train,y=y_train , cv =5)
m1.fit(x_train,y_train)
log_acc = scores.mean()
y_pred1= m1.predict(x_test)
ct_l=pd.crosstab(y_test,y_pred1)
print((ct_l.iloc[0,0]+ct_l.iloc[1,1])/(ct_l.iloc[0,0]+ct_l.iloc[0,1]+ct_l.iloc[1,0]+ct_l.iloc[1,1]))

# training accuracy is 82% whule test accuracy is 86 %
sensitivity = 59/66# (tpr/(tpr+fn))
# sensitivity is  approximately 90% which is high and good 

# now we will be do the knn
temp1 = []
for i in range(1,50):
    classifier = KNeighborsClassifier(n_neighbors=i)  
    scores = cross_val_score(estimator = clf, X = x_train,y=y_train , cv =5)
    temp1.append((i,scores.mean()))# we will take n_neighbours as 7 as it is giving highest scores when doing cv
# building the model with      n_neighbours as 7
classifier = KNeighborsClassifier(n_neighbors=7)
classifier.fit(x_train,y_train)
y_pred2=classifier.predict(x_test)
c_k =pd.crosstab(y_test, y_pred2)   
print((c_k.iloc[0,0]+c_k.iloc[1,1])/(c_k.iloc[0,0]+c_k.iloc[0,1]+c_k.iloc[1,0]+c_k.iloc[1,1]))    

# accuracy is 77.41%
sensitivity_k =54/66 # (tpr/(tpr+fn))
# sensitivity is  approximately 81 % which is realtively moderate 

# now we will be do the naive bayes
clf = GaussianNB()
scores = cross_val_score(estimator = clf, X = x_train,y=y_train , cv =5)
# accuracy is 76%
clf.fit(x_train,y_train)
pred= clf.predict(x_test)

c_n =pd.crosstab(y_test, y_pred)  
print((c_n.iloc[0,0]+c_n.iloc[1,1])/(c_n.iloc[0,0]+c_n.iloc[0,1]+c_n.iloc[1,0]+c_n.iloc[1,1]))    
# accuracy is 80%
sensitivity_n= 55/66 # # (tpr/(tpr+fn))
# sensitivity is  approximately 84 % which is realtively moderate 