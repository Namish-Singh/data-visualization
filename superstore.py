# -*- coding: utf-8 -*-
"""
Created on Wed Nov 14 14:37:54 2018

@author: Namish Kaushik
"""

import pandas as pd
import numpy as np
oda = pd.ExcelFile('SS.xls')
odf = oda.parse(1) 
data = pd.ExcelFile('sse.xlsx')
df= data.parse(0)
pd.set_option('max_columns',10)
pd.set_option('max_rows',200)
df['CP'] = df['Sales']- df['Profit']

df.head()
df[df['Ship Mode'].str.startswith('S')]
df.shape
df['Ship Mode'].str.contains('Standard')


def removedigit(string):
    return ''.join(ch for ch in string if not ch.isdigit())
df['an']= df['Order ID'].apply(removedigit)
df['an'].head()
#

df['ab']= df['Order ID'].str.replace('[0-9]','')

df['ab'].head()


#Remove everything after a character in a columns string entries


df = fresh_df.copy()
def removeAfterComma(string):
    """
    input is a string 
    output is a string with everything after comma removed
    """
    return string.split(',')[0].strip()
df.Name = df.Name.apply(removeAfterComma)
df.head()
#or
j= "bio,kop,plo nj"

df = fresh_df.copy()

df['a'] = df['Order ID'].str.split('-').str[0].str.strip()
df.a.head()


df['Order ID']=df['Order ID'].astype(str)
.str.capitalize()
df.Quantity= df.Quantity.astype('float64')
fd= df.loc[:,df.dtypes== 'float64']
fdd =fd.sample(n=100)
fdd.sub(fdd.iloc[:,5],axis ='index')
#fd =fd.set_index(['Row ID'])
fd.index.astype('Int64')
fd['CP'].isnull().value_counts()
def nullcount(x):
    return x.isnull().value_counts()
fd.apply(nullcount)
fd['pro2'] = fd['Sales']- fd['CP']
fd.index.name = 'Serial no.'

fd['area_code'] = fd['Postal Code'] == 33311
fd.values
del fd['pro2']
fd.columns.name = 'attributes'
fd.apply(max)
fd.iloc[fd['CP'].idxmax()]
def arg(x):
    return df.iloc[x.idxmax()] # still not done
df['Postal Code'].idxmax()
fd.apply(arg)
fd.iloc[fd['CP'].sort_values().index] # important
fcp=fd.sort_values(by= 'CP',ascending = False)
fd =fd.merge(df[['Product ID']],left_index= True,right_index = True)
fd.head()
max_sal_prod =odf['Sales'].groupby(odf['Product ID'],group_keys =False).sum().sort_values(ascending = False)[:1]
#max_sal_prod very imp
max_quan_prod =odf['Quantity'].groupby(odf['Product ID'],group_keys =False).sum().sort_values(ascending = False)[:1]
#cross check
odf[odf['Product ID']== 'TEC-CO-10004722']['Quantity'].sum()
odf.columns
# region wise sales
odf['Sales'].groupby(odf['Region']).sum().sort_values(ascending = False)
# region wise highest revenue product
pd=odf.groupby(['Region','Product ID'],sort= True,as_index = True)['Sales'].agg([max,sum])
m1=pd.crosstab(odf['Product ID'],odf.Region,values = odf.Sales,aggfunc = sum).idxmax()
m2=pd.crosstab(odf['Product ID'],odf.Region,values = odf.Sales,aggfunc = sum).max()
odf.columns = odf.columns.str.strip()
m2.name = 'Sales'
m1.name = 'Product'
pd.concat([m1,m2],axis=1,names =['Product','sales'])
tup = ({'Sales' :'sum' , 'Quantity': 'max'})

# applying different aggregate function to different columns
odf2 =odf.pivot_table(['Sales','Quantity'],columns = 'Region',
                      index = 'Product ID', aggfunc = tup,fill_value= 0)
odf1 =odf.pivot_table(['Sales','Quantity'],columns = 'Region',
                      index = 'Product ID', aggfunc = sum,fill_value= 0)


# return to 5 products from each reason accoridng to sales
group= odf.groupby(['Region','Product ID'],as_index= False,sort = True)['Sales'].sum()#.sort_values(by = 'Sales',ascending = False)
def top(df, n=5, column = 'col'):   
    return df.sort_values(by =column,ascending = False)[:n]
    #return df.column.sum.sort_values(by= column,ascending= False)[:5]

group.groupby(['Region'],as_index = False).apply(top,n=6,column='Sales')

# applying different function to different columns
odf.groupby(['Region','Product ID']).agg(tup).unstack(level=0,fill_value =0)[:5]


# region wise ,subcategory wise top 5 profitable sub category

g1=odf.groupby(['Region','Sub-Category'],as_index= False)[['Profit','Sales']].sum()

def top(df, n=5, column = 'col'):   
    return df.sort_values(by =column,ascending = False)[:n]
g1.groupby(['Region'],as_index=False).apply(top,n=6, column = 'Profit')

# count na values in each column of odf
odf.apply(nullcount)

od= odf.copy()

of = od[['City','Sales','Quantity']].stack().sample(frac=0.8).unstack()
na_df=of.merge(od,left_index = True,right_index = True)

nad = na_df.loc[:,[ 'Row ID', 'Order ID', 'Order Date',
       'Ship Date', 'Ship Mode', 'Customer ID', 'Customer Name', 'Segment',
       'Country', 'City_x', 'State', 'Postal Code', 'Region', 'Product ID',
       'Category', 'Sub-Category', 'Product Name', 'Sales_x', 'Quantity_x',
       'Discount', 'Profit']]


nad[nad['City_x'].isnull()][['Country','Postal Code']]

nad['City_x'].isnull().value_counts()
nad['City_x'].dropna(inplace = True)

nad['Country'].value_counts()

df2= nad.drop_duplicates(['Country','Postal Code'])[['Country','Postal Code','City_x']]

df3=nad.merge(df2, left_on = ['Country','Postal Code'],right_on = ['Country','Postal Code'])

df3['City_x_x'] == df3['City_x_y']
del df3['City_x_x']

# top 5 profit making states
odf.groupby(['State'])['Profit'].sum().sort_values(ascending = False)[:5]
# top 5 loss making states
odf.groupby(['State'])['Profit'].sum().sort_values(ascending = False)[-5:]
#categorywise state's profit
odf.pivot_table(values ='Profit',index ='State',columns='Category',aggfunc= 'sum').sort_values(by= 'Furniture',ascending = False)

# region wise top 5 profit making state
g2=odf.groupby(['Region','State'], as_index= False)['Profit'].sum()
g2.groupby(['Region'],group_keys = False).apply(top,column= 'Profit',n=6)

# add a gender column
od['Gender'] = np.random.randint(0,2,size= 9994)
#replace 1 with male, 0 with female
od['Gender'].replace({1: 'Male',0:'Female'},inplace = True) 

# now we will go for binning and discrization
pro_cust=odf.groupby(['Customer ID'],as_index = False)['Profit'].sum().sort_values(by= 'Profit',ascending = False)
type(pro_cust)
# into 4  parts
#pro_cust['bin']=
pd.cut(pro_cust['Profit'],bins =4,labels=[1,2,3,4],retbins= True)
pro_cust['bin']=pd.cut(pro_cust['Profit'],bins =4,labels=[1,2,3,4])

pro_cust['bin'].value_counts() # no of customers in eaxh 4 bin
odf_cust=pd.merge(odf,pro_cust,left_on = 'Customer ID',right_on ='Customer ID')
del odf_cust['Profit_y']
odf_cust.groupby(['bin']).size()
odf_cust['bin'].value_counts()

# now we will se quantile cut
pd.qcut(pro_cust['Profit'],q=4,labels = [1,2,3,4],retbins = True)
pro_cust['q_bin']=pd.qcut(pro_cust['Profit'],q=4,labels = [1,2,3,4])
pro_cust['q_bin'].value_counts()
pro_cust['bin'].value_counts()

# customers with thier cities for whom the profit > 560.0078
q_cust= odf.groupby(['Customer ID'],as_index = False)['Profit','City'].agg(sum)#]
iv_qcust = q_cust[q_cust['Profit']>560.0078]
# checking the result
pro_cust[pro_cust['q_bin']==4].sort_index(ascending = False,axis =0)
# distributing the customers in to un even quazrtile
mpro_cust=odf.groupby(['Customer ID'],as_index = False)['Profit'].sum().sort_values(by= 'Profit',ascending = False)
mpro_cust['qbin']=pd.qcut(pro_cust['Profit'],q=[0,0.10,0.90,1],labels = [1,2,3])
pd.qcut(mpro_cust['Profit'],q=[0,0.10,0.90,1],labels = [1,2,3],retbins = True)
q_cust= odf.groupby(['Customer ID'],as_index = False)['Profit','City'].agg(sum)
iv_qcust = q_cust[q_cust['Profit']>1158.05174]
