# -*- coding: utf-8 -*-
"""
Created on Fri Mar 25 15:01:56 2022

Tickluck

All rights reserved
"""

# Import Libraries 
import pandas as pd
import seaborn as sns
import numpy as np
from datetime import datetime

import matplotlib
import matplotlib.pyplot as plt
plt.style.use('ggplot')
from matplotlib.pyplot import figure

%matplotlib inline
matplotlib.rcParams['figure.figsize'] = (12,8) # adjustment of our Plots Configs 

# Read in the Data

data = pd.read_csv('movies.csv')

data.head()

# Lets see if there is any missing data
for col in data.columns:
    pct_missing = np.mean(data[col].isnull())
    print('{} - {}%'.format(col,pct_missing))
    
# Look at Data types
data.dtypes
    
# Change data types of Columns 
data['budget']  = data['budget'].astype('int64')
data['gross']  = data['gross'].astype('int64')

# Note 'year' column is NOT always matching with "released" year
# Thus we can construct new column with the right data from "released"
# regex ^(?:\w+ ){2}\K\w+ 
# data['year_matching'] = \
#   data['released'].astype(str).str.extract(r'^(?:\w+ ){2}\K\w+', expand=True)

# get 4 chars from 3d space 
# ---------------------

data.sort_values(by = ['gross'], inplace = False, ascending = False)

# setting option to see ALL of the rows 
# is applied to the rest of the project
pd.set_option('display.max_rows', None)

data.company.drop_duplicates().sort_values(ascending = False) 

# Let us find Highly Correlated fields to GROSS 
# Initial Hypothesis 
# budget | company

# (1) Checking Budget

# Scatter plot -> budget vs gross
plt.scatter(x = data.budget, y = data.gross)
plt.title('Budget vs Gross Earnings')
plt.xlabel('Budget for Film')
plt.ylabel('Gross Earnings')

# Regression Plot Budget vs Gross using seaborn
sns.regplot(x = 'budget', y = 'gross', data = data, \
            scatter_kws={"color": "red"}, line_kws={"color": "blue"})

# Quick Look at correlation
data.corr(method = 'pearson') # Corr Methods: pearson | kendall | spearman 

# Result:
# High Correlation between Budget and Gross (~ 74%)

correlation_matrix = data.corr(method = 'pearson')
sns.heatmap(correlation_matrix, annot = True)

plt.title('Correlation Metrics for Numeric Features')
plt.xlabel('Movie Features')
plt.ylabel('Movie Features')

# (2) Checking Company

# Working on Non-numeric data
# We need to convert it to numeric to be able using Correlation Matrix
data_numerized = data
for col_name in data_numerized.columns:
    if(data_numerized[col_name].dtype == 'object'):
         data_numerized[col_name] = data_numerized[col_name].astype('category')
         data_numerized[col_name] = data_numerized[col_name].cat.codes
      
    
correlation_matrix = data_numerized.corr(method = 'pearson')
sns.heatmap(correlation_matrix, annot = True)

plt.title('Correlation Metrics for Numeric Features')
plt.xlabel('Movie Features')
plt.ylabel('Movie Features')

corr_pairs = correlation_matrix.unstack()

sorted_corr_pairs = corr_pairs.sort_values()

high_correlation = sorted_corr_pairs[(sorted_corr_pairs) > 0.5]

# Result:
# Correlation Between Gross and Company is really small (<20%)
# But Analysis shows that Votes has high correlation (~63%)















