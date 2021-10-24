#!/usr/bin/env python
# coding: utf-8

# #### Code and analysis by Gracieli Scremin
# 
# 
# # Visit with Us Project
# 
# ## Description
# 
# ### Background and Context
# 
# You are a Data Scientist for a tourism company named "Visit with us". The Policy Maker of the company wants to enable and establish a viable business model to expand the customer base.
# 
# A viable business model is a central concept that helps you to understand the existing ways of doing the business and how to change the ways for the benefit of the tourism sector.
# 
# One of the ways to expand the customer base is to introduce a new offering of packages.
# 
# Currently, there are 5 types of packages the company is offering - Basic, Standard, Deluxe, Super Deluxe, King. Looking at the data of the last year, we observed that 18% of the customers purchased the packages.
# 
# The company in the last campaign contacted the customers at random without looking at the available information. However, this time company is now planning to launch a new product i.e. Wellness Tourism Package. Wellness Tourism is defined as Travel that allows the traveler to maintain, enhance or kick-start a healthy lifestyle, and support or increase one's sense of well-being, and wants to harness the available data of existing and potential customers to make the marketing expenditure more efficient.
# 
# You as a Data Scientist at "Visit with us" travel company has to analyze the customers' data and information to provide recommendations to the Policy Maker and Marketing Team and also build a model to predict the potential customer who is going to purchase the newly introduced travel package.
# 
# ### Objective
# 
# To predict which customer is more likely to purchase the newly introduced travel package.
# 
# ##### Data Dictionary
# 
# Customer details:
# 
# * CustomerID: Unique customer ID
# * ProdTaken**: Whether the customer has purchased a package or not (0: No, 1: Yes)
# * Age: Age of customer
# * TypeofContact: How customer was contacted (Company Invited or Self Inquiry)
# * CityTier: City tier depends on the development of a city, population, facilities, and living standards. The categories are ordered i.e. Tier 1 > Tier 2 > Tier 3
# * Occupation: Occupation of customer
# * Gender: Gender of customer
# * NumberOfPersonVisiting: Total number of persons planning to take the trip with the customer
# * PreferredPropertyStar: Preferred hotel property rating by customer
# * MaritalStatus: Marital status of customer
# * NumberOfTrips: Average number of trips in a year by customer
# * Passport: The customer has a passport or not (0: No, 1: Yes)
# * OwnCar: Whether the customers own a car or not (0: No, 1: Yes)
# * NumberOfChildrenVisiting: Total number of children with age less than 5 planning to take the trip with the customer
# * Designation: Designation of the customer in the current organization
# * MonthlyIncome: Gross monthly income of the customer
# 
# Customer interaction data: 
# 
# * PitchSatisfactionScore: Sales pitch satisfaction score
# * ProductPitched: Product pitched by the salesperson
# * NumberOfFollowups: Total number of follow-ups has been done by the salesperson after the sales pitch
# * DurationOfPitch: Duration of the pitch by a salesperson to the customer
# 
# 
# 
# 
# ** ProdTaken is our target/dependent variable

# ### Loading Libraries

# In[2]:


# Library to suppress warnings or deprecation notes 
import warnings
warnings.filterwarnings('ignore')

# Libraries to help with reading and manipulating data
import numpy as np
import pandas as pd

# Libraries to help with data visualization
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns

# Libraries to split data, impute missing values 
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

# Libraries to import decision tree classifier and different ensemble classifiers
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.tree import DecisionTreeClassifier

# Libtune to tune model, get different metric scores
from sklearn import metrics
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import GridSearchCV


# ### Loading the data

# In[3]:


data_dictionary = pd.read_excel("Tourism.xlsx", sheet_name='Data Dict', header=1, usecols = "C:D")


# In[4]:


data_dictionary


# In[5]:


# loading the dataset
tourism_data = pd.read_excel("Tourism.xlsx", 'Tourism')


# In[6]:


tourism_data.head()


# In[7]:


tourism_data.shape


# In[8]:


df = tourism_data.copy()


# In[9]:


df.head()


# In[10]:


df.tail()


# In[11]:


df.shape


# In[12]:


df.info()


# #### Observation

# * Many columns are of type object i.e. strings. These need to be converted to ordinal type
# * Other variables, such as Age and NumberofChildrenVisiting, which we would expect to be integers are classified as float data type

# In[13]:


df.nunique()


# #### Observation
# 
# * We can drop 'CustomerID' column as it is an ID variable and will not add value to the model.

# In[14]:


#Dropping two columns from the dataframe
df.drop(columns=['CustomerID'], inplace=True)


# In[15]:


df.describe(include='all').T


# #### Observation
# 
# * ProdTaken (target/dependent variable): distribution correctly reflecting only 2 possible values: 0 for people who did not buy travel package and 1 for those who bought
# * Age: values falling on a range we would expect - mean age: 37.62, median: 36; 50% of people surveyed falling between 31 and 44 years of age
# * TypeofContact: 2 unique values, most common being "Self Enquiry"
# * CityTier: mean of range: 1.65 - city tier possibilities falling in three categories: 1, 2, or 3 (Tier 1 > Tier 2 > Tier 3)
# * DurationOfPitch: we observed a wide range, from a min of 5 to a maximum of 127 (assuming unit of measurement to be minutes). We also observed a right skew to the distribution probably due to presence of outliers
# * Occupation: column values fall in 4 different categories, the most common being "Salaried" with a count of 2368
# * Gender: 3 unique values - standard would be 2, we will look at values further
# * NumberOfPersonVisiting: min value of 1 and max of 5, mean of 2.9 falling very close to median of 3
# * NumberOfFollowUps: min value of 1 and max of 6. Mean of 3.7 and median of 4. 50% of values falling between 3 and 4 follow ups
# * ProductPitched: 5 unique values, most common being "Basic" with a count of 1842
# * PreferredPropertyStar: min value of 3 and max, as expected, of 5.
# * MaritalStatus: 4 unique values, most common being "Married" with a count of 2340
# * NumberOfTrips: wide range observed here - from 1 to 22. Closely alligned mean and median, indicative normal distribution
# * Passport: values distributed as expected between 0 and 1. Mean of 0.3 and median of 0 indicate that most people in the dataset do not own a passport
# * PitchSatisfactionScore: values it looks like falling on a scale from 1 to 5. 50% falling between 2 and 4
# * OwnCar: values distributed as expected between 0 and 1. Mean of 0.62 and median of 1 indicate that most people in the dataset own a car
# * NumberOfChildrenVisiting: values ranging from 1 to 3. Mean and median falling close to 1
# * Designation: 5 unique values, the most common being "Executive" with a count of 1842
# * MonthlyIncome: a wide range of values here, from 1000 to 98678. Mean and median not falling too far from each other: mean income 23,619.90 and median 22,347. Missing values observed which will need to be resolved. We will further examine distribution as well to treat variable for outliers as needed

# ### Looking more closely at Gender feature

# In[16]:


df["Gender"].value_counts()


# #### Observation
# 
# * It looks like 155 values have been mislabeled as "Fe Male". Correction needed here

# #### Gender feature: data clean-up

# In[17]:


# Extracting space from "Fe Male" values
def remove_whitespace(x):
    """
    Helper function to remove any blank space from a string
    x: a string
    """
    try:
        # Remove spaces inside of the string
        x = "".join(x.split())

    except:
        pass
    return x

df.Gender = df.Gender.apply(remove_whitespace)


# In[18]:


# Formatting all values in column as title to handle upper case "M" in FeMale
df.Gender = df.Gender.str.title()


# In[19]:


df.Gender.value_counts()


# #### Observation
#     
# * Data clean up completed successfully

# ## Handling missing values

# #### Getting percentage of missing values per column

# In[20]:


((df.isnull().sum().sort_values(ascending=False))/len(df))*100


# #### Observation
# 
# * Null values observed in several columns - these will need to be addressed
# * Except for missing values in DurationOfPitch which are slightly over 5%, missing values do not surpass 5% of the total number of values in each column
# * Most of the missing values are observed in numeric features - we will look at the value distributions for each column carefully to determine the best course of action for missing value imputation

# #### Columns with mising values:
# 
# * DurationOfPitch            
# * MonthlyIncome              
# * Age                        
# * NumberOfTrips               
# * NumberOfChildrenVisiting  
# * NumberOfFollowups          
# * PreferredPropertyStar      
# * TypeofContact 
# 
# 
# #### Let's examine value distributions more closely of the missing value features listed above through histograms and barplots

# In[21]:


# Function below helps to create a boxplot and histogram for any input numerical 
# variable. It takes the numerical column as the input and returns the boxplots 
# and histograms for the variable.

def histogram_boxplot(feature, figsize=(15,10), bins = None):
    """ Boxplot and histogram combined
    feature: 1-d feature array
    figsize: size of fig (default (9,8))
    bins: number of bins (default None / auto)
    """
    f2, (ax_box2, ax_hist2) = plt.subplots(nrows = 2, # Number of rows of the subplot grid= 2
                                           sharex = True, # x-axis will be shared among all subplots
                                           gridspec_kw = {"height_ratios": (.25, .75)}, 
                                           figsize = figsize 
                                           ) # creating the 2 subplots
    sns.boxplot(feature, ax=ax_box2, showmeans=True, color='green') # boxplot will be created and a star will indicate the mean value of the column
    sns.distplot(feature, kde=F, ax=ax_hist2, bins=bins) if bins else sns.distplot(feature, kde=False, ax=ax_hist2) # For histogram
    ax_hist2.axvline(np.mean(feature), color='g', linestyle='--') # Add mean to the histogram
    ax_hist2.axvline(np.median(feature), color='black', linestyle='-') # Add median to the histogram


# #### Treating missing values in continuous numeric features

# ##### DurationOfPitch

# In[22]:


df["DurationOfPitch"].describe()


# In[23]:


histogram_boxplot(df['DurationOfPitch'])


# #### Observation
# 
# * We observe here a positive/right-skewed distribution with mean greater than median value. This indicates missing values should be filled with median rather than mean

# ##### MonthlyIncome

# In[24]:


df["MonthlyIncome"].describe()


# In[25]:


histogram_boxplot(df['MonthlyIncome'])


# #### Observation
# 
# * We observe here a positive/right-skewed distribution with mean greater than median value which indicates missing values should be filled with median rather than mean

# ##### Age

# In[26]:


df["Age"].describe()


# In[27]:


histogram_boxplot(df['Age'])


# #### Observation
# 
# * We observe a slight positive/right-skewed distribution with mean greater than median value which indicates missing values should be filled with median rather than mean

# In[28]:


#using `fillna` with continuous numeric columns DurationOfPitch, MonthlyIncome, Age
median_fillna = ['DurationOfPitch','MonthlyIncome', 'Age']
for colname in median_fillna:
    print(df[colname].isnull().sum())
    df[colname].fillna(df[colname].median(), inplace=True) 
    print(df[colname].isnull().sum())


# #### Observation
# 
# * Filling missing values for DurationOfPitch, MonthlyIncome, Age successfully handled

# In[29]:


def labeled_barplot(data, feature, perc=False, n=None):
    """
    Barplot with percentage at the top

    data: dataframe
    feature: dataframe column
    perc: whether to display percentages instead of count (default is False)
    n: displays the top n category levels (default is None, i.e., display all levels)
    """

    total = len(data[feature])  # length of the column
    count = data[feature].nunique()
    if n is None:
        plt.figure(figsize=(count + 2, 6))
    else:
        plt.figure(figsize=(n + 2, 6))

    plt.xticks(rotation=90, fontsize=15)
    ax = sns.countplot(
        data=data,
        x=feature,
        palette="Paired",
        order=data[feature].value_counts().index[:n].sort_values(),
    )

    for p in ax.patches:
        if perc == True:
            label = "{:.1f}%".format(
                100 * p.get_height() / total
            )  # percentage of each class of the category
        else:
            label = p.get_height()  # count of each level of the category

        x = p.get_x() + p.get_width() / 2  # width of the plot
        y = p.get_height()  # height of the plot

        ax.annotate(
            label,
            (x, y),
            ha="center",
            va="center",
            size=12,
            xytext=(0, 5),
            textcoords="offset points",
        )  # annotate the percentage

    plt.show()  # show the plot


# #### Treating missing values in ordinal numeric features

# ##### NumberOfChildrenVisiting

# In[30]:


df["NumberOfChildrenVisiting"].describe()


# In[31]:


labeled_barplot(df, "NumberOfChildrenVisiting", perc=True) 


# #### Observation
# 
# * Mean and median closely alligned at 1 child planned to accompany in visit

# ##### NumberOfFollowups

# In[32]:


df["NumberOfFollowups"].describe()


# In[33]:


labeled_barplot(df, "NumberOfFollowups", perc=True) 


# #### Observation
# 
# * Mean and median closely alligned at 4 follow ups

# ##### NumberOfTrips

# In[34]:


df["NumberOfTrips"].describe()


# In[35]:


labeled_barplot(df, "NumberOfTrips", perc=True)


# #### Observation
# 
# * Mean and median closely alligned at 3 trips

# ##### PreferredPropertyStar

# In[36]:


df["PreferredPropertyStar"].describe()


# In[37]:


labeled_barplot(df, "PreferredPropertyStar", perc=True) 


# #### Observation
# 
# * The number of stars most often preferred is 3

# #### Overall observation:
# 
# * Mean will be used to fill missing values for ordinal numeric features since from the examined frequency distributions above mean and median are closely alligned

# In[38]:


#using `fillna` with ordinal numeric columns NumberOfTrips, NumberOfChildrenVisiting, NumberOfFollowups, and PreferredPropertyStar
mean_fillna = ['NumberOfTrips', 'NumberOfChildrenVisiting', 'NumberOfFollowups', 'PreferredPropertyStar']
for colname in mean_fillna:
    print(df[colname].isnull().sum())
    df[colname].fillna(df[colname].mean(), inplace=True) 
    print(df[colname].isnull().sum())


# #### Observation
# 
# * Filling missing values for NumberOfTrips, NumberOfChildrenVisiting, NumberOfFollowups, and PreferredPropertyStar successfully handled

# ##### TypeofContact

# In[39]:


df["TypeofContact"].describe()


# In[40]:


labeled_barplot(df, "TypeofContact", perc=True)


# #### Observation
# 
# * The most frequent value for TypeofContact column is "Self Enquiry" - we will use value to fill in missing values for this feature

# In[41]:


#fillna for categorical feature "TypeofContact" with most frequent category "Self Enquiry"
print(df["TypeofContact"].isnull().sum())
df["TypeofContact"].fillna("Self Enquiry", inplace = True)
print(df["TypeofContact"].isnull().sum())


# In[42]:


((df.isnull().sum().sort_values(ascending=False))/len(df))*100


# #### Observation
# 
# * Filling missing values successfully handled

# ### Converting columns with an 'object' datatype into categorical variables

# In[43]:


for feature in df.columns: # Loop through all columns in the dataframe
    if df[feature].dtype == 'object': # Only apply for columns with categorical strings
        df[feature] = pd.Categorical(df[feature])# Replace strings with an integer
df.info()


# #### Observation
# 
# * Object data types converted to category to optimize processing efficiency

# ### Converting non-float numeric columns to integer values
# 
# * Whole numbers expected from features below

# In[44]:


df["Age"] = df["Age"].astype(int)
df["NumberOfPersonVisiting"] = df["NumberOfPersonVisiting"].astype(int)
df["NumberOfFollowups"]=df["NumberOfFollowups"].astype(int)
df["NumberOfTrips"]=df["NumberOfTrips"].astype(int)
df["NumberOfChildrenVisiting"]=df["NumberOfChildrenVisiting"].astype(int)
df["PreferredPropertyStar"]=df["PreferredPropertyStar"].astype(int)


# ## Exploratory Data Analysis

# #### Age

# In[45]:


histogram_boxplot(df['Age'])


# ##### Observation
# 
# * The mean and median age of Visit With Us customer hover around late 30s - mean of 37.5 and median of 36

# #### DurationOfPitch

# In[46]:


histogram_boxplot(df['DurationOfPitch'])


# ##### Observation
# 
# * We observe a right-skewness in the data given the presence of large value outliers. Outliers will be handled later on

# #### MonthlyIncome

# In[47]:


histogram_boxplot(df['MonthlyIncome'])


# ##### Observation
# 
# * Mean and median annual income hovering around 23K
# * We observe the presence of outliers on both sides of the value distribution - these will be addressed later on

# #### NumberOfPersonVisiting

# In[48]:


labeled_barplot(df, "NumberOfPersonVisiting", perc=True) 


# ##### Observation
# 
# * The most frequent value observed for number of people intended to be travelling with customer is 3 followed by 2
# * Very few customers said they intended to take trip with just 1 person

# #### NumberOfFollowups

# In[49]:


labeled_barplot(df, "NumberOfFollowups", perc=True) 


# ##### Observations
# 
# * Over 70% of customers received 3 to 4 follow ups from Visit With Us sales team

# #### NumberOfTrips

# In[50]:


labeled_barplot(df, "NumberOfTrips", perc=True) 


# ##### Observation
# 
# * Over 50% of customers reported taking on average 2 to 3 trips per year

# #### NumberOfChildrenVisiting

# In[51]:


labeled_barplot(df, "NumberOfChildrenVisiting", perc=True) 


# ##### Observation
# 
# * 44% of customers reported planning to travel with 1 small child (age less than 5)

# #### CityTier

# In[52]:


labeled_barplot(df, "CityTier", perc=True) 


# ##### Observation
# 
# * Metadata describes "CityTier" as depending on the development of a city, population, facilities, and living standards. The categories are ordered i.e. Tier 1 > Tier 2 > Tier 3
# * The most frequent CityTier reported here as Tier 1

# #### PreferredPropertyStar

# In[53]:


labeled_barplot(df, "PreferredPropertyStar", perc=True) 


# ##### Observation
# 
# * The most popular preferred hotel property rating by customers is 3 stars

# #### PitchSatisfactionScore

# In[54]:


labeled_barplot(df, "PitchSatisfactionScore", perc=True) 


# ##### Observation
# 
# * Most common rating for sales pitch satisfaction was 3
# * Near 70% of customers rated sales pitch from 3 to 5

# #### Gender

# In[55]:


labeled_barplot(df, "Gender", perc=True) 


# ##### Observation
# 
# * Close to 60% of customers are male

# #### MaritalStatus

# In[56]:


labeled_barplot(df, "MaritalStatus", perc=True) 


# ##### Observation
# 
# * Nearly 50% of customers are married

# #### Occupation

# In[57]:


labeled_barplot(df, "Occupation", perc=True) 


# ##### Observation
# 
# * Most customers are salaried professionals
# * 2 customers reported being free lancers

# #### Designation

# In[58]:


labeled_barplot(df, "Designation", perc=True) 


# #### Observation
# 
# * Most customers reported being Executives (nearly 40%) followed by managers (35%)

# #### Passport

# In[59]:


labeled_barplot(df, "Passport", perc=True) 


# ##### Observation
# 
# * 70% of customers report not having a passport

# #### OwnCar

# In[60]:


labeled_barplot(df, "OwnCar", perc=True) 


# ##### Observation
# 
# * 62% of customers report having a car

# #### ProductPitched

# In[61]:


labeled_barplot(df, "ProductPitched", perc=True) 


# ##### Observation
# 
# * The most often pitched travel package was the Basic (38%) followed by the Deluxe (35%)
# * Packages ordered as Basic, Standard, Deluxe, Super Deluxe, King

# #### TypeofContact

# In[62]:


labeled_barplot(df, "TypeofContact", perc=True) 


# ##### Observation
# 
# * 71% of customers reported approaching Visit With Us with travel inquiries rather than being first contacted by company

# #### ProdTaken (Target Variable)

# In[63]:


labeled_barplot(df, "ProdTaken", perc=True) 


# ##### Observation
# 
# * Close to 19% of customers bought a Visit With Us travel package 

# ## Examining Relationships between Features and Target Variable (Bivariate Analysis)

# ##### Setting up functions for visualizations

# In[64]:


### function to plot distributions wrt target


def distribution_plot_wrt_target(data, predictor, target):

    fig, axs = plt.subplots(2, 2, figsize=(12, 10))

    target_uniq = data[target].unique()

    axs[0, 0].set_title("Distribution of target for target=" + str(target_uniq[0]))
    sns.histplot(
        data=data[data[target] == target_uniq[0]],
        x=predictor,
        kde=True,
        ax=axs[0, 0],
        color="teal",
        stat="density",
    )

    axs[0, 1].set_title("Distribution of target for target=" + str(target_uniq[1]))
    sns.histplot(
        data=data[data[target] == target_uniq[1]],
        x=predictor,
        kde=True,
        ax=axs[0, 1],
        color="orange",
        stat="density",
    )

    axs[1, 0].set_title("Boxplot w.r.t target")
    sns.boxplot(data=data, x=target, y=predictor, ax=axs[1, 0], palette="gist_rainbow")

    axs[1, 1].set_title("Boxplot (without outliers) w.r.t target")
    sns.boxplot(
        data=data,
        x=target,
        y=predictor,
        ax=axs[1, 1],
        showfliers=False,
        palette="gist_rainbow",
    )

    plt.tight_layout()
    plt.show()


# In[65]:


# function to plot stacked bar chart

def stacked_barplot(data, predictor, target):
    """
    Print the category counts and plot a stacked bar chart

    data: dataframe
    predictor: independent variable
    target: target variable
    """
    count = data[predictor].nunique()
    sorter = data[target].value_counts().index[-1]
    tab1 = pd.crosstab(data[predictor], data[target], margins=True).sort_values(
        by=sorter, ascending=False
    )
    print(tab1)
    print("-" * 120)
    tab = pd.crosstab(data[predictor], data[target], normalize="index").sort_values(
        by=sorter, ascending=False
    )
    tab.plot(kind="bar", stacked=True, figsize=(count + 5, 6))
    plt.legend(
        loc="lower left", frameon=False,
    )
    plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
    plt.show()


# #### Examining Target Variable Relationships with Numeric Features

# In[66]:


cont_var = df[['MonthlyIncome', 'Age', 'DurationOfPitch','ProdTaken']]


# In[67]:


sns.pairplot(cont_var, hue='ProdTaken')
plt.show()


# ##### Observation
# 
# * From the distributions above, it appears that in terms of age, monthly income, and duration of pitch, customers who purchased travel package do not differ strikingly from those who did not 

# In[68]:


num_var = df[['ProdTaken', 'MonthlyIncome', 'Age','DurationOfPitch','NumberOfPersonVisiting', 'NumberOfFollowups', 'NumberOfTrips', 'NumberOfChildrenVisiting','PitchSatisfactionScore']]


# In[69]:


plt.figure(figsize=(15, 7))
sns.heatmap(num_var.corr(), annot=True, vmin=-1, vmax=1, fmt=".2f", cmap="Blues")
plt.show()


# ##### Observation
# 
# * We do not observe correlations between the features 'MonthlyIncome', 'Age','DurationOfPitch','NumberOfPersonVisiting', 'NumberOfFollowups', 'NumberOfTrips', 'NumberOfChildrenVisiting','PitchSatisfactionScore' and the target variable, 'ProdTaken'
# * The highest ProdTaken correlation values being very mild negative correlations, -0.13 and -0.14 with Monthly Income and Age, respectively
# * We observe a positive correlation, 0.46, between Age and Monthly Income

# #### Age

# In[70]:


distribution_plot_wrt_target(df, 'Age', 'ProdTaken')


# #### Observation
# 
# * The graphs above give us an idea of the distributions separated between customers who purchased a travel package vs did not and accounting for the presence of outliers
# * We can see that because the distribution of Age values did not have outliers, that the boxplots give us the same results, showing that the age distribution for those who bought a travel package is slightly lower/younger than for those who did not buy

# #### DurationOfPitch

# In[71]:


distribution_plot_wrt_target(df, 'DurationOfPitch', 'ProdTaken')


# ##### Observation
# 
# * Without outliers, we see that duration of pitch was on average longer for customers who purchased a travel package

# #### MonthlyIncome

# In[72]:


distribution_plot_wrt_target(df, 'MonthlyIncome', 'ProdTaken')


# ##### Observation
# 
# * Looking at boxplots without the presence of outliers, we observe a monthly income distribution at lower income levels for customers who purchased a travel package

# #### NumberOfPersonVisiting

# In[73]:


stacked_barplot(df, "NumberOfPersonVisiting", 'ProdTaken')


# ##### Observation
# 
# * Customers who bought package looks like planned to travel in groups from 2 to 4 people

# #### NumberOfFollowups

# In[74]:


stacked_barplot(df, "NumberOfFollowups", 'ProdTaken')


# #### Observation
# 
# * The pattern here looks to be that more follow ups were related to greater sales though we cannot determine a causal link in that direction - meaning we cannot conclude that more follow ups resulted in more sales

# #### NumberOfTrips

# In[75]:


stacked_barplot(df, "NumberOfTrips", 'ProdTaken')


# ##### Observation
# 
# * Outliers, those who reported average 19 trips/year or more, are muddling the story told by the graph above. If we focus on average trips/year reported between 1 and 8, we see that customers who reported greater number of average trips annually also tended to purchase travel package in greater proportion

# #### NumberOfChildrenVisiting

# In[76]:


stacked_barplot(df, "NumberOfChildrenVisiting", 'ProdTaken')


# ##### Observation
# 
# * Customers who purchased travel package did not differ much from those who did not make purchase in terms of number of children planned to be travelling with though among those who bought travel package, over 70% planned to travel with 1 or 2 children

# #### Examining Target Variable Relationships with Categorical and Ordinal Features

# #### CityTier

# In[77]:


stacked_barplot(df, "CityTier", 'ProdTaken')


# #### Observation
# 
# * Customers from City Tiers 3 and 2 bought a travel package at higher rates than customers from tier 1 cities though just looking at customers who bought a travel package, 57% were from CityTier 1

# #### PreferredPropertyStar

# In[78]:


stacked_barplot(df, "PreferredPropertyStar", 'ProdTaken')


# #### Observation
# 
# * Customers who preferred 5 and 4 star hotels were more likely to purchase a travel package
# * Among the 920 customers who bought a travel package, most of them, 53%, reported preferrence for a 3 star hotel

# #### PitchSatisfactionScore

# In[79]:


stacked_barplot(df, "PitchSatisfactionScore", 'ProdTaken')


# ##### Observation
# 
# * Customers more satified with the sales pitch purchased a travel package at higher rates than customers less satified

# #### Gender

# In[80]:


stacked_barplot(df, "Gender", 'ProdTaken')


# ##### Observation
# 
# * Not much difference observed in terms of gender and product purchase

# #### MaritalStatus

# In[81]:


stacked_barplot(df, "MaritalStatus", 'ProdTaken')


# ##### Observation
# 
# * Single and Unmarried customers purchased a travel package at greater proportions than those married and divorced

# #### Occupation

# In[82]:


stacked_barplot(df, "Occupation", 'ProdTaken')


# ##### Observation
# 
# * The graph is being distorted by the 2 customers whose occupation is reported as free-lancer - these 2 happened to purchase a travel package giving us the false impression, through this graph, that free-lancers are highly inclined to purchase a travel package
# * Among the customers who bought a travel package, most were in the Salaried category

# #### Designation

# In[83]:


stacked_barplot(df, "Designation", 'ProdTaken')


# In[84]:


df.groupby(['Designation']).agg({'MonthlyIncome': ['mean', 'min', 'median', 'max']})


# ##### Observation
# 
# * Customers reporting designation as "Executive" purchased a travel package at a higher proportion than in other designation categories
# * Looking at a breakdown of salaries among the different designation categories, it looks like Executive designation is at the bottom of the monthly income level for this dataset - meaning that, proportionally, it looks like travel packages were more likely to be purchased among customers who made less per year than those who earned higher income

# #### Passport

# In[85]:


stacked_barplot(df, "Passport", 'ProdTaken')


# ##### Observation
# 
# * Among customers who own a passport, travel packages were sold at higher rates than among people who did not have a passport

# #### OwnCar

# In[86]:


stacked_barplot(df, "OwnCar", 'ProdTaken')


# ##### Observation
# 
# * Customers who own and who do not own a car seemed to have purchased a travel package at similar rates

# #### ProductPitched

# In[87]:


stacked_barplot(df, "ProductPitched", 'ProdTaken')


# ##### Observation
# 
# * Purchase of a travel package happened at a greater rate (30%) among customers pitched the Basic package
# * We observe much lower conversion rates among customers pitched the King (8.6%) and Super Deluxe (5.8%) packages

# #### TypeofContact

# In[88]:


stacked_barplot(df, "TypeofContact", 'ProdTaken')


# ##### Observations
# 
# * We observe a slightly greater proportion of sales among company-invited customers (approached by Visit With Us) though most sales came from self-inquiring customers

# ##### Please note
# * A detailed summary of observations from EDA provided below

# ### Examining customer characteristics by product package

# Filtering dataframe to include travel package buyers only to more easily observe customer characteristics
# by product type

# In[89]:


df1 = df.query('ProdTaken == 1')


# In[90]:


df1.drop("ProdTaken", axis=1, inplace=True)


# In[91]:


print(df1["ProductPitched"].value_counts())
print('-'*30)
print((df1["ProductPitched"].value_counts()/len(df1))*100)


# ##### Observation
# 
# * With low number of buyers (2% of the total each) it would be difficult to draw insights regarding the characteristics of customers who were pitched the Super Deluxe or King travel packages
# * 60% of customers pitched the Basic package purchased a travel package
# * In second place as the most popular product pitched came the Deluxe package with 22% of purchasers

# #### Product Pitched and Age

# In[92]:


df1.groupby(by=['ProductPitched'])['Age'].mean().reset_index().sort_values(['Age']
, ascending=True).plot(x='ProductPitched', y='Age', kind='bar', figsize=(15,5))
plt.show()


# In[93]:


# Relationship between travel package purchased and age

sns.boxplot(df1.ProductPitched, df1.Age);


# ##### Observation
# 
# * Purchasers who were pitched the Basic package tended to be younger

# #### Product Pitched and DurationOfPitch

# In[94]:


df1.groupby(by=['ProductPitched'])['DurationOfPitch'].mean().reset_index().sort_values(['DurationOfPitch']
, ascending=True).plot(x='ProductPitched', y='DurationOfPitch', kind='bar', figsize=(15,5))
plt.show()


# In[95]:


# Relationship between travel package purchased and DurationOfPitch

sns.boxplot(df1.ProductPitched, df1.DurationOfPitch);


# ##### Observation
# 
# * Duration of pitch for buyers who were pitched the Basic package were generally shorter than for buyers pitched Deluxe and Standard packages

# #### Product Pitched and Monthly Income

# In[96]:


df1.groupby(by=['ProductPitched'])['MonthlyIncome'].mean().reset_index().sort_values(['MonthlyIncome'], ascending=True).plot(x='ProductPitched', y='MonthlyIncome', kind='bar', figsize=(15,5))
plt.show()


# In[97]:


# Relationship between travel package purchased and monthly income

sns.boxplot(df1.ProductPitched, df1.MonthlyIncome);


# ##### Observation
# 
# * As expected, monthly income of customers who were pitched the more higher end travel packages tended to be higher than monthly income of purchasers pitched the Basic package

# #### Product Pitched and NumberOfPersonVisiting

# In[98]:


df1.groupby(by=['ProductPitched'])['NumberOfPersonVisiting'].mean().reset_index().sort_values(['NumberOfPersonVisiting']
, ascending=True).plot(x='ProductPitched', y='NumberOfPersonVisiting', kind='bar', figsize=(15,5))
plt.show()


# In[99]:


# Relationship between travel package purchased and NumberOfPersonVisiting

sns.boxplot(df1.ProductPitched, df1.NumberOfPersonVisiting);


# ##### Observation
# 
# * We do not observe much difference between purchasers pitched different travel packages and the expected number of people travelling with these customers

# #### Product Pitched and NumberOfFollowups 

# In[100]:


df1.groupby(by=['ProductPitched'])['NumberOfFollowups'].mean().reset_index().sort_values(['NumberOfFollowups']
, ascending=True).plot(x='ProductPitched', y='NumberOfFollowups', kind='bar', figsize=(15,5))
plt.show()


# In[101]:


# Relationship between travel package purchased and NumberOfFollowups

sns.boxplot(df1.ProductPitched, df1.NumberOfFollowups);


# #### Observation
# 
# * Number of follow ups for buyers pitched Basic and Deluxe packages did not seem to differ. We observe a narrower range for the distribution of mid 50% number of follow ups for customers pitched the Standard package

# #### Product Pitched and NumberOfTrips

# In[102]:


df1.groupby(by=['ProductPitched'])['NumberOfTrips'].mean().reset_index().sort_values(['NumberOfTrips']
, ascending=True).plot(x='ProductPitched', y='NumberOfTrips', kind='bar', figsize=(15,5))
plt.show()


# In[103]:


# Relationship between travel package purchased and NumberOfTrips

sns.boxplot(df1.ProductPitched, df1.NumberOfTrips);


# ##### Observation
# 
# * It looks like buyers who were pitched the Deluxe package reported being more avid travellers, with a higher average number of trips per year, than customer pitched the other travel packages

# #### Product Pitched and NumberOfChildrenVisiting

# In[104]:


df1.groupby(by=['ProductPitched'])['NumberOfChildrenVisiting'].mean().reset_index().sort_values(['NumberOfChildrenVisiting']
, ascending=True).plot(x='ProductPitched', y='NumberOfChildrenVisiting', kind='bar', figsize=(15,5))
plt.show()


# In[105]:


# Relationship between travel package purchased and NumberOfChildrenVisiting

sns.boxplot(df1.ProductPitched, df1.NumberOfChildrenVisiting);


# #### Observation
# 
# * We do not observe much difference in terms of the number of children planned to travel along between buyers pitched different packages

# #### Product Pitched and CityTier

# In[106]:


sns.countplot(df1.ProductPitched, hue=df1.CityTier);


# #### Observation
# 
# * We observe a larger number of buyers pitched the Basic package from CityTier 1
# * We also observe a larger number of buyers from CityTier 3 who were pitched the Deluxe package vs the Basic package

# #### Product Pitched and PreferredPropertyStar

# In[107]:


sns.countplot(df1.ProductPitched, hue=df1.PreferredPropertyStar.astype(int));


# ##### Observation
# 
# * Though we do not observe greatly different patterns in terms of preferred property stars between customers pitched different travel packages, buyers pitched the Basic package seem to prefer hotels rated at 3 start more so than customers pitched other packages

# #### Product Pitched and PitchSatisfactionScore

# In[108]:


sns.countplot(df1.ProductPitched, hue=df1.PitchSatisfactionScore.astype(int));


# ##### Observation
# 
# * In general, all buyers, regardless of product pitched, rated satisfaction with pitch most often at a 3

# #### Product Pitched and Gender

# In[109]:


sns.countplot(df1.ProductPitched, hue=df1.Gender);


# ##### Observation
# 
# * It looks like buyers, both male and female, were pitched the Basic package and Deluxe at higher rates than other packages

# #### Product Pitched and Marital Status

# In[110]:


sns.countplot(df1.ProductPitched, hue=df1.MaritalStatus);


# ##### Observation
# 
# * Buyers who were single were pitched the Basic package at higher rates than other buyers

# #### Product Pitched and Occupation

# In[111]:


sns.countplot(df1.ProductPitched, hue=df1.Occupation);


# ##### Observation
# 
# * We observe similar patterns in terms of product pitched for buyers in different occupations

# #### Product Pitched and Designation

# In[112]:


sns.countplot(df1.ProductPitched, hue=df1.Designation);


# ##### Observation
# 
# * It looks like Basic package was only pitched to buyers reported as Designation "Executive", Deluxe package only pitched to buyers reported as "Manager", Standard only pitched to buyers reported as "Senior Manager", King pitched only to "VPs" and "Super Deluxed" pitched only to "AVPs"

# #### Product Pitched and Passport

# In[113]:


sns.countplot(df1.ProductPitched, hue=df1.Passport);


# ##### Observation
# 
# * It looks like conversion rates were highest among buyers who have a passport and were pitched the Basic package - the opposite pattern observed for buyers who were pitched the Standard package

# #### Product Pitched and OwnCar

# In[114]:


sns.countplot(df1.ProductPitched, hue=df1.OwnCar);


# ##### Observation
# 
# * Patterns of product pitched look similar for buyers who own and do not own a car

# #### Product Pitched and TypeofContact

# In[115]:


sns.countplot(df1.ProductPitched, hue=df1.TypeofContact);


# ##### Observation
# 
# * Self-enquiry it looks like was the preferred contact mode for buyers across products pitched

# ## Handling outliers

# In[116]:


num_features = ["Age"
                , "DurationOfPitch"
                , "NumberOfPersonVisiting"
                , "NumberOfFollowups"
                , "NumberOfTrips"
                , "NumberOfChildrenVisiting"
                , "PitchSatisfactionScore"
                , "MonthlyIncome"]


# In[117]:


# let's look at box plot to see if outliers have been treated or not
plt.figure(figsize=(20, 30))

for i, variable in enumerate(num_features):
    plt.subplot(5, 4, i + 1)
    plt.boxplot(df[variable], whis=1.5)
    plt.tight_layout()
    plt.title(variable)

plt.show()


# In[118]:


outlier_features = [
                "DurationOfPitch"
                , "NumberOfPersonVisiting"
                , "NumberOfFollowups"
                , "NumberOfTrips"
                , "MonthlyIncome"]


# In[119]:


# Let's treat outliers by flooring and capping
def treat_outliers(df, col):
    """
    treats outliers in a variable
    col: str, name of the numerical variable
    df: dataframe
    col: name of the column
    """
    Q1 = df[col].quantile(0.25)  # 25th quantile
    Q3 = df[col].quantile(0.75)  # 75th quantile
    IQR = Q3 - Q1
    Lower_Whisker = Q1 - 1.5 * IQR
    Upper_Whisker = Q3 + 1.5 * IQR

    # all the values smaller than Lower_Whisker will be assigned the value of Lower_Whisker
    # all the values greater than Upper_Whisker will be assigned the value of Upper_Whisker
    df[col] = np.clip(df[col], Lower_Whisker, Upper_Whisker)

    return df


def treat_outliers_all(df, col_list):
    """
    treat outlier in all numerical variables
    col_list: list of numerical variables
    df: data frame
    """
    for c in col_list:
        df = treat_outliers(df, c)

    return df


# In[120]:


df = treat_outliers_all(df, outlier_features)


# In[121]:


plt.figure(figsize=(20, 30))

for i, variable in enumerate(outlier_features):
    plt.subplot(5, 4, i + 1)
    plt.boxplot(df[variable], whis=1.5)
    plt.tight_layout()
    plt.title(variable)

plt.show()


# #### Observation
# 
# * Outliers successfully treated

# ### Transforming categorical and ordinal features to numeric values for model-building purposes

# #### Examining value counts for categorical and ordinal features

# In[122]:


print(df.TypeofContact.value_counts())
print('-'*30)
print(df.Occupation.value_counts())
print('-'*30)
print(df.Gender.value_counts())
print('-'*30)
print(df.ProductPitched.value_counts())
print('-'*30)
print(df.MaritalStatus.value_counts())
print('-'*30)
print(df.Designation.value_counts())
print('-'*30)


# ##### Observation
# * We observe 2 outlier values for Occupation feature for the category "Free Lancer" - these two values will be altered to be counted among the most frequent value in the Occupation distribution, "Salaried"
# 
# In order to transform the features above to numeric values for model-building purposes
# 
# * We will apply One-Hot Encoding to the categorical features that are not ordinal: TypeofContact, Occupation, Gender,
# and MaritalStatus.
# 
# * We will apply Label Encoding to the categorical features that are ordinal: Designation and ProductPitched

# In[123]:


#Replacing "Free Lancer" category in Occupation for "Salaried"
df["Occupation"].replace({"Free Lancer": "Salaried"}, inplace=True)


# In[124]:


#Looking at MonthlyIncome by Designation position to ensure ordinal values are coded correctly
df.groupby(['Designation']).agg({'MonthlyIncome': ['mean', 'min', 'median', 'max']})


# ##### Observation
# 
# * Using MonthlyIncome as a proxy for job role rank we will code Designation as follows (breakdown of monthly income averages by job designation above):
#     VP (highest average income) > AVP (second to highest average income) > Senior Manager > Manager > Executive (lowest average income)

# In[125]:


replaceStruct = {
                "ProductPitched": {"Basic": 1, "Standard":2 , "Deluxe": 3, "Super Deluxe": 4, "King":5},
                 "Designation": {"Executive": 1 ,"Manager": 2, "Senior Manager": 3, "AVP": 4, "VP": 5},
                    }
oneHotCols=["TypeofContact","Occupation","Gender","MaritalStatus"]


# In[126]:


#Label encoding done for ProductPitched and Designation:
df =df.replace(replaceStruct)

#One-Hot encoding done for TypeofContact, Occupation, Gender, and MaritalStatus
df=pd.get_dummies(df, columns=oneHotCols,drop_first=True)
df.head(10)


# ### Converting non-float numeric columns to integer values
# 
# * Whole numbers expected from features below

# In[127]:


# Outlier treatment resulted in reverting integers back to float numbers, the code below used to correct that
df["Age"] = df["Age"].astype(int)
df["NumberOfPersonVisiting"] = df["NumberOfPersonVisiting"].astype(int)
df["NumberOfFollowups"] = df["NumberOfFollowups"].astype(int)
df["PreferredPropertyStar"] = df["PreferredPropertyStar"].astype(int)
df["NumberOfTrips"] = df["NumberOfTrips"].astype(int)
df["NumberOfChildrenVisiting"] = df["NumberOfChildrenVisiting"].astype(int)


# In[128]:


df.info()


# # <a id='link1'>Summary of EDA and Data Pre-Processing for Model Building</a>
# **Data Description:**
# 
# * 4800 rows of data with 17 features/independent variables (CustomerID is an ID variable so it was dropped from the data)
# * Dependent variable is **"ProdTaken"** indicating whether or not a customer purchased a pitched travel package. A value of 1 indicates customer purchased a travel package and 0 indicates customer did not
# * There were 1012 missing values in the dataset: 251 for DurationOfPitch, 233 for MonthlyIncome, 226 for Age, 140 for NumberOfTrips, 66 for NumberOfChildrenVisiting, 45 for NumberOfFollowUps, 26 for PreferredPropertyStar, and 25 for TypeOfContact
# 
# 
# **Data Cleaning:**
# 
# * CustomerID is an ID variable so it was dropped from the data.
# * All the features with missing values were carefully examined to assess the best approach for inputting missing values. Median value inputed for DurationOfPitch, MonthlyIncome, and Age missing values. Mean value inputed for missing values in NumberOfTrips, NumberOfChildrenVisiting, NumberOfFollowups, PreferredPropertyStar. TypeOfContact missing values filled with its most frequent value, "Self-Enquiry"
# * Gender had 3 different values: Male, Female, and Fe Male. The latter, a total of 155 values, was cleaned up and added to the Female category
# * Outliers were observed and treated for the features "DurationOfPitch", "NumberOfPersonVisiting", "NumberOfFollowups", "NumberOfTrips", and "MonthlyIncome"
# * We also observed 2 outlier values for Occupation feature for the category "Free Lancer" - these two values were altered to be counted among the most frequent value in the Occupation distribution, "Salaried"
# 
# 
# 
# **Observations from EDA:**
# 
# ###### Univariate Analysis
# * `Age`: The mean and median age of Visit With Us customer hover around late 30s - mean of 37.5 and median of 36 
# * `Duration of pitch`: We observe a right-skewness in the data given presence of large value outliers
# * `Monthly Income`: Mean and median annual income hovering around 23K. We observe the presence of outliers on both sides of the value distribution 
# * `Number Of Person Visiting`: The most frequent value observed for number of people intended to be travelling with customer is 3 followed by 2. Very few customers said they intended to take trip with just 1 person
# * `Number Of Followups`: Over 70% of customers received 3 to 4 follow ups from Visit With Us sales team
# * `Number of Trips`: Over 50% of customers reported taking on average 2 to 3 trips per year
# * `Number of Children Visiting`: 44% of customers reported planning to travel with 1 small child (age less than 5)
# * `City Tier`: Metadata describes "CityTier" as depending on the development of a city, population, facilities, and living standards. The categories are ordered i.e. Tier 1 > Tier 2 > Tier 3. The most frequent CityTier reported here as Tier 1 - unsure if this is the intended travel destination or reported demographic information related to customer residence
# * `Preferred Property Star`: The most popular preferred hotel property rating by customers is 3 stars
# * `Pitch Satisfaction Score`: Most common rating for sales pitch satisfaction was 3. Near 70% of customers rated sales pitch from 3 to 5
# * `Gender`: Close to 60% of customers are male
# * `Marital Status`: Nearly 50% of all customers in dataset are married
# * `Occupation`: Most customers are salaried professionals. 2 customers reported as free lancers
# * `Designation`: Most customers reported being Executives (nearly 40%) followed by managers (35%)
# * `Passport`: 70% of customers report not having a passport
# * `Own car`: 62% of customers report having a car
# * `Product Pitched`: The most often pitched travel package was the Basic (38%) followed by the Deluxe (35%). Packages ordered as Basic, Standard, Deluxe, Super Deluxe, King (see data pre-processing details below)
# * `Type of Contact`: 71% of customers reported approaching Visit With Us with travel inquiries rather than being first contacted by company
# * `ProdTaken`(target variable): Close to 19% of customers bought a Visit With Us travel package
# 
# ###### Bivariate Analysis
# 
# Examining relationships between features and target variable
# 
# **Correlations** between target variable and numeric features:
# * We do not observe correlations between the features 'MonthlyIncome', 'Age','DurationOfPitch','NumberOfPersonVisiting', 'NumberOfFollowups', 'NumberOfTrips', 'NumberOfChildrenVisiting','PitchSatisfactionScore' and the target variable, 'ProdTaken'
# * The highest ProdTaken correlation values being very mild negative correlations, -0.13 and -0.14 with Monthly Income and Age, respectively
# 
# * `Age and Target Variable`: age distribution for those who bought a travel package is slightly lower (meaning purchasers are in general younger) than for those who did not buy
# * `Duration of pitch and Target Variable`: we observe that duration of sales pitch was on average longer for customers who purchased a travel package
# * `Monthly Income and Target Variable`: we observe a monthly income distribution at lower income levels for customers who purchased a travel package 
# * `Number Of Person Visiting and Target Variable`: Customers who planned to travel with 2 to 4 people bought a travel package at greater proportions than customers who indicated plans to travel with 1 or 5 people
# * `Number Of Followups and Target Variable`: more follow ups were related to greater sales though we cannot conclude more follow ups resulted in more sales
# * `Number of Trips and Target Variable`: of customers who reported average trips/year between 1 and 8, we see that those who reported greater number of average trips annually also tended to purchase a travel package in greater proportion
# * `Number of Children Visiting and Target Variable`: customers who purchased a travel package did not differ much from non-buyers in terms of number of children travelling though among those who bought travel package, over 70% planned to travel with 1 or 2 children
# * `City Tier and Target Variable`: Customers from City Tiers 3 and 2 bought a travel package at higher rates than customers from tier 1 cities though just looking at customers who bought a travel package, 57% were from CityTier 1
# * `Preferred Property Star and Target Variable`: customers who preferred 5 and 4 star hotels were more likely to purchase a travel package. Among the 920 customers who bought a travel package, most of them, 53%, reported preferrence for a 3 star hotel
# * `Pitch Satisfaction Score and Target Variable`: customers more satified with the sales pitch purchased a travel package at higher rates than customers less satified
# * `Gender and Target Variable`: not much difference observed in terms of gender and product purchase
# * `Marital Status and Target Variable`: single and Unmarried customers purchased a travel package at greater proportions than those married and divorced
# * `Occupation and Target Variable`: among customers who bought a travel package, most were in the Salaried category
# * `Designation and Target Variable`: customers reporting designation as "Executive" purchased a travel package at a higher proportion than in other designation categories. Looking at a breakdown of salaries among the different designation categories, it looks like Executive designation is at the bottom of the monthly income level for this dataset - meaning that it looks like travel packages were more likely to be purchased by customers who made less per year than by those who earned higher income
# * `Passport and Target Variable`: among customers who own a passport, travel packages were sold at higher rates than among people who did not have a passport
# * `Own car and Target Variable`: customers who own and who do not own a car seemed to have purchased a travel package at similar rates
# * `Product Pitched and Target Variable`: purchase of a travel package occurred at a greater rate (30%) among customers pitched the Basic package. We observe much lower conversion rates among customers pitched the King (8.6%) and Super Deluxe (5.8%) packages
# * `Type of Contact and Target Variable`: we observe a slightly greater proportion of sales among company-invited customers (approached by Visit With Us) though most sales came from self-inquiring customers
# 
# 
# 
# ##### Customer Profiles by Product Pitched
# Here dataframe was filtered to include travel package buyers only to more easily observe customer characteristics
# by travel packaged pitched
# 
# * With low number of buyers (2% of the total each) it would be difficult to draw insights regarding the characteristics of customers who were pitched the Super Deluxe or King travel packages
# * 60% of customers pitched the Basic package purchased a travel package
# * In second place as the most popular product pitched came the Deluxe package with 22% of purchasers
# 
# BASIC PACKAGE:
# * Most popular travel package
# * Customer base made primarily of single adults in their early to mid 30s
# * Salaried professionals with monthly income at the lower-end of the scale compared to other customers, approximately 20k
# * Basic package customers tend to reside in larger, more developed cities (CityTier 1)
# * They have passports and show eagerness to travel, taking 3 trips per year on average
# * Customers in this category tend to approach Visit With Us - most initial contact done through self-inquiry
# 
# DELUXE PACKAGE:
# * The most travel-eager customer of the group, taking 3.5 trips per year on average
# * Professionals in their late 30s and 40s, mostly married or with a partner ("Unmarried")
# * Compared to other customer groups, Deluxe customers on average tended to take longer to be persuaded to purchase and/or need more assistance from sales
# * Salaried and professionals working for small businesses making about 23k/month
# * Mostly residents of smaller cities, with less facilities and lower standard of living (CityTier 3)
# * Customers here also tended to approach Visit With Us - most contact done through self-inquiry
# 
# STANDARD PACKAGE:
# * The 3rd most popular of the 5 travel packages
# * Customers in this group tended not to own a passport and be less prone to travel than other customers - perhaps also that is why duration of pitch was on average the longest for this group
# * Group consisting of salaried and professionals working for mostly small businesses. Monthly income around 25k
# * Customers in their late 30s to late 40s, mostly married and residing in CityTier 3 cities (smaller, less facilities)
# 
# SUPER DELUXE AND KING:
# **Important**: merging the 2 customer groups into a single customer profile category given the low number (2% of total buyers for each) of customers in these categories
# * The oldest age set among Visit With Us customers consisting of customers in their late 40s and 50s
# * Advanced career professionals (VPs and AVPs) with monthly income at 30 to 35k
# * They tend to own a passport and report taking on average 3 trips/year
# 
# **Actions for data pre-processing:**
# 
# * ProductPitched and Designation features were assigned numeric values indicating ordinal levels as follows:
# **ProductPitched**: King (5) > Super Deluxe (4) > Deluxe (3) > Standard (2) > Basic (1)
# **Designation** (based on monthly income levels): VP (5) > AVP (4) > Senior Manager (3) > Manager (2) > Executive (1)
# * One-Hot encoding was used to create dummy variables for TypeofContact, Occupation, Gender and Marital Status
# * Outliers, non-sensical, and missing values all treated to prepare data for model building
# * All features transformed in numeric data types for model building purposes

# ## Model Building - Approach
# 1. Data preparation
# 2. Partition the data into train and test set.
# 3. Build model on the train data.
# 4. Tune the model if required.
# 5. Test the data on test set.

# ## Split Data
# 
# * When classification problems exhibit a significant imbalance in the distribution of the target classes, it is good to use stratified sampling to ensure that relative class frequencies are approximately preserved in train and test sets. 
# * This is done using the `stratify` parameter in the train_test_split function.

# In[129]:


X = df.drop(["ProdTaken"], axis=1)
y = df.pop("ProdTaken")


# **The Stratify arguments maintain the original distribution of classes in the target variable while splitting the data into train and test sets.**

# In[130]:


y.value_counts()/len(y)


# In[131]:


# Splitting data into training and test set:
X_train, X_test, y_train, y_test =train_test_split(X, y, test_size=0.3, random_state=1,stratify=y)
print(X_train.shape, X_test.shape)


# In[132]:


y_train.value_counts()/len(y_train)


# In[133]:


y_test.value_counts()/len(y_test)


# ## Model evaluation criterion
# 
# 
# ### Goal of the model
# 
# * To recap, the goal is to build a model that will help the Policy Maker and Marketing Teamt predict which customer is more likely to purchase the newly introduced travel package (Wellness Tourism Package)
# 
# 
# For the predictive model it is important to remember the following in terms of target variable classification: 
# 
# - True Positive: customer predicted by the model as “will purchase travel package” and who will in fact purchase package.
# - False Positive: customer predicted by the model as “will purchase travel package” but who actually will not make the purchase.
# - True Negative: customer predicted by the model as “will not purchase travel package” and who in fact will not make the purchase.
# - False Negative: customer predicted by the model as “will not purchase travel package” but who will actually buy the travel package.
# 
# ###### Precision:
# 
# - Of customers predicted by model to purchase travel package what proportion actually did?
# - True positive / (True positive + False positive)
# 
# 
# ###### Recall: 
# - Of customers who actually bought a travel package, what proportion were actually correctly predicted as buyers?
# - True positive / (True positive + False negative)
# 
# 
# ###### F1 score, also known as balanced F-score or F-measure:
# 
# The F1 score can be interpreted as a weighted average of the precision and recall, where an F1 score reaches its best value at 1 and worst score at 0. The relative contribution of precision and recall to the F1 score are equal. The formula for the F1 score is:
# 
# F1 = 2 * (precision * recall) / (precision + recall)
# 
# 
# ##### Better models have higher values for precision and recall.
# 
# For example, 94% precision would mean that almost all customers predicted as buyers did in fact make purchase and 97% recall would mean almost all customers who bought travel package were correctly identified as buyers.
# 
# If the model performed with 95% precision but 50% recall this would mean that when it identifies someone as a buyer it is largely correct, but it predicts as non-buyers half of those customers who did in fact made a purchase.
# 
# 
# ### Precision, Recall, F1 scores
# 
# Within any one model, we can decide to emphasize precision, recall or a balance of the two which is provided by F1 scores - the decision comes down to context. 
# 
# 
# ### The model can make wrong predictions as:
# 1. Predicting customer will purchase travel package when customer will not.
# 2. Predicting customer will not purchase travel package but customer will.
# 
# 
# ### Which case is more important? 
# 
# 1. If the model predicts a customer will not buy travel package but customer will then the company would miss branding opportunity with a potential loyal customer (attempting to maximize 80/20 rule, where 80% of a company's sales come from 20% of its customers, i.e., the most loyal customer base)
# 2. If the model predicts a customer will buy the travel package but customer actually will not then the company would be risking losing resources reaching out to customers less likely to buy. 
# 
# ### Which metric to optimize?
# * To balance out the risks and opportunities related to precision and recall we would want F1 metric to be maximized, the greater the F1-Score higher the chances of predicting both classes correctly.

# **Let's define a function to provide metric scores on the train and test set and a function to show confusion matrix so that we do not have to use the same code repetitively while evaluating models.**

# In[134]:


# defining a function to compute different metrics to check performance of a classification model built using sklearn
def model_performance_classification_sklearn(model, predictors, target):
    """
    Function to compute different metrics to check classification model performance

    model: classifier
    predictors: independent variables
    target: dependent variable
    """

    # predicting using the independent variables
    pred = model.predict(predictors)

    acc = accuracy_score(target, pred)  # to compute Accuracy
    recall = recall_score(target, pred)  # to compute Recall
    precision = precision_score(target, pred)  # to compute Precision
    f1 = f1_score(target, pred)  # to compute F1-score

    # creating a dataframe of metrics
    df_perf = pd.DataFrame(
        {
            "Accuracy": acc,
            "Recall": recall,
            "Precision": precision,
            "F1": f1,
        },
        index=[0],
    )

    return df_perf


# In[135]:


def confusion_matrix_sklearn(model, predictors, target):
    """
    To plot the confusion_matrix with percentages

    model: classifier
    predictors: independent variables
    target: dependent variable
    """
    y_pred = model.predict(predictors)
    cm = confusion_matrix(target, y_pred)
    labels = np.asarray(
        [
            ["{0:0.0f}".format(item) + "\n{0:.2%}".format(item / cm.flatten().sum())]
            for item in cm.flatten()
        ]
    ).reshape(2, 2)

    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=labels, fmt="")
    plt.ylabel("True label")
    plt.xlabel("Predicted label")


# ## Decision Tree Classifier

# In[136]:


#Fitting the model
d_tree = DecisionTreeClassifier(random_state=1)
d_tree.fit(X_train,y_train)

#Calculating different metrics
d_tree_model_train_perf=model_performance_classification_sklearn(d_tree,X_train,y_train)
print("Training performance:\n",d_tree_model_train_perf)
d_tree_model_test_perf=model_performance_classification_sklearn(d_tree,X_test,y_test)
print("Testing performance:\n",d_tree_model_test_perf)

#Creating confusion matrix
confusion_matrix_sklearn(d_tree,X_test,y_test)


# ##### Observation
# 
# * The results reveal that the decision tree model is overfitting the training data and yielding not such a good performance in terms of recall, precision and the balance of these two metrics, F1 

# ### Hyperparameter Tuning

# In[156]:


#Choose the type of classifier. 
dtree_estimator = DecisionTreeClassifier(class_weight={0:0.19,1:0.81},random_state=1)

# Grid of parameters to choose from
parameters = {'max_depth': np.arange(2,30), 
              'min_samples_leaf': [1, 2, 5, 7, 10],
              'max_leaf_nodes' : [2, 3, 5, 10,15],
              'min_impurity_decrease': [0.0001,0.001,0.01,0.1]
             }

# Type of scoring used to compare parameter combinations
scorer = metrics.make_scorer(metrics.f1_score)

# Run the grid search
grid_obj = GridSearchCV(dtree_estimator, parameters, scoring=scorer,n_jobs=-1)
grid_obj = grid_obj.fit(X_train, y_train)

# Set the clf to the best combination of parameters
dtree_estimator = grid_obj.best_estimator_

# Fit the best algorithm to the data. 
dtree_estimator.fit(X_train, y_train)


# In[157]:


#Calculating different metrics
dtree_estimator_model_train_perf=model_performance_classification_sklearn(dtree_estimator,X_train,y_train)
print("Training performance:\n",dtree_estimator_model_train_perf)
dtree_estimator_model_test_perf=model_performance_classification_sklearn(dtree_estimator,X_test,y_test)
print("Testing performance:\n",dtree_estimator_model_test_perf)

#Creating confusion matrix
confusion_matrix_sklearn(dtree_estimator,X_test,y_test)


# ##### Observation
# 
# * The good news for the tuned decision tree model is that it is no longer overfitting
# * As we can see, however, it is making a  lot of mistakes, misclassifying 21.33% of the target value predictions in the test dataset

# ## Random Forest Classifier

# In[141]:


#Fitting the model
rf_estimator = RandomForestClassifier(random_state=1)
rf_estimator.fit(X_train,y_train)

#Calculating different metrics
rf_estimator_model_train_perf=model_performance_classification_sklearn(rf_estimator,X_train,y_train)
print("Training performance:\n",rf_estimator_model_train_perf)
rf_estimator_model_test_perf=model_performance_classification_sklearn(rf_estimator,X_test,y_test)
print("Testing performance:\n",rf_estimator_model_test_perf)

#Creating confusion matrix
confusion_matrix_sklearn(rf_estimator,X_test,y_test)


# ##### Observation
# 
# * We see training model overfitting data
# * Model produces high accuracy and precision scores but low recall

# ### Hyperparameter Tuning

# In[142]:


# Choose the type of classifier. 
rf_tuned = RandomForestClassifier(class_weight={0:0.19,1:0.81},random_state=1,oob_score=True,bootstrap=True)

parameters = {  
                'max_depth': list(np.arange(5,30,5)) + [None],
                'max_features': ['sqrt','log2',None],
                'min_samples_leaf': np.arange(1,15,5),
                'min_samples_split': np.arange(2, 20, 5),
                'n_estimators': np.arange(10,150,10)}


# Type of scoring used to compare parameter combinations
scorer = metrics.make_scorer(metrics.f1_score)

# Run the grid search
grid_obj = GridSearchCV(rf_tuned, parameters, scoring=scorer, cv=5,n_jobs=-1)
grid_obj = grid_obj.fit(X_train, y_train)

# Set the clf to the best combination of parameters
rf_tuned = grid_obj.best_estimator_

# Fit the best algorithm to the data. 
rf_tuned.fit(X_train, y_train)


# In[143]:


#Calculating different metrics
rf_tuned_model_train_perf=model_performance_classification_sklearn(rf_tuned,X_train,y_train)
print("Training performance:\n",rf_tuned_model_train_perf)
rf_tuned_model_test_perf=model_performance_classification_sklearn(rf_tuned,X_test,y_test)
print("Testing performance:\n",rf_tuned_model_test_perf)

#Creating confusion matrix
confusion_matrix_sklearn(rf_tuned,X_test,y_test)


# ##### Observation
# 
# * The tuned model is still overfitting but less so than the original random forest model
# * We observe a slight improvement in F1 score (i.e., balance between recall and precision performance), from 0.706 to 0.754

# ## Bagging Classifier

# In[144]:


#Fitting the model
bagging_classifier = BaggingClassifier(random_state=1)
bagging_classifier.fit(X_train,y_train)

#Calculating different metrics
bagging_classifier_model_train_perf=model_performance_classification_sklearn(bagging_classifier,X_train,y_train)
print(bagging_classifier_model_train_perf)
bagging_classifier_model_test_perf=model_performance_classification_sklearn(bagging_classifier,X_test,y_test)
print(bagging_classifier_model_test_perf)

#Creating confusion matrix
confusion_matrix_sklearn(bagging_classifier,X_test,y_test)


# ##### Observation
# 
# * Bagging model is overfitting training data though less so than decision tree model
# * We observe higher performance in terms of precision but not so strong recall

# ### Hyperparameter Tuning

# In[145]:


# Choose the type of classifier. 
bagging_estimator_tuned = BaggingClassifier(random_state=1)

# Grid of parameters to choose from
parameters = {'max_samples': [0.7,0.8,0.9,1], 
              'max_features': [0.7,0.8,0.9,1],
              'n_estimators' : [10,20,30,40,50],
             }

# Type of scoring used to compare parameter combinations
scorer = metrics.make_scorer(metrics.f1_score)

# Run the grid search
grid_obj = GridSearchCV(bagging_estimator_tuned, parameters, scoring=scorer,cv=5)
grid_obj = grid_obj.fit(X_train, y_train)

# Set the clf to the best combination of parameters
bagging_estimator_tuned = grid_obj.best_estimator_

# Fit the best algorithm to the data.
bagging_estimator_tuned.fit(X_train, y_train)


# In[146]:


#Calculating different metrics
bagging_estimator_tuned_model_train_perf=model_performance_classification_sklearn(bagging_estimator_tuned,X_train,y_train)
print(bagging_estimator_tuned_model_train_perf)
bagging_estimator_tuned_model_test_perf=model_performance_classification_sklearn(bagging_estimator_tuned,X_test,y_test)
print(bagging_estimator_tuned_model_test_perf)

#Creating confusion matrix
confusion_matrix_sklearn(bagging_estimator_tuned,X_test,y_test)


# ##### Observation
# 
# * After fine-tuning adjustments we still see model overfitting training data
# * No real improvement in the performance on test data observed
# * Although F1 score performance was attempted to be maximize we do not observe much improvement

# ## AdaBoost Classifier

# In[147]:


#Fitting the model
ab_classifier = AdaBoostClassifier(random_state=1)
ab_classifier.fit(X_train,y_train)

#Calculating different metrics
ab_classifier_model_train_perf=model_performance_classification_sklearn(ab_classifier,X_train,y_train)
print(ab_classifier_model_train_perf)
ab_classifier_model_test_perf=model_performance_classification_sklearn(ab_classifier,X_test,y_test)
print(ab_classifier_model_test_perf)

#Creating confusion matrix
confusion_matrix_sklearn(ab_classifier,X_test,y_test)


# ##### Observation
# 
# * The good news we observe here is that the model is not overfitting the data
# * The not-so-good news is that it is not performing well, especially in terms of recall

# ### Hyperparameter Tuning

# In[148]:


# Choose the type of classifier. 
abc_tuned = AdaBoostClassifier(random_state=1)

# Grid of parameters to choose from
parameters = {
    #Let's try different max_depth for base_estimator
    "base_estimator":[DecisionTreeClassifier(max_depth=1),DecisionTreeClassifier(max_depth=2),
                      DecisionTreeClassifier(max_depth=3)],
    "n_estimators": np.arange(10,110,10),
    "learning_rate":np.arange(0.1,2,0.1)
}

# Type of scoring used to compare parameter  combinations
scorer = metrics.make_scorer(metrics.f1_score)

# Run the grid search
grid_obj = GridSearchCV(abc_tuned, parameters, scoring=scorer,cv=5)
grid_obj = grid_obj.fit(X_train, y_train)

# Set the clf to the best combination of parameters
abc_tuned = grid_obj.best_estimator_

# Fit the best algorithm to the data.
abc_tuned.fit(X_train, y_train)


# In[149]:


#Calculating different metrics
abc_tuned_model_train_perf=model_performance_classification_sklearn(abc_tuned,X_train,y_train)
print(abc_tuned_model_train_perf)
abc_tuned_model_test_perf=model_performance_classification_sklearn(abc_tuned,X_test,y_test)
print(abc_tuned_model_test_perf)

#Creating confusion matrix
confusion_matrix_sklearn(abc_tuned,X_test,y_test)


# ##### Observation
# 
# * Unfortunately, the tuned model started to overfit data
# * We observe improved though still underperforming F1 score due to low recall performance

# ## Gradient Boosting Classifier

# In[150]:


#Fitting the model
gb_classifier = GradientBoostingClassifier(random_state=1)
gb_classifier.fit(X_train,y_train)

#Calculating different metrics
gb_classifier_model_train_perf=model_performance_classification_sklearn(gb_classifier,X_train,y_train)
print("Training performance:\n",gb_classifier_model_train_perf)
gb_classifier_model_test_perf=model_performance_classification_sklearn(gb_classifier,X_test,y_test)
print("Testing performance:\n",gb_classifier_model_test_perf)

#Creating confusion matrix
confusion_matrix_sklearn(gb_classifier,X_test,y_test)


# ##### Observation
# 
# * Though not overfitting the data, the model is performing poorly in terms of recall (meaning presence of a high number of false negative predictions) and thereby F1 score

# ### Hyperparameter Tuning

# In[151]:


# Choose the type of classifier. 
gbc_tuned = GradientBoostingClassifier(init=AdaBoostClassifier(random_state=1),random_state=1)

# Grid of parameters to choose from
parameters = {
    "n_estimators": [100,150,200,250],
    "subsample":[0.8,0.9,1],
    "max_features":[0.7,0.8,0.9,1]
}

# Type of scoring used to compare parameter combinations
scorer = metrics.make_scorer(metrics.f1_score)

# Run the grid search
grid_obj = GridSearchCV(gbc_tuned, parameters, scoring=scorer,cv=5)
grid_obj = grid_obj.fit(X_train, y_train)

# Set the clf to the best combination of parameters
gbc_tuned = grid_obj.best_estimator_

# Fit the best algorithm to the data.
gbc_tuned.fit(X_train, y_train)


# In[152]:


#Calculating different metrics
gbc_tuned_model_train_perf=model_performance_classification_sklearn(gbc_tuned,X_train,y_train)
print("Training performance:\n",gbc_tuned_model_train_perf)
gbc_tuned_model_test_perf=model_performance_classification_sklearn(gbc_tuned,X_test,y_test)
print("Testing performance:\n",gbc_tuned_model_test_perf)

#Creating confusion matrix
confusion_matrix_sklearn(gbc_tuned,X_test,y_test)


# ##### Observation
# 
# * Tuned model improved slightly in terms of F1 performance but began to overfit
# * Type II error: model misclassifying a lot of actual buyers as non-buyers (nearly 10%)

# ## XGBoost Classifier

# In[153]:


#Fitting the model
xgb_classifier = XGBClassifier(random_state=1, eval_metric='logloss')
xgb_classifier.fit(X_train,y_train)

#Calculating different metrics
xgb_classifier_model_train_perf=model_performance_classification_sklearn(xgb_classifier,X_train,y_train)
print("Training performance:\n",xgb_classifier_model_train_perf)
xgb_classifier_model_test_perf=model_performance_classification_sklearn(xgb_classifier,X_test,y_test)
print("Testing performance:\n",xgb_classifier_model_test_perf)

#Creating confusion matrix
confusion_matrix_sklearn(xgb_classifier,X_test,y_test)


# ##### Observation
# 
# * Model is overfitting data as we observe especially large discrepancy between training and test scores for recall and F1

# ### Hyperparameter Tuning

# In[154]:


# Choose the type of classifier. 
xgb_tuned = XGBClassifier(random_state=1, eval_metric='logloss')

# Grid of parameters to choose from
parameters = {
    "n_estimators": [30,50,75],
    "scale_pos_weight":[1,2,5],
    "subsample":[0.7,0.9,1],
    "gamma":[0, 1, 3, 5],
    "learning_rate":[0.05, 0.1,0.2],
    "colsample_bytree":[0.7,0.9,1],
    "colsample_bylevel":[0.5,0.7,1]
}

# Type of scoring used to compare parameter combinations
scorer = metrics.make_scorer(metrics.f1_score)

# Run the grid search
grid_obj = GridSearchCV(xgb_tuned, parameters,scoring=scorer,cv=5)
grid_obj = grid_obj.fit(X_train, y_train)

# Set the clf to the best combination of parameters
xgb_tuned = grid_obj.best_estimator_

# Fit the best algorithm to the data.
xgb_tuned.fit(X_train, y_train)


# In[155]:


#Calculating different metrics
xgb_tuned_model_train_perf=model_performance_classification_sklearn(xgb_tuned,X_train,y_train)
print("Training performance:\n",xgb_tuned_model_train_perf)
xgb_tuned_model_test_perf=model_performance_classification_sklearn(xgb_tuned,X_test,y_test)
print("Testing performance:\n",xgb_tuned_model_test_perf)

#Creating confusion matrix
confusion_matrix_sklearn(xgb_tuned,X_test,y_test)


# #### Observation
# 
# * We see XGBoost tuned model overfitting less than the original and with improved performance, yielding the highest F1 performance score of all our models: 0.805

# ## Stacking Classifier

# In[162]:


estimators = [('Random Forest Tuned',rf_tuned), ('Decision Tree Estimator',dtree_estimator), ('AdaBoost Classifier Tuned',abc_tuned)]

final_estimator = xgb_tuned

stacking_classifier= StackingClassifier(estimators=estimators,final_estimator=final_estimator)

stacking_classifier.fit(X_train,y_train)


# In[163]:


#Calculating different metrics
stacking_classifier_model_train_perf=model_performance_classification_sklearn(stacking_classifier,X_train,y_train)
print("Training performance:\n",stacking_classifier_model_train_perf)
stacking_classifier_model_test_perf=model_performance_classification_sklearn(stacking_classifier,X_test,y_test)
print("Testing performance:\n",stacking_classifier_model_test_perf)

#Creating confusion matrix
confusion_matrix_sklearn(stacking_classifier,X_test,y_test)


# ##### Observation
# 
# * Although overall model is delivering pretty good performance scores it is unfortunately overfitting training data

# ## Comparing all models

# In[164]:


# training performance comparison

models_train_comp_df = pd.concat(
    [d_tree_model_train_perf.T,dtree_estimator_model_train_perf.T,rf_estimator_model_train_perf.T,rf_tuned_model_train_perf.T,
     bagging_classifier_model_train_perf.T,bagging_estimator_tuned_model_train_perf.T,ab_classifier_model_train_perf.T,
     abc_tuned_model_train_perf.T,gb_classifier_model_train_perf.T,gbc_tuned_model_train_perf.T,xgb_classifier_model_train_perf.T,
    xgb_tuned_model_train_perf.T,stacking_classifier_model_train_perf.T],
    axis=1,
)
models_train_comp_df.columns = [
    "Decision Tree",
    "Decision Tree Estimator",
    "Random Forest Estimator",
    "Random Forest Tuned",
    "Bagging Classifier",
    "Bagging Estimator Tuned",
    "Adaboost Classifier",
    "Adabosst Classifier Tuned",
    "Gradient Boost Classifier",
    "Gradient Boost Classifier Tuned",
    "XGBoost Classifier",
    "XGBoost Classifier Tuned",
    "Stacking Classifier"]
print("Training performance comparison:")
models_train_comp_df


# In[165]:


# testing performance comparison

models_test_comp_df = pd.concat(
    [d_tree_model_test_perf.T,dtree_estimator_model_test_perf.T,rf_estimator_model_test_perf.T,rf_tuned_model_test_perf.T,
     bagging_classifier_model_test_perf.T,bagging_estimator_tuned_model_test_perf.T,ab_classifier_model_test_perf.T,
     abc_tuned_model_test_perf.T,gb_classifier_model_test_perf.T,gbc_tuned_model_test_perf.T,xgb_classifier_model_test_perf.T,
    xgb_tuned_model_test_perf.T,stacking_classifier_model_test_perf.T],
    axis=1,
)
models_test_comp_df.columns = [
    "Decision Tree",
    "Decision Tree Estimator",
    "Random Forest Estimator",
    "Random Forest Tuned",
    "Bagging Classifier",
    "Bagging Estimator Tuned",
    "Adaboost Classifier",
    "Adabosst Classifier Tuned",
    "Gradient Boost Classifier",
    "Gradient Boost Classifier Tuned",
    "XGBoost Classifier",
    "XGBoost Classifier Tuned",
    "Stacking Classifier"]
print("Testing performance comparison:")
models_test_comp_df


# #### Observation
# 
# * The majority of the models are overfitting the training data in terms of f1-score.
# * When comparing performance scores across all of the models tested, we conclude that the tuned XGBoost model was our highest performer
# * Though we still observe overfitting, tuned XGBoost is yielding a good, balanced performance in all measures and the highest F1 score of all models tested  

# ### Feature importance of Tuned XGBoost

# In[166]:


# importance of features in the tree building ( The importance of a feature is computed as the 
#(normalized) total reduction of the criterion brought by that feature. It is also known as the Gini importance )

print(pd.DataFrame(xgb_tuned.feature_importances_, columns = ["Imp"], index = X_train.columns).sort_values(by = 'Imp', ascending = False))


# In[167]:


feature_names = X_train.columns
importances = xgb_tuned.feature_importances_
indices = np.argsort(importances)

plt.figure(figsize=(12,12))
plt.title('Feature Importances')
plt.barh(range(len(indices)), importances[indices], color='violet', align='center')
plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
plt.xlabel('Relative Importance')
plt.show()


# ##### Observation
# 
# * The graph above shows us a break down of feature importance for the tuned XGBoost model
# * We see that the top 5 most important predictive features for the model are:
# 1. Whether customer owns a passport
# 2. Job designation (rank) of customer
# 3. City tier level in which customer resides
# 4. Travel package pitched to customer
# 5. Marital Status
# * The model indicates that such features should be weighted more heavily by the marketing team as it devises a campaign to launch the new Wellness travel package

# ## Business Insights and Recommendations

# ###### Insights
# 
# * Based on the best predictive model, owning a passport (or not owning a passport) is a crucial component of whether a customer will purchase a travel package and what package they will be inclined to purchase
# * Job rank (which is also a proxy for age and income) is also an important factor in terms of what travel package customers would be more likely to purchase: for example, younger, lower-ranked professionals are more likely to buy the Basic Package, whereas older higher-ranked professionals incline towards Standard and Deluxe Packages
# * Another important and less obvious factor revealed by the model suggests the importance of level of development, population, facilities, and living standards of the city in which a customer resides
# * The type of travel package pitched to customer also plays an important role, suggesting that a well-targeted marketing effort is crucial
# * Marital status also plays a factor in determining travel package purchase decisions
# 
# ###### Recommendations
# 
# Overall, the model gives us clues as to what factors should be prioritized in marketing the new Wellness package. 
# Based on the insights, we suggest a customer segmentation strategy that takes into account:
# * Whether or not a customer owns a passport - customers who do being generally more inclined to purchase a travel package
# * The professional rank of the customer - an idea would be to develop different kinds of Wellness packages catering to the needs of customers at different points in their career development
# * The characteristics of the customer’s city of residence which can give the marketing team important clues as to the needs of the customer - what kind of experiences a customer would likely seek given what their city of residence currently has (or does not have) to offer
# * The package that is pitched to ensure that it is carefully tailored and promoted to the right customer
# * The marital status of the customer. Again, this speaks to the importance of understanding key aspects of the life of the customer (both in personal and professional realms) and of developing a marketing strategy that caters to and is shaped by these characteristics

# In[ ]:




