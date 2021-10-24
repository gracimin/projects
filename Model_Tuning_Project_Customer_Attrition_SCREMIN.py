#!/usr/bin/env python
# coding: utf-8

# #### Code and analysis by Gracieli Scremin

# # Credit Card Users Churn Prediction : Problem Statement
# 
# #### Description
# 
# ##### Background & Context
# 
# The Thera bank recently saw a steep decline in the number of users of their credit card, credit cards are a good source of income for banks because of different kinds of fees charged by the banks like annual fees, balance transfer fees, and cash advance fees, late payment fees, foreign transaction fees, and others. Some fees are charged to every user irrespective of usage, while others are charged under specified circumstances.
# 
# Customers’ leaving credit cards services would lead bank to loss, so the bank wants to analyze the data of customers and identify the customers who will leave their credit card services and reason for same – so that bank could improve upon those areas
# 
# You as a Data scientist at Thera bank need to come up with a classification model that will help the bank improve its services so that customers do not renounce their credit cards
# 
# You need to identify the best possible model that will give the required performance
# 
# ##### Objective
# 
# Explore and visualize the dataset.
# Build a classification model to predict if the customer is going to churn or not
# Optimize the model using appropriate techniques
# Generate a set of insights and recommendations that will help the bank
# 
# ##### Data Dictionary:
# 
# * CLIENTNUM: Client number. Unique identifier for the customer holding the account
# * Attrition_Flag: Internal event (customer activity) variable - if the account is closed then "Attrited Customer" else "Existing Customer"
# * Customer_Age: Age in Years
# * Gender: Gender of the account holder
# * Dependent_count: Number of dependents
# * Education_Level:  Educational Qualification of the account holder - Graduate, High School, Unknown, Uneducated, College(refers to a college student), Post-Graduate, Doctorate.
# * Marital_Status: Marital Status of the account holder
# * Income_Category: Annual Income Category of the account holder
# * Card_Category: Type of Card
# * Months_on_book: Period of relationship with the bank
# * Total_Relationship_Count: Total no. of products held by the customer
# * Months_Inactive_12_mon: No. of months inactive in the last 12 months
# * Contacts_Count_12_mon: No. of Contacts between the customer and bank in the last 12 months
# * Credit_Limit: Credit Limit on the Credit Card
# * Total_Revolving_Bal: The balance that carries over from one month to the next is the revolving balance
# * Avg_Open_To_Buy: Open to Buy refers to the amount left on the credit card to use (Average of last 12 months)
# * Total_Trans_Amt: Total Transaction Amount (Last 12 months)
# * Total_Trans_Ct: Total Transaction Count (Last 12 months)
# * Total_Ct_Chng_Q4_Q1: Ratio of the total transaction count in 4th quarter and the total transaction count in 1st quarter
# * Total_Amt_Chng_Q4_Q1: Ratio of the total transaction amount in 4th quarter and the total transaction amount in 1st quarter
# * Avg_Utilization_Ratio: Represents how much of the available credit the customer spent

# #### Importing libraries

# In[360]:


# Libraries to help with reading and manipulating data
import numpy as np
import pandas as pd

# Libraries to help with data visualization
import matplotlib.pyplot as plt
import seaborn as sns

# to display all columns
pd.set_option('display.max_columns', None)

# To tune model, get different metric scores and split data
from sklearn import metrics
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    f1_score,
    accuracy_score,
    recall_score,
    precision_score,
    confusion_matrix,
    roc_auc_score,
    plot_confusion_matrix,
)

# To impute missing values
from sklearn.impute import KNNImputer
from sklearn.impute import SimpleImputer

# To help with model building
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
    BaggingClassifier,
)
from xgboost import XGBClassifier

# To use statistical functions
import scipy.stats as stats

# To oversample and undersample data
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

# To be used for data scaling and one hot encoding
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder

# To be used for tuning the model
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

# To be used for creating pipelines and personalizing them
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# To define maximum number of columns to be displayed in a dataframe
pd.set_option("display.max_columns", None)

# To supress scientific notations for a dataframe
pd.set_option("display.float_format", lambda x: "%.3f" % x)


# To suppress the warning
import warnings

warnings.filterwarnings("ignore")


# #### Loading and running a basic check on the data

# In[361]:


data = pd.read_csv("BankChurners.csv")


# In[362]:


data.shape


# In[363]:


data.head()


# In[364]:


#copying dataframe to maintain original dataset intact
df = data.copy()


# In[365]:


#checking to make sure new dataframe details match original
df.shape


# In[366]:


df.head()


# #### Dropping customer ID number column "CLIENTNUM"
# * An identifier variable for bank customers, not suitable for model building purposes

# In[367]:


df.drop("CLIENTNUM", axis=1, inplace=True)


# In[368]:


df.info()


# In[369]:


# let's check for duplicate values in the data
df.duplicated().sum()


# In[370]:


#let's check the number of unique values in each column
df.nunique().sort_values(ascending=False)


# #### Observations
# 
# * We observe 3 unique values for Marital_Status, 4 for Card_Category, 6 for Total_Relationship_Count, Income_Category, Education_Level, and Dependent_count
# * 45 unique values for customer ages
# * Greater range of unique values for continuous features
# * No duplicate values found in the data

# In[371]:


# let's check for missing values in the data
round(df.isnull().sum() / df.isnull().count() * 100, 2)


# #### Observations
# 
# * 15% values missing from Education_Level feature and 7.4% missing from Marital_Status - these missing values will be treated later on

# #### Changing data type of "object" columns to "category" for processing efficiency purposes

# In[372]:


category_cols = ["Attrition_Flag"
                 , "Gender"
                 , "Education_Level"
                 , "Marital_Status"
                 , "Income_Category"
                 , "Card_Category"]


# In[373]:


for col in category_cols:
    df[col] = df[col].astype("category")


# In[374]:


df.info()


# #### Looking at descriptives for all columns

# In[375]:


# let's view the statistical summary of the numerical columns in the data
df.describe().T


# #### Observations
# 
# * Customer age in this dataset hovers around middle age with the mid 50 percentile of customer falling between the ages of 41 and 52
# * Distribution for Dependent_count looks to be slightly positively skewed with a mean of 2.35 and median of 2
# * The mean and median period of relationship customers have had with bank is about 36 months
# * Customers tend to hold approximately 4 products from the bank on average
# * The mean months inactive over the last 12 for customers is approximately 2 months
# * Number of contacts on average between bank and customer over the last 12 months is about 2.5
# * In terms of credit card limit, we observe a highly positively skewed distribution with likely presence of outliers at the higher end of the distribution. Mean credit card limit of 8,631.95, median of 4,549.00
# * In terms of the balance that carries over from one month to the next, Total_Revolving_Bal, we observe a slightly negatively skewed distribution, with median (1,276.00) greater than mean (1,162.81)
# * On average, customers have approximately 7,500 left on their limit to spend on their credit card - we also observe here a large discrepancy between median and max scores which indicates the presence of outliers in the distribution
# * In terms of the ratio of the total transaction amount in 4th quarter and the total transaction amount in 1st quarter, we observe a mean ratio of 0.76 and 50% of mid-range scores falling between 0.63 and 0.86
# * For total transaction amount over the previous 12 months we observe an average of 4,404.08 and median of 4,000 indicating a positive skewed distribution
# * For count of total transactions over the previos 12 months, 50% mid range of the distribution falls between 45 and 81 transactions
# * In terms of the ratio of the total transaction count in 4th quarter and the total transaction amount in 1st quarter, we observe a mean ratio of 0.71 and 50% of mid-range scores falling between 0.58 and 0.82
# * Average Utilization Ratio represents how much of the available credit the customer spent and here we observe a mean of 0.27 with is greater than the median of the distribution, 0.18. This might indicate the presence of high-end outliers - we will further examine the distribution to assess

# In[376]:


# let's view the statistical summary of the non-numerical columns in the data
df.describe(exclude=np.number).T


# #### Observations
# 
# * Attrition_Flag is our dependent variable, that is, the values of which we will attempt to predict in our models. We observe a total count of 10,127 values, the most frequent of which indicates customer falls in category "Existing Customer" (8,500 instances or 84% of the distribution) as opposed to "Attrited Customer"
# * For Gender, the most common among the bank's customers is female, with 5,358 female customers
# * Total count for Education_Level reflects missing values. Most frequent value being "Graduate" with 3,128 customers identifying as such
# * 3 marital status categories, the most frequent of which being married 4,687 - perhaps indicative of the large number of mid-age adults that make up the bank's customer base
# * Most frequent income category indicates that most of the bank's customers earn less than $40k/year
# * Most popular card category is Blue, held by 9,436 of the bank's customers

# In[377]:


# Printing the number of occurrences of each unique value, including null values, in each categorical column
for column in category_cols:
    print(df[column].value_counts(dropna=False))
    print("-" * 50)


# #### Observations
# 
# * As previously noted, missing values observed for Marital_Status and Education_Level
# * We also observe the non-sensical value "abc" in Income_Category, we will consider abc as missing values and will code them so below in order to be able to treat them as missing values later on

# ##### Recategorizing 'abc' value in Income_Category as NaN

# In[378]:


df['Income_Category'] = df['Income_Category'].apply(lambda x: np.nan if x == 'abc' else x)


# In[379]:


df['Income_Category'] = df['Income_Category'].astype("category")


# In[380]:


df['Income_Category'].value_counts(dropna=False)


# #### Observations
# 
# * Replacement of 1112 'abc' values for NaN completed successfully

# ## Exploratory Data Analysis

# ### Univariate Analysis

# In[381]:


# function to plot a boxplot and a histogram along the same scale.


def histogram_boxplot(data, feature, figsize=(12, 7), kde=False, bins=None):
    """
    Boxplot and histogram combined

    data: dataframe
    feature: dataframe column
    figsize: size of figure (default (12,7))
    kde: whether to the show density curve (default False)
    bins: number of bins for histogram (default None)
    """
    f2, (ax_box2, ax_hist2) = plt.subplots(
        nrows=2,  # Number of rows of the subplot grid= 2
        sharex=True,  # x-axis will be shared among all subplots
        gridspec_kw={"height_ratios": (0.25, 0.75)},
        figsize=figsize,
    )  # creating the 2 subplots
    sns.boxplot(
        data=data, x=feature, ax=ax_box2, showmeans=True, color="violet"
    )  # boxplot will be created and a star will indicate the mean value of the column
    sns.histplot(
        data=data, x=feature, kde=kde, ax=ax_hist2, bins=bins, palette="winter"
    ) if bins else sns.histplot(
        data=data, x=feature, kde=kde, ax=ax_hist2
    )  # For histogram
    ax_hist2.axvline(
        data[feature].mean(), color="green", linestyle="--"
    )  # Add mean to the histogram
    ax_hist2.axvline(
        data[feature].median(), color="black", linestyle="-"
    )  # Add median to the histogram


# #### Customer_Age

# In[382]:


histogram_boxplot(df, "Customer_Age", figsize=(12, 7), kde=False, bins=None)


# #### Observations
# 
# * Customer age in this dataset hovers around middle age with the mid 50 percentile of customer falling between the ages of 41 and 52
# 
# * There are some outliers on the right end of the boxplot but we will not treat them as some variation is always expected in real-world scenarios for age - in this case, the max value for age in dataset is 73 which is reasonable

# #### Dependent_count

# In[383]:


histogram_boxplot(df, "Dependent_count", figsize=(12, 7), kde=False, bins=None)


# #### Observations
# 
# * Distribution for Dependent_count looks to be slightly positively skewed with a mean of 2.35 and median of 2
# * Outliers not observed

# #### Months_on_book

# In[384]:


histogram_boxplot(df, "Months_on_book", figsize=(12, 7), kde=False, bins=None)


# #### Observations
# 
# * The mean and median period of relationship customers have had with bank is about 36 months
# * Most customers fall in the 36-month mark
# * We observe the presence of outliers which reflect expected real-world fluctuations in terms of period of relationship customers have with bank. Non-sensical or out of ordinary values not observed in this distribution

# #### Total_Relationship_Count

# In[385]:


histogram_boxplot(df, "Total_Relationship_Count", figsize=(12, 7), kde=False, bins=None)


# #### Observations
# 
# * Customers tend to hold approximately 4 products from the bank on average - that is what the Total_Relationship_Count feature reflects
# 
# * We do not observe presence of outliers in the distribution

# #### Months_Inactive_12_mon

# In[386]:


histogram_boxplot(df, "Months_Inactive_12_mon", figsize=(12, 7), kde=False, bins=None)


# #### Observations
# 
# * The mean months inactive over the last 12 for customers is approximately 2 months
# * Although we observe presence of outliers, values in the distribution appear to reflect variation expected in a real-world scenario

# #### Contacts_Count_12_mon

# In[387]:


histogram_boxplot(df, "Contacts_Count_12_mon", figsize=(12, 7), kde=False, bins=None)


# #### Observations
# 
# * Number of contacts on average between bank and customer over the last 12 months is about 2.5
# * Although we observe presence of outliers, values in the distribution appear to reflect variation expected in a real-world scenario

# #### Credit_Limit

# In[388]:


histogram_boxplot(df, "Credit_Limit", figsize=(12, 7), kde=False, bins=None)


# #### Observations
# 
# * In terms of credit card limit, we observe a highly positively skewed distribution with likely presence of outliers at the higher end of the distribution. Mean credit card limit of 8,631.95, median of 4,549.00
# * We can see that there are some extreme observations in the variable that can be considered as outliers as they very far from the rest of the values. 
# * There are about 500 customers with credit limit at the high end creating an outlier cluster of values. Dropping these would mean a significant loss of data and capping would not resolve the issue as we would still end up with a cluster of values at the high end of the distribution. As such we will not remove or cap variable

# #### Total_Revolving_Bal

# In[389]:


histogram_boxplot(df, "Total_Revolving_Bal", figsize=(12, 7), kde=False, bins=None)


# #### Observations
# 
# * In terms of the balance that carries over from one month to the next, Total_Revolving_Bal, we observe a slightly negatively skewed distribution, with median (1,276.00) greater than mean (1,162.81)
# * We do not observe the presence of outliers in this distribution

# #### Avg_Open_To_Buy

# In[390]:


histogram_boxplot(df, "Avg_Open_To_Buy", figsize=(12, 7), kde=False, bins=None)


# #### Observations
# 
# * On average, customers have approximately 7,500 left on their limit to spend on their credit card - we also observe here a large discrepancy between median and max scores which indicates the presence of outliers in the distribution
# * We observe the presence of outliers at the higher end of the distribution reflecting cluster of customers with higher credit limits - these customers would naturally have more money left over to spend in credit card accounts
# * Dropping outliers would result in losing valuable data and capping would not resolve cluster issue leading to right-skewed distribution

# #### Total_Amt_Chng_Q4_Q1

# In[391]:


histogram_boxplot(df, "Total_Amt_Chng_Q4_Q1", figsize=(12, 7), kde=False, bins=None)


# #### Observations
# 
# * In terms of the ratio of the total transaction amount in 4th quarter and the total transaction amount in 1st quarter, we observe a mean ratio of 0.76 and 50% of mid-range scores falling between 0.63 and 0.86
# * We observe outliers in both the lower and higher ends of the distribution
# * Lower end outliers represent what we would expect - for some customers ratio of total transaction amount would correctly be 0
# * Higher end outliers could be capped at next highest value to help narrow distribution and handle extreme values

# In[392]:


df[df["Total_Amt_Chng_Q4_Q1"] > 2.5]["Total_Amt_Chng_Q4_Q1"].nunique()


# * Let's cap value at 2.5 to capture the most extreme values

# In[393]:


# Capping values for Total_Amt_Chng_Q4_Q1 at 2.5
df["Total_Amt_Chng_Q4_Q1"].clip(upper=2.5, inplace=True)


# #### Total_Trans_Amt

# In[394]:


histogram_boxplot(df, "Total_Trans_Amt", figsize=(12, 7), kde=False, bins=None)


# #### Observations
# 
# * For total transaction amount over the previous 12 months we observe an average of 4,404.08 and median of 4,000 
# * Distribution is positively skewed with fluctuations clustering around $2000, $4000, $7800, and $15000
# * Although distribution is skewed to the right, values observed are not unexpected in terms of real world financial transaction amounts over a 12-month period

# #### Total_Trans_Ct

# In[395]:


histogram_boxplot(df, "Total_Trans_Ct", figsize=(12, 7), kde=False, bins=None)


# #### Observations
# 
# * For count of total transactions over the previos 12 months, 50% mid range of the distribution falls between 45 and 81 transactions
# * We observe two extreme high values which can be capped to bring them in allignment with the rest of the distribution - we will cap at Q3 + (1.5 * IQR)

# In[396]:


Q1 = df["Total_Trans_Ct"].quantile(0.25)  # 25th quantile
Q3 = df["Total_Trans_Ct"].quantile(0.75)  # 75th quantile
IQR = Q3 - Q1
Upper_Whisker = Q3 + 1.5 * IQR


# In[397]:


# Capping values for Total_Trans_Ct
df["Total_Trans_Ct"].clip(upper=Upper_Whisker, inplace=True)


# #### Total_Ct_Chng_Q4_Q1

# In[398]:


histogram_boxplot(df, "Total_Ct_Chng_Q4_Q1", figsize=(12, 7), kde=False, bins=None)


# #### Observations
# 
# * In terms of the ratio of the total transaction count in 4th quarter and the total transaction amount in 1st quarter, we observe a mean ratio of 0.71 and 50% of mid-range scores falling between 0.58 and 0.82
# * Presence of outliers at the low and higher end of the distribution though at the higher end we observe more extreme values - we can cap those to help narrow the distribution

# In[399]:


df[df["Total_Ct_Chng_Q4_Q1"] > 2.5]["Total_Ct_Chng_Q4_Q1"].nunique()


# * Let's cap value at 2.5 to capture the most extreme values

# In[400]:


# Capping values for Total_Ct_Chng_Q4_Q1 at 2.5
df["Total_Ct_Chng_Q4_Q1"].clip(upper=2.5, inplace=True)


# #### Avg_Utilization_Ratio

# In[401]:


histogram_boxplot(df, "Avg_Utilization_Ratio", figsize=(12, 7), kde=False, bins=None)


# #### Observations
# 
# * Average Utilization Ratio represents how much of the available credit the customer spent and here we observe a mean of 0.27 with is greater than the median of the distribution, 0.18. 
# * Although we observe a long right tail in the distribution, no values falling outside of Q3 + 1.5*IQR observed

# In[402]:


# function to create labeled barplots


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
        plt.figure(figsize=(count + 1, 5))
    else:
        plt.figure(figsize=(n + 1, 5))

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


# In[427]:


def value_perc_dist(col):
    print("{}".format((df[col].value_counts()/df[col].count())*100))


# #### Attrition_Flag

# In[428]:


value_perc_dist("Attrition_Flag")


# In[403]:


labeled_barplot(df, "Attrition_Flag")


# #### Observations
# 
# * Attrition_Flag is our dependent variable - the values of which we will attempt to predict
# * "Attrited Customer" makes up about 16% of customers in this dataset

# #### Gender

# In[429]:


value_perc_dist("Gender")


# In[39]:


labeled_barplot(df, "Gender")


# #### Observations
# 
# * Most of the customers are female with close to 53% of total count

# #### Education_Level

# In[430]:


value_perc_dist("Education_Level")


# In[40]:


labeled_barplot(df, "Education_Level")


# #### Observations
# 
# * 36% of customers have graduate level education
# * The next largest group are high school graduates with 23% followed by "Uneducated" with 17%

# #### Marital_Status

# In[431]:


value_perc_dist("Marital_Status")


# In[41]:


labeled_barplot(df, "Marital_Status")


# #### Observations
# 
# * About half of the customers in the dataset are married
# * Single makes up the second largest group, 42% of customers

# #### Income_Category

# In[432]:


value_perc_dist("Income_Category")


# In[42]:


labeled_barplot(df, "Income_Category")


# #### Observations
# 
# * About 40% of customers fall in the Less than $40k income bracket
# * The next largest group being customers with incomes between 40 and 60k

# #### Card_Category

# In[433]:


value_perc_dist("Card_Category")


# In[43]:


labeled_barplot(df, "Card_Category")


# #### Observations
# 
# * 93% of customers hold a Blue card, Silver is the second largest category with 5% of customers

# ### Bivariate Analysis

# ###### Encoding target variable "Attrition_Flag" to 0 for "Exixisting Customer" and 1 for "Attrited Customer", for the purpose of analysis.

# In[434]:


df["Attrition_Flag"].replace("Existing Customer", 0, inplace=True)
df["Attrition_Flag"].replace("Attrited Customer", 1, inplace=True)


# In[435]:


df["Attrition_Flag"] = df["Attrition_Flag"].astype(int)


# In[436]:


plt.figure(figsize=(15, 7))
sns.heatmap(df.corr(), annot=True, vmin=-1, vmax=1, fmt=".2f", cmap="Spectral")
plt.show()


# #### Observations
# 
# Positive correlations with customer attrition:
# * Attrition_Flag and Contacts_Count_12_mon: we observe a slight positive correlation here (0.20) between customer attrition and number of contacts between the customer and bank over the previous 12 months
# 
# * Attrition_Flag and Months_Inactive_12_mon: we observe a slight positive correlation here (0.15) between customer attrition and customer inactivity over the previous 12 months
# 
# 
# Negative correlations with customer attrition:
# * Attrition_Flag and Total_Trans_Ct: we observe a negative correlation (the strongest correlation with the target variable) of -0.37 between total transaction count (i.e. total number of transactions) over the last 12 months and customer attrition.
# 
# * Attrition_Flag and Total_Ct_Chng_Q4_Q1: we observe a negative correlation, -0.30, between customer attrition and ratio of the total transaction count in 4th quarter to the total transaction count in 1st quarter
# 
# * Attrition_Flag and Total_Revolving_Bal: we observe a negative correlation of -0.26 between balance that carries over from month to month and customer attrition.
# 
# * Attrition_Flag and Avg_Utilization_Ratio: we observe a negative correlation, -0.18, between how much of the available credit the customer has spent and customer attrition.
# 
# * Attrition_Flag and Total_Trans_Amt: we observe a negative correlation, -0.17, between total transaction amount (last 12 months) and customer attrition.
# 
# * Attrition_Flag and Total_Relationship_Count: we observe a negative correlation, -0.15, between the total number of bank products used by customer and customer attrition.
# 
# * Attrition_Flag and Total_Amt_Chng_Q4_Q1: we observe a slightly negative correlation, -0.13, between customer attrition and ratio of the total transaction amount in 4th quarter to the total transaction amount in 1st quarter
# 
# 
# 

# In[437]:


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
    tab.plot(kind="bar", stacked=True, figsize=(count + 1, 5))
    plt.legend(
        loc="lower left",
        frameon=False,
    )
    plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
    plt.show()


# ###### Encoding target variable back to "Attrited Customer" (1) and "Exixisting Customer" (0), for the purpose of analysis.

# In[438]:


df["Attrition_Flag"].replace(0, "Existing Customer", inplace=True)
df["Attrition_Flag"].replace(1, "Attrited Customer", inplace=True)


# In[439]:


df["Attrition_Flag"] = df["Attrition_Flag"].astype("category")


# In[440]:


df.info()


# #### Attrition_Flag vs Gender

# In[441]:


stacked_barplot(df, "Gender", "Attrition_Flag")


# #### Observations
# 
# * We observe a slightly greater proportion of attrited customers among female than male customers

# #### Attrition_Flag vs Dependent_count

# In[442]:


stacked_barplot(df, "Dependent_count", "Attrition_Flag")


# #### Observations
# 
# * Among attrited customers, those with a dependent count of 5 show up with the lowest attrition incidence, followed by 0 and then 4

# #### Attrition_Flag vs Education_Level

# In[443]:


stacked_barplot(df, "Education_Level", "Attrition_Flag")


# #### Observations
# 
# * We observe higher proportion of attrited customers among those with doctorate and post-doctorate degrees

# #### Attrition_Flag vs Marital_Status

# In[444]:


stacked_barplot(df, "Marital_Status", "Attrition_Flag")


# #### Observations
# 
# * We observe just a slightly lower proportion of attrited customers among customers who are married

# #### Attrition_Flag vs Income_Category

# In[445]:


stacked_barplot(df, "Income_Category", "Attrition_Flag")


# #### Observations
# 
# * Attrited customers show up slightly more frequently among customers in the 120k+ income bracket and in the less than 40k bracket

# #### Attrition_Flag vs Card_Category

# In[446]:


stacked_barplot(df, "Card_Category", "Attrition_Flag")


# #### Observations
# 
# * In total, there are only 20 Platinum customers in this dataset so it makes it hard to draw meaningful observations about this group
# * Attrited customers appear in slightly greater proportion among Gold card customers in comparison to Blue and Silver

# #### Attrition_Flag vs Total_Relationship_Count

# In[447]:


stacked_barplot(df, "Total_Relationship_Count", "Attrition_Flag")


# #### Observations
# 
# * Total_Relationship_Count and Customer Attrition: We observe that attrition happens in lower proportions among customers who hold more bank products

# #### Attrition_Flag vs Months_Inactive_12_mon

# In[448]:


stacked_barplot(df, "Months_Inactive_12_mon", "Attrition_Flag")


# #### Observations
# 
# * Nearly half of customers who opened an account at the bank attrited less than a month later
# * Once customers stay for at least a month with the bank then attrition looks more likely at the 4 and 3 month mark for account inactivity

# #### Attrition_Flag vs Contacts_Count_12_mon

# In[449]:


stacked_barplot(df, "Contacts_Count_12_mon", "Attrition_Flag")


# #### Observations
# 
# * Customers in contact with the bank 6 times attrited 100%
# * Pattern of higher number of contacts, higher attrition follows what it looks like a linear pattern, increasing attrition as number of contacts increases (or vice-versa)

# #### Attrition_Flag vs Customer_Age

# In[450]:


sns.set(rc={"figure.figsize": (10, 7)})
sns.boxplot(x="Attrition_Flag", y="Customer_Age", data=df, orient="vertical")


# #### Observations
# 
# * We observe very similar distributions between attrited and existing customers in terms of age
# * The median age for attrited customers being a little greater than for existing customers indicating a slight bent towards older customers in the attrited group

# #### Attrition_Flag vs Months_on_book

# In[451]:


sns.set(rc={"figure.figsize": (10, 7)})
sns.boxplot(x="Attrition_Flag", y="Months_on_book", data=df, orient="vertical")


# #### Observations
# 
# * Distributions for Attrited and Existing customers based on period of relationship with the bank look similar though we observe a greater number of outliers in the lower end for Attrited Customers suggesting that a slightly greater left-hand skew to the distribution and perhaps more attrition among newer bank customers

# #### Attrition_Flag vs Credit_Limit

# In[452]:


sns.set(rc={"figure.figsize": (10, 7)})
sns.boxplot(x="Attrition_Flag", y="Credit_Limit", data=df, orient="vertical")


# #### Observations
# 
# * The range of credit limit for attrited customers is lower overall than for existing customers, with lower median, 75% percentile and mid 50% of credit limit distribution

# #### Attrition_Flag vs Total_Revolving_Bal

# In[453]:


sns.set(rc={"figure.figsize": (10, 7)})
sns.boxplot(x="Attrition_Flag", y="Total_Revolving_Bal", data=df, orient="vertical")


# #### Observations
# 
# * Revolving balance for Existing Customers are higher than for attrited customers - meaning that the balance that carries over from one month to the next is lower (between 0 and approximately 1,400 for the mid 50% range) for attrited customers than for existing customers (between 800 and 1,800 for the mid 50% range)

# #### Attrition_Flag vs Avg_Open_To_Buy

# In[454]:


sns.set(rc={"figure.figsize": (10, 7)})
sns.boxplot(x="Attrition_Flag", y="Avg_Open_To_Buy", data=df, orient="vertical")


# #### Observations
# 
# * Distributions look similar though attrited customers' range of amount left on credit card to use is slightly lower and narrower in comparison to existing customers

# #### Attrition_Flag vs Total_Amt_Chng_Q4_Q1

# In[455]:


sns.set(rc={"figure.figsize": (10, 7)})
sns.boxplot(x="Attrition_Flag", y="Total_Amt_Chng_Q4_Q1", data=df, orient="vertical")


# #### Observations
# 
# * We observe the presence of a more pronounced right side skew in the distribution of the 4th quarter to 1st quarter ratio of total transaction amounts - meaning we observe a long tail of outlier values on the higher end of the distribution, indicative of existing customers have greater levels of transaction amounts in 4th vs 1st quarter than attrited customers

# #### Attrition_Flag vs Total_Trans_Amt

# In[456]:


sns.set(rc={"figure.figsize": (10, 7)})
sns.boxplot(x="Attrition_Flag", y="Total_Trans_Amt", data=df, orient="vertical")


# #### Observations
# 
# * Total transaction amounts over the last 12 months are generally greater for existing customers than for attrited customers

# #### Attrition_Flag vs Total_Trans_Ct

# In[457]:


sns.set(rc={"figure.figsize": (10, 7)})
sns.boxplot(x="Attrition_Flag", y="Total_Trans_Ct", data=df, orient="vertical")


# #### Observations
# 
# * The total number of transactions over the last 12 months is generally greater for existing customers than for attrited customers

# #### Attrition_Flag vs Total_Ct_Chng_Q4_Q1

# In[458]:


sns.set(rc={"figure.figsize": (10, 7)})
sns.boxplot(x="Attrition_Flag", y="Total_Ct_Chng_Q4_Q1", data=df, orient="vertical")


# #### Observations
# 
# * The ratio of the total number of transactions in 4th versus 1st quarter is generally higher for existing customers than for attrited customers

# #### Attrition_Flag vs Avg_Utilization_Ratio

# In[459]:


sns.set(rc={"figure.figsize": (10, 7)})
sns.boxplot(x="Attrition_Flag", y="Avg_Utilization_Ratio", data=df, orient="vertical")


# #### Observations
# 
# * Though with some higher value outliers, the amount of available credit that the customer has spent is generally lower for attrited than existing customers

# # <a id='link1'>Summary of EDA</a>
# **Data Description:**
# 
# * 10127 rows of data with 19 features/independent variables (CLIENTNUM is an ID variable so it was dropped from the data)
# * Dependent variable is **"Attrition_Flag"** indicating whether a customer is an Attrited Customer or an Existing Customer.
# * We observed 2268 missing values in the dataset: 1519 for Education_Level, 749 for Marital_Status
# * We also observed 1112 values labeled "abc" under the Income_Category variable. These were converted to NaN (missing) values
# 
# 
# **Data Cleaning:**
# 
# * CLIENTNUM is an ID variable so it was dropped from the data.
# * As mentioned above, all values labeled "abc" in the Income_Category column were converted to NaN
# * Columns categorized as data type "object" (Attrition_Flag, Gender, Education_Level, Marital_Status, Income_Category, Card_Category) were changed to data type "category"
# * Outliers observed for Total_Amt_Chng_Q4_Q1 and Total_Ct_Chng_Q4_Q1 were capped at 2.5 - any ratio values greater than cap were converted to 2.5
# * Outliers observed for Total_Trans_Ct were capped at the upper whisker (Q3 + 1.5 IQR) value of the distribution
# 
# 
# 
# **Observations from EDA:**
# 
# ###### Univariate Analysis
# * `Gender`: Most of the customers are female with close to 53% of total count
# * `Education_Level`: Most frequent value being "Graduate" with 3,128 customers (36%) identifying as such
# * `Marital_Status`: 3 marital status categories, the most frequent of which being married 4,687 (50% of customers)
# * `Income_Category`: Most frequent income category indicates that most of the bank's customers earn less than 40K per year (40% of customers)
# * `Card_Category`: Most popular card category is Blue, held by 93% of customers
# * `Customer_Age`: Customer age in this dataset hovers around middle age with the mid 50 percentile of customer falling between the ages of 41 and 52
# * `Dependent_count`: Distribution for Dependent_count looks to be slightly positively skewed with a mean of 2.35 and median of 2
# * `Months_on_book`: The mean and median period of relationship customers have had with bank is about 36 months
# * `Total_Relationship_Count`: Customers tend to hold approximately 4 products from the bank on average
# * `Months_Inactive_12_mon`: The mean months inactive over the last 12 for customers is approximately 2 months
# * `Contacts_Count_12_mon`: Number of contacts on average between bank and customer over the last 12 months is about 2.5
# * `Credit_Limit`: In terms of credit card limit, we observe a highly positively skewed distribution with likely presence of outliers at the higher end of the distribution. Mean credit card limit of 8,631.95, median of 4,549.00
# * `Total_Revolving_Bal`: In terms of the balance that carries over from one month to the next, Total_Revolving_Bal, we observe a slightly negatively skewed distribution, with median (1,276.00) greater than mean (1,162.81)
# * `Avg_Open_To_Buy`: On average, customers have approximately 7,500 left on their limit to spend on their credit card
# * `Total_Amt_Chng_Q4_Q1`: In terms of the ratio of the total transaction amount in 4th quarter and the total transaction amount in 1st quarter, we observe a mean ratio of 0.76 and 50% of mid-range scores falling between 0.63 and 0.86
# * `Total_Trans_Amt`: For total transaction amount over the previous 12 months we observe an average of 4,404.08 and median of 4,000 indicating a positive skewed distribution
# * `Total_Trans_Ct`: For count of total transactions over the previous 12 months, 50% mid range of the distribution falls between 45 and 81 transactions
# * `Total_Ct_Chng_Q4_Q1`: In terms of the ratio of the total transaction count in 4th quarter and the total transaction amount in 1st quarter, we observe a mean ratio of 0.71 and 50% of mid-range scores falling between 0.58 and 0.82
# * `Avg_Utilization_Ratio`: Average Utilization Ratio represents how much of the available credit the customer spent and here we observe a mean of 0.27 with is greater than the median of the distribution, 0.18. 
# * `Attrition_Flag`(target variable): We observe a total count of 10,127 values, the most frequent of which indicates customer falls in category "Existing Customer" (8,500 instances or 84% of the distribution) as opposed to "Attrited Customer"
# 
# 
# ###### Bivariate Analysis
# 
# Examining relationships between features and target variable
# 
# **Correlations** between target variable and numeric features:
# `Positive correlations with customer attrition`:
# * Attrition_Flag and Contacts_Count_12_mon: we observe a slight positive correlation here (0.20) between customer attrition and number of contacts between the customer and bank over the previous 12 months
# 
# * Attrition_Flag and Months_Inactive_12_mon: we observe a slight positive correlation here (0.15) between customer attrition and customer inactivity over the previous 12 months
# 
# 
# `Negative correlations with customer attrition`:
# * Attrition_Flag and Total_Trans_Ct: we observe a negative correlation (the strongest correlation with the target variable) of -0.37 between total transaction count (i.e. total number of transactions) over the last 12 months and customer attrition.
# 
# * Attrition_Flag and Total_Ct_Chng_Q4_Q1: we observe a negative correlation, -0.30, between customer attrition and ratio of the total transaction count in 4th quarter to the total transaction count in 1st quarter
# 
# * Attrition_Flag and Total_Revolving_Bal: we observe a negative correlation of -0.26 between balance that carries over from month to month and customer attrition.
# 
# * Attrition_Flag and Avg_Utilization_Ratio: we observe a negative correlation, -0.18, between how much of the available credit the customer has spent and customer attrition.
# 
# * Attrition_Flag and Total_Trans_Amt: we observe a negative correlation, -0.17, between total transaction amount (last 12 months) and customer attrition.
# 
# * Attrition_Flag and Total_Relationship_Count: we observe a negative correlation, -0.15, between the total number of bank products used by customer and customer attrition.
# 
# * Attrition_Flag and Total_Amt_Chng_Q4_Q1: we observe a slightly negative correlation, -0.13, between customer attrition and ratio of the total transaction amount in 4th quarter to the total transaction amount in 1st quarter
# 
# 
# **Relationships between each feature and target variable**
# * `Gender and Target Variable`: We observe a slightly greater proportion of attrited customers among female than male customers
# * `Dependent_count and Target Variable`: Among attrited customers, those with a dependent count of 5 show up with the lowest attrition incidence, followed by 0 and then 4
# * `Education_Level and Target Variable`: We observe higher proportion of attrited customers among those with doctorate and post-doctorate degrees
# * `Marital_Status and Target Variable`: We observe just a slightly lower proportion of attrited customers among customers who are married
# * `Card_Category and Target Variable`: In total, there are only 20 Platinum customers in this dataset so it makes it hard to draw meaningful observations about this group. Attrited customers appear in slightly greater proportion among Gold card customers in comparison to Blue and Silver
# * `Income_Category and Target Variable`: Attrited customers show up slightly more frequently among customers in the 120k+ income bracket and in the less than 40k bracket
# * `Total_Relationship_Count and Target Variable`: Total_Relationship_Count and Customer Attrition: We observe that attrition happens in lower proportions among customers who hold more bank products
# * `Months_Inactive_12_mon and Target Variable`: Nearly half of customers who opened an account at the bank attrited less than a month later. Once customers stay for at least a month with the bank then attrition looks more likely at the 4 and 3 month mark for account inactivity
# * `Contacts_Count_12_mon and Target Variable`: Customers in contact with the bank 6 times attrited 100%. Pattern of higher number of contacts, higher attrition follows what it looks like a linear pattern, increasing attrition as number of contacts increases (or vice-versa)
# * `Customer_Age and Target Variable`: We observe very similar distributions between attrited and existing customers in terms of age. The median age for attrited customers being a little greater than for existing customers indicating a slight bent towards older customers in the attrited group
# * `Months_on_book and Target Variable`: Distributions for Attrited and Existing customers based on period of relationship with the bank look similar though we observe a greater number of outliers in the lower end for Attrited Customers suggesting that a slightly greater left-hand skew to the distribution and perhaps more attrition among newer bank customers
# * `Credit_Limit and Target Variable`: The range of credit limit for attrited customers is lower overall than for existing customers, with lower median, 75% percentile and mid 50% of credit limit distribution
# * `Total_Revolving_Bal and Target Variable`: Revolving balance for Existing Customers are higher than for attrited customers - meaning that the balance that carries over from one month to the next is lower (between 0 and approximately 1,400 for the mid 50% range) for attrited customers than for existing customers (between 800 and 1,800 for the mid 50% range)
# * `Avg_Open_To_Buy and Target Variable`: Distributions look similar though attrited customers' range of amount left on credit card to use is slightly lower and narrower in comparison to existing customers
# * `Total_Amt_Chng_Q4_Q1 and Target Variable`: We observe the presence of a more pronounced right side skew in the distribution of the 4th quarter to 1st quarter ratio of total transaction amounts - meaning we observe a long tail of outlier values on the higher end of the distribution, indicative of existing customers have greater levels of transaction amounts in 4th vs 1st quarter than attrited customers
# * `Total_Trans_Amt and Target Variable`: Total transaction amounts over the last 12 months are generally greater for existing customers than for attrited customers
# * `Total_Trans_Ct and Target Variable`: The total number of transactions over the last 12 months is generally greater for existing customers than for attrited customers
# * `Total_Ct_Chng_Q4_Q1 and Target Variable`: The ratio of the total number of transactions in 4th versus 1st quarter is generally higher for existing customers than for attrited customers
# * `Avg_Utilization_Ratio and Target Variable`: Though with some higher value outliers, the amount of available credit that the customer has spent is generally lower for attrited than existing customers
# 

# ## Data Preparation for Modeling

# ### Copying dataset to ensure integrity of values prior to missing values treatment and avoid data leaks later on

# In[467]:


#Copying df dataset to run model training, validation and testing on data1
data1 = df.copy()


# In[470]:


X = data1.drop(["Attrition_Flag"], axis=1)
y = data1["Attrition_Flag"].apply(lambda x: 1 if x == "Attrited Customer" else 0)


# In[471]:


# Splitting data into training, validation and test set:
# first we split data into 2 parts, say temporary and test

X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1, stratify=y
)

# then we split the temporary set into train and validation

X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.25, random_state=1, stratify=y_temp
)
print(X_train.shape, X_val.shape, X_test.shape)


# In[472]:


print("Number of rows in train data =", X_train.shape[0])
print("Number of rows in validation data =", X_val.shape[0])
print("Number of rows in test data =", X_test.shape[0])


# ### Missing Value Treatment

# ### Preparation for Missing Value Treatment
# 
# * We will use KNN imputer to impute missing values.
# * `KNNImputer`: Each sample's missing values are imputed by looking at the n_neighbors nearest neighbors found in the training set. Default value for n_neighbors=5.
# * KNN imputer replaces missing values using the average of k nearest non-missing feature values.
# * Nearest points are found based on euclidean distance.
# 
# 
# **The values obtained might not be integer always which is not be the best way to impute categorical values**
# - To take care of that we will round off the obtained values to nearest integer value

# In[460]:


data1.isnull().sum()


# In[461]:


imputer = KNNImputer(n_neighbors=5)


# In[462]:


# defining a list with names of columns that will be used for imputation
reqd_col_for_impute = [
    "Education_Level",
    "Marital_Status",
    "Income_Category"
]


# In[468]:


# we need to pass numerical values for each categorical column for KNN imputation so we will label encode them
Education_Level = {"Uneducated": 0, "High School": 1, "College": 2
                   , "Graduate": 3, "Post-Graduate": 4, "Doctorate": 5}
data1["Education_Level"] = data1["Education_Level"].map(Education_Level)

Marital_Status = {"Single": 0, "Married": 1, "Divorced": 2}
data1["Marital_Status"] = data1["Marital_Status"].map(Marital_Status)

Income_Category = {
    "Less than $40K": 0,
    "$40K - $60K": 1,
    "$60K - $80K": 2,
    "$80K - $120K": 3,
    "$120K +": 4,
}
data1["Income_Category"] = data1["Income_Category"].map(Income_Category)


# **As defined above, we will use KNN imputer to impute missing values for "Education_Level","Marital_Status",
# "Income_Category"**

# In[473]:


# Fit and transform the train data
X_train[reqd_col_for_impute] = imputer.fit_transform(X_train[reqd_col_for_impute])

# Transform the train data
X_val[reqd_col_for_impute] = imputer.fit_transform(X_val[reqd_col_for_impute])

# Transform the test data
X_test[reqd_col_for_impute] = imputer.transform(X_test[reqd_col_for_impute])


# In[474]:


# Checking that no column has missing values in train, validation or test sets
print(X_train.isna().sum())
print("-" * 30)
print(X_val.isna().sum())
print("-" * 30)
print(X_test.isna().sum())


# ##### Observations
# 
# * All missing values have been treated.
# * Let's inverse map the encoded values.

# In[475]:


## Function to inverse the encoding
def inverse_mapping(x, y):
    inv_dict = {v: k for k, v in x.items()}
    X_train[y] = np.round(X_train[y]).map(inv_dict).astype("category")
    X_val[y] = np.round(X_val[y]).map(inv_dict).astype("category")
    X_test[y] = np.round(X_test[y]).map(inv_dict).astype("category")


# In[476]:


inverse_mapping(Education_Level, "Education_Level")
inverse_mapping(Marital_Status, "Marital_Status")
inverse_mapping(Income_Category, "Income_Category")


# #### Checking inverse mapped values/categories for train, validation, and test sets

# In[477]:


cols = X_train.select_dtypes(include=["object", "category"])
for i in cols.columns:
    print(X_train[i].value_counts())
    print("*" * 30)


# In[478]:


cols = X_val.select_dtypes(include=["object", "category"])
for i in cols.columns:
    print(X_val[i].value_counts())
    print("*" * 30)


# In[479]:


cols = X_test.select_dtypes(include=["object", "category"])
for i in cols.columns:
    print(X_test[i].value_counts())
    print("*" * 30)


# * Inverse mapping returned original labels.

# ### Creating Dummy Variables

# In[480]:


X_train = pd.get_dummies(X_train, drop_first=True)
X_val = pd.get_dummies(X_val, drop_first=True)
X_test = pd.get_dummies(X_test, drop_first=True)
print(X_train.shape, X_val.shape, X_test.shape)


# * After encoding there are 29 columns.

# ## Building the model

# ## Model evaluation criterion
# 
# 
# ### Goal of the model
# 
# * To recap, the goal is to develop a model to help classify customers more likely to leave Thera bank's credit cards services. Losing customers means financial losses for the bank, so the bank wants data science team to analyze data from customers to help identify those more likely to leave and reason for same – so that bank could improve upon those areas.
# 
# 
# For the predictive model it is important to remember the following in terms of target variable classification: 
# 
# - True Positive: occurs when model classifies customer correctly as "attrited customer"
# - False Positive: occurs when model classifies customer incorrectly as "existing customer", predicting customer would not close account when in fact account was closed
# - True Negative: occurs when model classifies customer correctly as "existing customer"
# - False Negative: occurs when model classifies customer incorrectly as "attrited customer", predicting customer would leave bank when in fact customer account was not closed
# 
# ###### Precision:
# 
# - Of customers predicted by model to close account what proportion actually did?
# - True positive / (True positive + False positive)
# 
# 
# ###### Recall: 
# - Of customers who actually closed account, what proportion were actually correctly predicted as attrited customers?
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
# For example, 94% precision would mean that almost all customers predicted as attrited customers did in fact close account and 97% recall would mean almost all customers who closed accounts were correctly predicted to do so.
# 
# If the model performed with 95% precision but 50% recall this would mean that when it identifies someone as an attrited customer it is largely correct, but it predicts as existing customers half of those customers who actually ended up closing their account.
# 
# 
# ### Precision, Recall, F1 scores
# 
# Within any one model, we can decide to emphasize precision, recall or a balance of the two which is provided by F1 scores - the decision comes down to context. 
# 
# 
# ### The model can make wrong predictions as:
# 1. Predicting customer will close account when customer will not.
# 2. Predicting customer will not close account but customer in fact will.
# 
# 
# ### Which case is more important? 
# 
# 1. The bank is interested in understanding what motivates customers to leave and therefore the model should emphasize ability to predict correctly customers who actually attrited - this means we should prioritize performance on recall, i.e. minimizing false negatives.
# 
# ### How to reduce this loss i.e need to reduce False Negatives?
# * Bank would want Recall to be maximized, the greater the Recall lesser the chances of false negatives.

# In[481]:


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


# In[482]:


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


# **Let's start by building different models using KFold and cross_val_score and tune the best models using GridSearchCV and RandomizedSearchCV**
# 
# - `Stratified K-Folds cross-validation` provides dataset indices to split data into train/validation sets. Split dataset into k consecutive folds (without shuffling by default) keeping the distribution of both classes in each fold the same as the target variable. Each fold is then used once as validation while the k - 1 remaining folds form the training set.

# In[485]:


models = []  # Empty list to store all the models

# Appending models into the list
models.append(("LR", LogisticRegression(random_state=1)))
models.append(("Bagging", BaggingClassifier(random_state=1)))
models.append(("Random forest", RandomForestClassifier(random_state=1)))
models.append(("GBM", GradientBoostingClassifier(random_state=1)))
models.append(("Adaboost", AdaBoostClassifier(random_state=1)))
models.append(("Xgboost", XGBClassifier(random_state=1, eval_metric="logloss")))
models.append(("dtree", DecisionTreeClassifier(random_state=1)))

results = []  # Empty list to store all model's CV scores
names = []  # Empty list to store name of the models
score = []
# loop through all models to get the mean cross validated score
print("\n" "Cross-Validation Performance:" "\n")
for name, model in models:
    scoring = "recall"
    kfold = StratifiedKFold(
        n_splits=5, shuffle=True, random_state=1
    )  # Setting number of splits equal to 5
    cv_result = cross_val_score(
        estimator=model, X=X_train, y=y_train, scoring=scoring, cv=kfold
    )
    results.append(cv_result)
    names.append(name)
    print("{}: {}".format(name, cv_result.mean() * 100))

print("\n" "Validation Performance:" "\n")

for name, model in models:
    model.fit(X_train, y_train)
    scores = recall_score(y_val, model.predict(X_val))
    score.append(scores)
    print("{}: {}".format(name, scores * 100))


# In[486]:


# Plotting boxplots for CV scores of all models defined above
fig = plt.figure()

fig.suptitle("Algorithm Comparison")
ax = fig.add_subplot(111)

plt.boxplot(results)
ax.set_xticklabels(names, rotation=90)

plt.show()


# #### Observations
# 
# * XGBoost and GBM are delivering the highest recall scores in cross-validation and validation performance is also good
# * Next we will apply oversampling and undersampling methods and observe model performance to further assess which models work best at predicting customer attrition

# ### Oversampling train data using SMOTE

# In[487]:


print("Before UpSampling, counts of label 'Yes': {}".format(sum(y_train == 1)))
print("Before UpSampling, counts of label 'No': {} \n".format(sum(y_train == 0)))

sm = SMOTE(
    sampling_strategy=1, k_neighbors=5, random_state=1
)  # Synthetic Minority Over Sampling Technique
X_train_over, y_train_over = sm.fit_resample(X_train, y_train)


print("After UpSampling, counts of label 'Yes': {}".format(sum(y_train_over == 1)))
print("After UpSampling, counts of label 'No': {} \n".format(sum(y_train_over == 0)))


print("After UpSampling, the shape of train_X: {}".format(X_train_over.shape))
print("After UpSampling, the shape of train_y: {} \n".format(y_train_over.shape))


# In[488]:


models = []  # Empty list to store all the models

# Appending models into the list
models.append(("LR_over", LogisticRegression(random_state=1)))
models.append(("Bagging_over", BaggingClassifier(random_state=1)))
models.append(("Random forest_over", RandomForestClassifier(random_state=1)))
models.append(("GBM_over", GradientBoostingClassifier(random_state=1)))
models.append(("Adaboost_over", AdaBoostClassifier(random_state=1)))
models.append(("Xgboost_over", XGBClassifier(random_state=1, eval_metric="logloss")))
models.append(("dtree_over", DecisionTreeClassifier(random_state=1)))

results_over = []  # Empty list to store all model's CV scores
names_over = []  # Empty list to store name of the models
score_over = []
# loop through all models to get the mean cross validated score
print("\n" "Cross-Validation Performance - Oversampling:" "\n")
for name, model in models:
    scoring = "recall"
    kfold = StratifiedKFold(
        n_splits=5, shuffle=True, random_state=1
    )  # Setting number of splits equal to 5
    cv_result = cross_val_score(
        estimator=model, X=X_train_over, y=y_train_over, scoring=scoring, cv=kfold
    )
    results_over.append(cv_result)
    names_over.append(name)
    print("{}: {}".format(name, cv_result.mean() * 100))

print("\n" "Validation Performance - Oversampling:" "\n")

for name, model in models:
    model.fit(X_train_over, y_train_over)
    scores_over = recall_score(y_val, model.predict(X_val))
    score_over.append(scores_over)
    print("{}: {}".format(name, scores_over * 100))


# In[489]:


# Plotting boxplots for CV scores of all models defined above
fig = plt.figure()

fig.suptitle("Algorithm Comparison - Oversampling")
ax = fig.add_subplot(111)

plt.boxplot(results_over)
ax.set_xticklabels(names_over, rotation=90)

plt.show()


# #### Observations
# 
# * After oversampling XGBoost continues to be the best performer of all the models
# * In cross validation, Random Forest model performed really well, but validation performance dropped, suggesting overfitting
# * We observe overfitting in other models as well
# * Let's try undersampling the train to handle the imbalance between classes and check performance of the models

# ### Undersampling train data using Random Under Sampler

# In[490]:


rus = RandomUnderSampler(random_state=1)
X_train_under, y_train_under = rus.fit_resample(X_train, y_train)


# In[491]:


print("Before Under Sampling, counts of label 'Yes': {}".format(sum(y_train == 1)))
print("Before Under Sampling, counts of label 'No': {} \n".format(sum(y_train == 0)))

print("After Under Sampling, counts of label 'Yes': {}".format(sum(y_train_under == 1)))
print("After Under Sampling, counts of label 'No': {} \n".format(sum(y_train_under == 0)))

print("After Under Sampling, the shape of train_X: {}".format(X_train_under.shape))
print("After Under Sampling, the shape of train_y: {} \n".format(y_train_under.shape))


# In[492]:


models = []  # Empty list to store all the models

# Appending models into the list
models.append(("LR_under", LogisticRegression(random_state=1)))
models.append(("Bagging_under", BaggingClassifier(random_state=1)))
models.append(("Random forest_under", RandomForestClassifier(random_state=1)))
models.append(("GBM_under", GradientBoostingClassifier(random_state=1)))
models.append(("Adaboost_under", AdaBoostClassifier(random_state=1)))
models.append(("Xgboost_under", XGBClassifier(random_state=1, eval_metric="logloss")))
models.append(("dtree_under", DecisionTreeClassifier(random_state=1)))

results_under = []  # Empty list to store all model's CV scores
names_under = []  # Empty list to store name of the models
score_under = []
# loop through all models to get the mean cross validated score
print("\n" "Cross-Validation Performance - Undersampling:" "\n")
for name, model in models:
    scoring = "recall"
    kfold = StratifiedKFold(
        n_splits=5, shuffle=True, random_state=1
    )  # Setting number of splits equal to 5
    cv_result = cross_val_score(
        estimator=model, X=X_train_under, y=y_train_under, scoring=scoring, cv=kfold
    )
    results_under.append(cv_result)
    names_under.append(name)
    print("{}: {}".format(name, cv_result.mean() * 100))

print("\n" "Validation Performance - Undersampling:" "\n")

for name, model in models:
    model.fit(X_train_under, y_train_under)
    scores_under = recall_score(y_val, model.predict(X_val))
    score_under.append(scores_under)
    print("{}: {}".format(name, scores_under * 100))


# In[493]:


# Plotting boxplots for CV scores of all models defined above
fig = plt.figure()

fig.suptitle("Algorithm Comparison - Undersampling")
ax = fig.add_subplot(111)

plt.boxplot(results_under)
ax.set_xticklabels(names_under, rotation=90)

plt.show()


# #### Observations
# 
# * Model performance improved in general with undersampling
# * Top three models in cross validation and validation - note that models are not overfitting as recall results are similar between them and we see improved performance in validation
# 
# Cross Validation performance in training set:
# - GBM_under: 94.26
# - Adaboost_under: 93.14
# - Xgboost_under: 94.98
# 
# Validation set performance:
# - GBM_under: 96.63
# - Adaboost_under: 96.01
# - Xgboost_under: 95.71

# In[494]:


# Calculating different metrics on validation set for top 3 undersampled models

gbm_under = GradientBoostingClassifier(random_state=1)
adaboost_under = AdaBoostClassifier(random_state=1)
xgboost_under = XGBClassifier(random_state=1, eval_metric="logloss")

gbm_under.fit(X_train_under, y_train_under)
adaboost_under.fit(X_train_under, y_train_under)
xgboost_under.fit(X_train_under, y_train_under)


gbm_under_val_perf = model_performance_classification_sklearn(
    gbm_under, X_val, y_val
)
adaboost_under_val_perf = model_performance_classification_sklearn(
    adaboost_under, X_val, y_val
)
xgboost_under_val_perf = model_performance_classification_sklearn(
    xgboost_under, X_val, y_val
)

print("Validation performance - Undersampling:")
print("Gradient Boosting:")
print("{}".format(gbm_under_val_perf))
print("Ada Boost:")
print("{}".format(adaboost_under_val_perf))
print("XGBoost:")
print("{}".format(xgboost_under_val_perf))


# #### Observations
# 
# * Top 3 recall performance models are also holding up well in terms of accuracy, precision, and F1 scores
# * Given the performance superiority of these 3 models, we will continue to tweak with them, using GridSearchCV and RandomizeSearchCV to assist with hyperparameter tuning

# # Hyperparameter Tuning
# **We will tune Gradient Boosting, Ada Boost, and XGBoost models using GridSearchCV and RandomizedSearchCV and compare the models' performance against each other.**

# ## Gradient Boosting

# ### GridSearchCV

# In[508]:


# Choose the type of classifier. 
gbc_tuned = GradientBoostingClassifier(random_state=1)

# Grid of parameters to choose from
param_grid = {
    "n_estimators": [100,150,200,250],
    "subsample":[0.8,0.9,1],
    "max_features":[0.7,0.8,0.9,1]
}

# Type of scoring used to compare parameter combinations
scorer = metrics.make_scorer(metrics.recall_score)

# Calling GridSearchCV
grid_cv = GridSearchCV(estimator=gbc_tuned, param_grid=param_grid, scoring=scorer, cv=5)

# Fitting parameters in GridSeachCV
grid_cv.fit(X_train_under, y_train_under)

print(
    "Best Parameters:{} \nScore: {}".format(grid_cv.best_params_, grid_cv.best_score_)
)


# In[509]:


# Building model with best parameters
gbc_tuned1 = GradientBoostingClassifier(
    random_state=1, max_features=0.7, n_estimators=200, subsample=0.8
)

# Fit the model on training data
gbc_tuned1.fit(X_train_under, y_train_under)


# In[510]:


# Calculating different metrics on train set
gbc_grid_train = model_performance_classification_sklearn(
    gbc_tuned1, X_train_under, y_train_under
)
print("Training performance:")
gbc_grid_train


# In[511]:


# Calculating different metrics on validation set
gbc_grid_val = model_performance_classification_sklearn(gbc_tuned1, X_val, y_val)
print("Validation performance:")
gbc_grid_val


# In[512]:


# creating confusion matrix
confusion_matrix_sklearn(gbc_tuned1, X_val, y_val)


# #### Observations
# 
# * Strong training and validation performance in recall. There seems to be some overfitting in terms of precision but overall the model holds up well in validation

# ### RandomizedSearchCV

# In[513]:


# Creating pipeline
gbc_tuned = GradientBoostingClassifier(random_state=1)

# Parameter grid to pass in RandomizedSearchCV
param_grid = {
    "n_estimators":np.arange(100,250,50),
    "subsample":[0.8,0.9,1],
    "max_features":np.arange(0.5,1,0.1)
}

# Type of scoring used to compare parameter combinations
scorer = metrics.make_scorer(metrics.recall_score)

# Calling RandomizedSearchCV
randomized_cv = RandomizedSearchCV(
    estimator=gbc_tuned,
    param_distributions=param_grid,
    n_iter=20,
    scoring=scorer,
    cv=5,
    random_state=1,
)

# Fitting parameters in RandomizedSearchCV
randomized_cv.fit(X_train_under, y_train_under)

print(
    "Best parameters are {} with CV score={}:".format(
        randomized_cv.best_params_, randomized_cv.best_score_
    )
)


# In[515]:


# Building model with best parameters
gbc_tuned2 = GradientBoostingClassifier(
    random_state=1, subsample=0.9, n_estimators=200,  max_features=0.8
)

# Fit the model on training data
gbc_tuned2.fit(X_train_under, y_train_under)


# In[516]:


# Calculating different metrics on train set
gbc_random_train = model_performance_classification_sklearn(
    gbc_tuned2, X_train_under, y_train_under
)
print("Training performance:")
gbc_random_train


# In[517]:


# Calculating different metrics on validation set
gbc_random_val = model_performance_classification_sklearn(gbc_tuned2, X_val, y_val)
print("Validation performance:")
gbc_random_val


# In[518]:


# creating confusion matrix
confusion_matrix_sklearn(gbc_tuned2, X_val, y_val)


# #### Observations
# 
# * We continue to observe strong training and validation performance in recall for the best model yielded from RandomizedSearchCV. There seems also to be some overfitting in terms of precision but overall the model holds up well in validation

# ## Ada Boost

# ### GridSearchCV

# In[519]:


# Choose the type of classifier. 
abc_tuned = AdaBoostClassifier(random_state=1)

# Grid of parameters to choose from
param_grid = {
    "base_estimator":[DecisionTreeClassifier(max_depth=1),DecisionTreeClassifier(max_depth=2),
                      DecisionTreeClassifier(max_depth=3)],
    "n_estimators": np.arange(10,110,10),
    "learning_rate":np.arange(0.1,2,0.1)
}
# Type of scoring used to compare parameter combinations
scorer = metrics.make_scorer(metrics.recall_score)

# Calling GridSearchCV
grid_cv = GridSearchCV(estimator=abc_tuned, param_grid=param_grid, scoring=scorer, cv=5)

# Fitting parameters in GridSeachCV
grid_cv.fit(X_train_under, y_train_under)

print(
    "Best Parameters:{} \nScore: {}".format(grid_cv.best_params_, grid_cv.best_score_)
)


# In[520]:


#Building model with best parameters
abc_tuned1 = AdaBoostClassifier(random_state=1, base_estimator=DecisionTreeClassifier(max_depth=2)
                                ,learning_rate=1.2, n_estimators=100)

# Fit the model on training data
abc_tuned1.fit(X_train_under, y_train_under)


# In[521]:


# Calculating different metrics on train set
abc_grid_train = model_performance_classification_sklearn(
    abc_tuned1, X_train_under, y_train_under
)
print("Training performance:")
abc_grid_train


# In[522]:


# Calculating different metrics on validation set
abc_grid_val = model_performance_classification_sklearn(abc_tuned1, X_val, y_val)
print("Validation performance:")
abc_grid_val


# In[523]:


# creating confusion matrix
confusion_matrix_sklearn(abc_tuned1, X_val, y_val)


# #### Observations
# 
# * Although validation performace looks good, performance differences between training and validation indicate that the model is overfitting the data 

# ### RandomizedSearchCV

# In[328]:


# Choose the type of classifier. 
abc_tuned = AdaBoostClassifier(random_state=1)

# Grid of parameters to choose from
param_grid = {
    "base_estimator":[DecisionTreeClassifier(max_depth=1),DecisionTreeClassifier(max_depth=2),
                      DecisionTreeClassifier(max_depth=3)],
    "n_estimators": np.arange(10,110,10),
    "learning_rate":np.arange(0.1,2,0.1)
}

# Type of scoring used to compare parameter combinations
scorer = metrics.make_scorer(metrics.recall_score)

# Calling RandomizedSearchCV
randomized_cv = RandomizedSearchCV(
    estimator=abc_tuned,
    param_distributions=param_grid,
    n_iter=20,
    scoring=scorer,
    cv=5,
    random_state=1,
)

# Fitting parameters in RandomizedSearchCV
randomized_cv.fit(X_train_under, y_train_under)

print(
    "Best parameters are {} with CV score={}:".format(
        randomized_cv.best_params_, randomized_cv.best_score_
    )
)


# In[524]:


# Building model with best parameters
abc_tuned2 = AdaBoostClassifier(random_state=1, n_estimators=80
                                , learning_rate=0.5, base_estimator=DecisionTreeClassifier(max_depth=2))

# Fit the model on training data
abc_tuned2.fit(X_train_under, y_train_under)


# In[525]:


# Calculating different metrics on train set
abc_random_train = model_performance_classification_sklearn(
    abc_tuned2, X_train_under, y_train_under
)
print("Training performance:")
abc_random_train


# In[526]:


# Calculating different metrics on validation set
abc_random_val = model_performance_classification_sklearn(abc_tuned2, X_val, y_val)
print("Validation performance:")
abc_random_val


# In[527]:


# creating confusion matrix
confusion_matrix_sklearn(abc_tuned2, X_val, y_val)


# #### Observations
# 
# * We observe less overfitting in the RandomizedSearchCV model and strong recall performance

# ## XGBoost Classifier

# ### GridSearchCV

# In[499]:


# Choose the type of classifier. 
xgb_tuned = XGBClassifier(random_state=1, eval_metric='logloss')

# Grid of parameters to choose from
param_grid = {
    "n_estimators": [30,50,75],
    "scale_pos_weight":[1,2,5],
    "subsample":[0.7,0.9,1],
    "gamma":[0, 1, 3, 5],
    "learning_rate":[0.05, 0.1,0.2],
    'reg_lambda':[5,10],
    "colsample_bytree":[0.7,0.9,1],
    "colsample_bylevel":[0.5,0.7,1]
}


# Type of scoring used to compare parameter combinations
scorer = metrics.make_scorer(metrics.recall_score)

# Calling GridSearchCV
grid_cv = GridSearchCV(estimator=xgb_tuned, param_grid=param_grid, scoring=scorer, cv=5)

# Fitting parameters in GridSeachCV
grid_cv.fit(X_train_under, y_train_under)

print(
    "Best Parameters:{} \nScore: {}".format(grid_cv.best_params_, grid_cv.best_score_)
)


# In[500]:


# Building model with best parameters
xgb_tuned1 = XGBClassifier(random_state=1, 
                           eval_metric='logloss',
                           colsample_bylevel=0.5,
                           colsample_bytree=0.7,
                           gamma=5,
                           learning_rate=0.05,
                           n_estimators=50,
                           reg_lambda=10,
                           scale_pos_weight=5,
                           subsample=0.7)

# Fit the model on training data
xgb_tuned1.fit(X_train_under, y_train_under)


# In[501]:


# Calculating different metrics on train set
xgb_grid_train = model_performance_classification_sklearn(
    xgb_tuned1, X_train_under, y_train_under
)
print("Training performance:")
xgb_grid_train


# In[502]:


# Calculating different metrics on validation set
xgb_grid_val = model_performance_classification_sklearn(xgb_tuned1, X_val, y_val)
print("Validation performance:")
xgb_grid_val


# In[316]:


# creating confusion matrix
confusion_matrix_sklearn(xgb_tuned1, X_val, y_val)


# #### Observations
# 
# * The model is overfitting the data - we observe this most of all in terms of precision performance which deteriorated significantly between training and validation, from 81.7 to 42.8

# ### RandomizedSearchCV

# In[503]:


# Choose the type of classifier. 
xgb_tuned = XGBClassifier(random_state=1, eval_metric='logloss')

# Grid of parameters to choose from
param_grid = {
   "n_estimators": np.arange(25,75),
    "scale_pos_weight":[1,2,5],
    "subsample":[0.7,0.9,1],
    "gamma":[0, 1, 3, 5],
    "learning_rate": np.arange(0.05, 0.2, 0.05),
    "reg_lambda":[5,10],
    "colsample_bytree":[0.7,0.9,1],
    "colsample_bylevel":[0.5,0.7,1]
}


# Type of scoring used to compare parameter combinations
scorer = metrics.make_scorer(metrics.recall_score)

# Calling RandomizedSearchCV
randomized_cv = RandomizedSearchCV(
    estimator=xgb_tuned,
    param_distributions=param_grid,
    n_iter=20,
    scoring=scorer,
    cv=5,
    random_state=1,
)

# Fitting parameters in RandomizedSearchCV
randomized_cv.fit(X_train_under, y_train_under)

print(
    "Best parameters are {} with CV score={}:".format(
        randomized_cv.best_params_, randomized_cv.best_score_
    )
)


# In[505]:


# Building model with best parameters
xgb_tuned2 = XGBClassifier(random_state=1, 
                           eval_metric='logloss', 
                           subsample=1, 
                           scale_pos_weight=5,
                           reg_lambda=10,
                           n_estimators=55,
                           learning_rate=0.05,
                           gamma=0, 
                           colsample_bytree=0.7, 
                           colsample_bylevel=0.7)

# Fit the model on training data
xgb_tuned2.fit(X_train_under, y_train_under)


# In[506]:


# Calculating different metrics on train set
xgb_random_train = model_performance_classification_sklearn(
    xgb_tuned2, X_train_under, y_train_under
)
print("Training performance:")
xgb_random_train


# In[507]:


# Calculating different metrics on validation set
xgb_random_val = model_performance_classification_sklearn(xgb_tuned2, X_val, y_val)
print("Validation performance:")
xgb_random_val


# In[341]:


# creating confusion matrix
confusion_matrix_sklearn(xgb_tuned2, X_val, y_val)


# #### Observations
# 
# * Model continues to overfit the data and we observe a low precision score in validation of 49.6

# ## Comparing all models

# In[528]:


# training performance comparison

models_train_comp_df = pd.concat(
    [
        gbc_grid_train.T,
        gbc_random_train.T,
        abc_grid_train.T,
        abc_random_train.T,
        xgb_grid_train.T,
        xgb_random_train.T,

    ],
    axis=1,
)
models_train_comp_df.columns = [
    "Gradient Boosting Tuned with Grid search",
    "Gradient Boosting with Random search",
    "AdaBoost Tuned with Grid search",
    "AdaBoost Tuned with Random search",
    "Xgboost Tuned with Grid search",
    "Xgboost Tuned with Random Search",
]
print("Training performance comparison:")
models_train_comp_df


# #### Observations
# 
# * All models performed well in training though it looks like some resulted in overfitting - particularly AdaBoost tuned by GridSearchCV and the 2 XGBoost models

# In[529]:


# Validation performance comparison

models_val_comp_df = pd.concat(
    [
        gbc_grid_val.T,
        gbc_random_val.T,
        abc_grid_val.T,
        abc_random_val.T,
        xgb_grid_val.T,
        xgb_random_val.T,

    ],
    axis=1,
)
models_val_comp_df.columns = [
    "Gradient Boosting Tuned with Grid search",
    "Gradient Boosting with Random search",
    "AdaBoost Tuned with Grid search",
    "AdaBoost Tuned with Random search",
    "Xgboost Tuned with Grid search",
    "Xgboost Tuned with Random Search",
]
print("Validation performance comparison:")
models_val_comp_df


# #### Observations
# 
# * When comparing all of the models side by side, we are able to more clearly see the strongest performer which in this case is the Gradient Boosting model yielded from GridSearchCV hyper parameters - the model held up the best in validation and without as much overfitting as we observe in other models

# ## Performance on the test set

# In[530]:


# Calculating different metrics on the test set
gbc_grid_test = model_performance_classification_sklearn(gbc_tuned1, X_test, y_test)
print("Test performance:")
gbc_grid_test


# - The performance on test data is generalised

# In[533]:


feature_names = X_test.columns
importances = gbc_tuned1.feature_importances_
indices = np.argsort(importances)

plt.figure(figsize=(12, 12))
plt.title("Feature Importances")
plt.barh(range(len(indices)), importances[indices], color="violet", align="center")
plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
plt.xlabel("Relative Importance")
plt.show()


# #### Observations
# 
# * Above we see the features, that is, the factors that carry the most weight in predicting customer attrition - we discuss these in more detail below
# * Overall, predictive features indicate that the number of transactions as well as their amount exert great influence on whether or not a customer is likely to become attrited

# ## Pipelines for productionizing the model
# - Now, we have a final model. let's use pipelines to put the model into production
# 
# 
# 
# ## Column Transformer
# - We know that we can use pipelines to standardize the model building, but the steps in a pipeline are applied to each and every variable - how can we personalize the pipeline to perform different processing on different columns
# - Column transformer allows different columns or column subsets of the input to be transformed separately and the features generated by each transformer will be concatenated to form a single feature space. This is useful for heterogeneous or columnar data, to combine several feature extraction mechanisms or transformations into a single transformer.

# - We will create 2 different pipelines, one for numerical columns and one for categorical columns
# - For numerical columns, we will do missing value imputation as pre-processing
# - For categorical columns, we will do one hot encoding and missing value imputation as pre-processing
# 
# - We are doing missing value imputation for the whole data, so that if there is any missing value in the data in future that can be taken care of.

# In[535]:


# creating a list of numerical variables
numerical_features = [
        "Customer_Age",
        "Dependent_count",
        "Months_on_book",
        "Total_Relationship_Count",
        "Months_Inactive_12_mon",
        "Contacts_Count_12_mon",
        "Credit_Limit",
        "Total_Revolving_Bal",
        "Avg_Open_To_Buy",
        "Total_Amt_Chng_Q4_Q1",
        "Total_Trans_Amt",
        "Total_Trans_Ct",
        "Total_Ct_Chng_Q4_Q1",
        "Avg_Utilization_Ratio",

]

# creating a transformer for numerical variables, which will apply simple imputer on the numerical variables
numeric_transformer = Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))])


# creating a list of categorical variables
categorical_features = ["Gender",
                        "Education_Level",
                        "Marital_Status",
                        "Income_Category",
                        "Card_Category",
]

# creating a transformer for categorical variables, which will first apply simple imputer and 
#then do one hot encoding for categorical variables
categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ]
)
# handle_unknown = "ignore", allows model to handle any unknown category in the test data

# combining categorical transformer and numerical transformer using a column transformer

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numerical_features),
        ("cat", categorical_transformer, categorical_features),
    ],
    remainder="passthrough",
)
# remainder = "passthrough" has been used, it will allow variables that are present in original data 
# but not in "numerical_columns" and "categorical_columns" to pass through the column transformer without any changes


# In[536]:


# Separating target variable and other variables
X = df.drop(columns="Attrition_Flag")
Y = df["Attrition_Flag"]


# - Now we already know the best model we need to process with, so we don't need to divide data into 3 parts

# In[537]:


# Splitting the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.30, random_state=1, stratify=Y
)
print(X_train.shape, X_test.shape)


# In[538]:


# Creating new pipeline with best parameters
model = Pipeline(
    steps=[
        ("pre", preprocessor),
        ("Gradient Boosting", GradientBoostingClassifier(
            random_state=1, 
            max_features=0.7, 
            n_estimators=200, 
            subsample=0.8))
    ]
)
# Fit the model on training data
model.fit(X_train, y_train)


# #### Observations
# 
# * With the final model in hand, we were then able to put it into production by creating a pipeline which sequentially implements a series of data transformations and lastly applies the model
# * Note that the accuracy or performance of the final model has nothing to do with the pipeline. It is used only to make the process of data transformation and fit more structured and organized therefore mitigating chances of error in production

# ## ACTIONABLE INSIGHTS AND RECOMMENDATIONS

# **Actionable Insights**
# * Let’s revisit the features, that is, the factors that carry the most weight in predicting customer attrition - we list features with relative importance of 0.05 and above below:
#     * Total number of transactions over the last 12 months
#     * Total transaction amount over the last 12 months
#     * Total revolving balance carried over from one month to the next
#     * Ratio of the total number of transactions between 4th and 1st quarters
#     * Ratio of the total transaction amount between 4th and 1st quarters 
# 
# * The main insight to draw from these factors is that bank customers who use their credit cards often are the least likely to close their accounts
# * To a lesser degree but still playing a role in predicting customer attrition we have features such as average utilization ratio, number of bank products held by customer, customer age, number of contacts between bank and customer, months inactive, and months on book
# * Average utilization ratio and number of bank products both also point to the importance of use - the more use and uses a customer finds for bank products, the more customer is likely to stay
# * From the Exploratory Data Analysis (EDA) section we learned that age among attrited customers was slightly older, this indicates that a retention strategy should take into account different customer age groups and their needs - with special attention to customers as they enter their 40s and 50s
# * Also from the EDA we saw a positive correlation between the number of contacts a customer has had with the bank and attrition - which means that customers more likely to close accounts will be contacting bank more often prior to doing so. This finding leads to opportunities for the bank to target these customers and devise special strategies to improve their retention 
# * Months inactive and months on book both point out to an interesting finding: that attrition happens at greater proportions among brand new customers. Remember that from the EDA we learned that nearly half of customers who opened an account at the bank attrited less than a month later. This finding should also play a significant role for the bank in devising its new customer retention strategies
# 
# **Recommendations**
# * Given that the final predictive model placed greater importance on features related to credit card use (including total number of transactions, transaction amounts, month-to-month balance, average utilization ratio), we believe that a customer retention strategy should place credit card use front and center, driving main tactics to be put in place
# 
# Here are some recommendations we believe could be help Thera Bank to better retain its credit card customers:
# * To stimulate use, devise promotional strategies that include big-ticket purchase offers of promotional pricing and deferred interest and cash-back offers 
# * Define target audience segments by different demographic (e.g. customer age), psychographic & behavioral characteristics (e.g., shopping habits) in order to develop messages that speak to different customers’ spending needs and desires
# * Segment and target customers also based on their average spending/transaction amount: for example, frequent spenders, medium spenders, high-ticket occasional spenders, less frequent spenders
# * From the EDA, it looks like many customers are lost less than one month after opening an account - to improve customer retention during this crucial period, we recommend looking for ways to incentivize a first purchase within the first weeks after customer signs up for their credit card account
# * Develop a retention strategy specially tailored to customers contacting the bank to report issues with their credit cards or expressing desire to close their account. Attrition currently is really high among customers who have contacted the bank over 5 times in the last 12 months which indicates that there is room for improvement in the bank’s service to these customers

# In[ ]:




