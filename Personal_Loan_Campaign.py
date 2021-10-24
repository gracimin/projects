#!/usr/bin/env python
# coding: utf-8

# #### Code and analysis by Gracieli Scremin

# ## Personal Loan Campaign Project

# ### Description
# 
# #### Background and Context
# 
# AllLife Bank is a US bank that has a growing customer base. The majority of these customers are liability customers (depositors) with varying sizes of deposits. The number of customers who are also borrowers (asset customers) is quite small, and the bank is interested in expanding this base rapidly to bring in more loan business and in the process, earn more through the interest on loans. In particular, the management wants to explore ways of converting its liability customers to personal loan customers (while retaining them as depositors).
# 
# A campaign that the bank ran last year for liability customers showed a healthy conversion rate of over 9% success. This has encouraged the retail marketing department to devise campaigns with better target marketing to increase the success ratio.
# 
# You as a Data scientist at AllLife bank have to build a model that will help the marketing department to identify the potential customers who have a higher probability of purchasing the loan.
# 
# #### Objective
# 
# * To predict whether a liability customer will buy a personal loan or not.
# * Which variables are most significant.
# * Which segment of customers should be targeted more.
# 
# #### Data Dictionary
# * ID: Customer ID
# * Age: Customer’s age in completed years
# * Experience: #years of professional experience
# * Income: Annual income of the customer (in thousand dollars)
# * ZIP Code: Home Address ZIP code.
# * Family: the Family size of the customer
# * CCAvg: Average spending on credit cards per month (in thousand dollars)
# * Education: Education Level. 1: Undergrad; 2: Graduate;3: Advanced/Professional
# * Mortgage: Value of house mortgage if any. (in thousand dollars)
# * Personal_Loan: Did this customer accept the personal loan offered in the last campaign?
# * Securities_Account: Does the customer have securities account with the bank?
# * CD_Account: Does the customer have a certificate of deposit (CD) account with the bank? 
# * Online: Do customers use internet banking facilities?
# * CreditCard: Does the customer use a credit card issued by any other Bank (excluding All life Bank)?

# ### Loading Libraries

# In[2]:


# Library to suppress warnings or deprecation notes
import warnings

warnings.filterwarnings("ignore")

# Libraries to help with reading and manipulating data

import pandas as pd
import numpy as np

# Library to split data
from sklearn.model_selection import train_test_split

# libaries to help with data visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Removes the limit for the number of displayed columns
pd.set_option("display.max_columns", None)
# Sets the limit for the number of displayed rows
pd.set_option("display.max_rows", None)

# Libraries to build decision tree classifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree

# To tune different models
from sklearn.model_selection import GridSearchCV

# To perform statistical analysis
import scipy.stats as stats

# To build model for prediction
import statsmodels.stats.api as sms
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm
from statsmodels.tools.tools import add_constant
from sklearn.linear_model import LogisticRegression

# To get diferent metric scores
from sklearn.metrics import (
    f1_score,
    accuracy_score,
    recall_score,
    precision_score,
    confusion_matrix,
    roc_auc_score,
    plot_confusion_matrix,
    precision_recall_curve,
    roc_curve,
    make_scorer,
)


# ### Loading the data

# In[3]:


loan_data = pd.read_csv("Loan_Modelling.csv")


# In[4]:


loan_data.head()


# In[5]:


df = loan_data.copy()


# In[6]:


df.shape


# In[7]:


df.info()


# In[8]:


df.isnull().sum()


# #### Observations
# * We observe that the original dataset has 5000 rows and 14 columns
# * We also observe no missing values
# * It looks like units of measurement are not recorded along with values which helps in pre-processing

# ##### Renaming variables for ease of interpretation

# In[9]:


df.rename(columns = {'Family': 'Family_Size', 
                          'CCAvg': 'CreditCard_Spending',
                          'Mortgage': 'Mortgage_Value',
                          'Online': 'Online_Banking',
                          'CreditCard':'Other_CreditCard'}, inplace = True)


# ##### Enriching the dataset by adding information based on zipcode from uszipcode

# In[10]:


from uszipcode import SearchEngine #, Zipcode, SimpleZipcode
search = SearchEngine(simple_zipcode=True)

df['State'] = df['ZIPCode'].map(lambda x: search.by_zipcode(x).state).astype('category',copy=False)


# In[11]:


#Re-examining the structure of data after changes
df.info()


# In[12]:


df.head()


# In[13]:


df.shape


# #### Observations
# 
# * We observe 34 missing values for the zipcode-related feature, State, which was added to the dataset

# In[14]:


df.describe().T


# #### Observations on continuous numeric columns
# 
# * The mean age of customers in this dataset is 45.3. Minimum age 23 and maximum age 67. 50% of customers fall between the ages of 35 and 55
# * In terms of professional experience, the mean is 20 years though we need to look at this feature more closely as we see, for instance, the presence of non-sensical outliers with minimum professional experience of -3 years
# * Mean annual income for this group is 73.7k, median of 64k, indicating the distribution is right-skewed; presence of outliers also noted
# * Mean average family size is about 2 people, indicating customers in this set have small families - max family size of 4
# * Average spending on credit cards per month: mean of 1.93k, median 1.5k
# * It looks like most customers in this set do not have a mortgage - median of 0. We will examine the distribution further
# * Aside from experience with what appears are non-sensical values below 0, the other features in this dataset have values falling within the bounds of what we would expect to be plausible

# ##### Printing number of count of each unique value in each categorical column
# 
# * Although data type for the columns below is numeric, these are essentially categorical variables. We will look at the value counts of each level below

# In[15]:


cat_cols = ['Personal_Loan', 
            'Education', 
            'Securities_Account', 
            'CD_Account', 
            'Online_Banking', 
            'Other_CreditCard',
            #'ZIPCode', ommitting ZIPCode here due to the high number of values, we will handle Zipcodes later on
            'State']


# In[16]:


for column in cat_cols:
    print(df[column].value_counts())
    print("-" * 40)


# #### Observations on categorical variables (numeric data type for analysis purposes)
# 
# * All zipcodes in dataset, with the exception of some due to missing uszipcode data, come from the state of California
# * In terms of education (1=undergrad, 2=graduate, 3=advanced/professional), most customers fall in the 1 category meaning they have undergraduate degrees (not clear whether customers with some college also fall into this category)
# * Most customers, approximately 60%, use online banking
# * A minority of customers have a securities account or CD account at the bank, 10 and 6%, respectively
# * 30% of customers use a credit card issued by another bank
# * For the target variable, Personal Loan, we see distribution as follows: 480 customers (9.6%) accepted the personal loan offered by the bank in the last marketing campaign whereas 4520 did not

# ##### Dropping State feature given all zipcodes from CA

# In[17]:


df.drop(["State"], axis=1, inplace=True)


# ##### Importing County associated with each CA zipcode to help segment zipcode data

# In[18]:


df['County'] = df['ZIPCode'].map(lambda x: search.by_zipcode(x).county).astype('category',copy=False)


# In[19]:


df.County.value_counts()


# ### Segmenting Counties in CA by Region
# 
# ###### Regions in California by County (source: https://census.ca.gov/regions/)
# 
# 1 - Superior California: Butte, Colusa, El Dorado, Glenn, Lassen, Modoc, Nevada, Placer, Plumas, Sacramento, Shasta, Sierra, Siskiyou, Sutter, Tehama, Yolo, Yuba
# 
# 2 - North Coast: Del Norte, Humboldt, Lake, Mendocino, Napa, Sonoma, Trinity
# 
# 3 - San Francisco Bay Area: Alameda, Contra Costa, Marin, San Francisco, San Mateo, Santa Clara, Solano
# 
# 4 - Northern San Joaquin Valley: Alpine, Amador, Calaveras, Madera, Mariposa, Merced, Mono, San Joaquin, Stanislaus, Tuolumne
# 
# 5 - Central Coast: Monterey, San Benito, San Luis Obispo, Santa Barbara, Santa Cruz, Ventura
# 
# 6 - Southern San Joaquin Valley: Fresno, Inyo, Kern, Kings, Tulare
# 
# 7 - Inland Empire: Riverside, San Bernardino
# 
# 8 - Los Angeles County: Los Angeles
# 
# 9 - Orange Country: Orange
# 
# 10 - San Diego - Imperial: Imperial, San Diego

# In[20]:


#Removing "County" from county name
df.County = df.County.str.rsplit(' ',1).str[0]


# In[21]:


Superior_California = [
    "Butte", 
    "Colusa", 
    "El Dorado", 
    "Glenn", 
    "Lassen",
    "Modoc", 
    "Nevada", 
    "Placer", 
    "Plumas", 
    "Sacramento",
    "Shasta", 
    "Sierra", 
    "Siskiyou", 
    "Sutter", 
    "Tehama", 
    "Yolo", 
    "Yuba",
]
North_Coast = [
    "Del Norte",
    "Humboldt",
    "Lake",
    "Mendocino",
    "Napa",
    "Sonoma",
    "Trinity",
]
    
San_Francisco_Bay_Area = [
    "Alameda", 
    "Contra Costa", 
    "Marin", 
    "San Francisco", 
    "San Mateo", 
    "Santa Clara", 
    "Solano",
]
Northern_San_Joaquin_Valley = [
    "Alpine", 
    "Amador", 
    "Calaveras", 
    "Madera", 
    "Mariposa", 
    "Merced", 
    "Mono",
    "San Joaquin", 
    "Stanislaus", 
    "Tuolumne",
]

Central_Coast = [
    "Monterey", 
    "San Benito", 
    "San Luis Obispo", 
    "Santa Barbara", 
    "Santa Cruz", 
    "Ventura",
]

Southern_San_Joaquin_Valley = [
    "Fresno", 
    "Inyo", 
    "Kern", 
    "Kings", 
    "Tulare",
]

Inland_Empire = ["Riverside", "San Bernardino"]

Los_Angeles_County = ["Los Angeles"]

Orange_County = ["Orange"]

San_Diego_Imperial = ["Imperial", "San Diego"]


# In[22]:


def ca_region_combining(x):
    if x in Superior_California:
        return "Superior_California"
    elif x in North_Coast:
        return "North_Coast"
    elif x in San_Francisco_Bay_Area:
        return "San_Francisco_Bay_Area"
    elif x in Northern_San_Joaquin_Valley:
        return "Northern_San_Joaquin_Valley"
    elif x in Central_Coast:
        return "Central_Coast"
    elif x in Southern_San_Joaquin_Valley:
        return "Southern_San_Joaquin_Valley"
    elif x in Inland_Empire:
        return "Inland_Empire"
    elif x in Los_Angeles_County:
        return "Los_Angeles_County"
    elif x in Orange_County:
        return "Orange_County"
    elif x in San_Diego_Imperial:
        return "San_Diego_Imperial"
    else:
        return x


# In[23]:


df["CA_Region"] = df["County"].apply(ca_region_combining)


# In[24]:


df.CA_Region.value_counts(dropna=False)


# In[25]:


#Replacing null values in CA_Region with "Unknown"
df["CA_Region"].fillna("Unknown", inplace = True)


# In[26]:


#dropping County since reclassified into Regions
df.drop(["County"], axis=1, inplace=True)


# In[27]:


df["CA_Region"] = df["CA_Region"].astype('category',copy=False)


# In[28]:


df.info()


# In[29]:


df.isnull().sum()


# #### Observation
# 
# * CA Region feature successfully created and treated for missing values: 34 labeled as "Unknown"

# ### Handling non-sensical "< 0" years of experience

# In[30]:


df_x = df[df["Experience"] < 0]


# In[31]:


df_x.Experience.value_counts()


# In[32]:


df_x.Experience.count()


# In[33]:


del df_x


# #### Observation
# 
# * We see that 52 customers have professional experience lower than 0 years, which is non-sensical. We will replace these with the lowest possible value, 0.

# In[34]:


df["Experience"].replace([-1, -2, -3],0, inplace=True)


# In[35]:


df.Experience.min()


# # Exploratory Data Analysis - EDA

# ## Univariate Analysis

# In[36]:


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


# ## Continuous Features

# #### Age

# In[37]:


df['Age'].describe()


# In[38]:


histogram_boxplot(df['Age'])


# #### Experience

# In[39]:


df['Experience'].describe()


# In[40]:


histogram_boxplot(df['Experience'])


# #### Income

# In[41]:


df['Income'].describe()


# In[42]:


histogram_boxplot(df['Income'])


# ##### Observations
# 
# 'Income' feature looks to be right-skewed. 
# To handle the skewness and help data behave more like a normal distribution, we will examine whether square root transformation can help

# In[43]:


# looking at data distribution when transformed by taking the square root of Income values
histogram_boxplot(np.sqrt(df['Income']))


# ##### Observation
# Using the square root of values helped getting data distribution closer to normality.

# In[44]:


df['Income' + '_sqrt'] = np.sqrt(df['Income'])
df.drop(['Income'], axis=1, inplace=True)


# In[45]:


print(df['Income_sqrt'].describe())
histogram_boxplot(df['Income_sqrt'])


# #### Credit Card Spending

# In[46]:


df['CreditCard_Spending'].describe()


# In[47]:


histogram_boxplot(df['CreditCard_Spending'])


# #### Observations
# 
# 'CreditCard_Spending' feature looks to be right-skewed. 
# To handle the skewness and help data behave more like a normal distribution, we will examine if square root transformation can help

# In[48]:


# looking at data distribution when transformed by taking the square root of Income values
histogram_boxplot(np.sqrt(df['CreditCard_Spending']))


# #### Observation
# Using the square root of values helped getting data distribution closer to normality.

# In[49]:


df['CreditCard_Spending' + '_sqrt'] = np.sqrt(df['CreditCard_Spending'])
df.drop(['CreditCard_Spending'], axis=1, inplace=True)


# In[50]:


print(df['CreditCard_Spending_sqrt'].describe())
histogram_boxplot(df['CreditCard_Spending_sqrt'])


# #### Mortgage Value

# In[51]:


df['Mortgage_Value'].describe()


# In[52]:


histogram_boxplot(df['Mortgage_Value'])


# * Seeing that most customers in the dataset do not have a mortgage
# * Let's create a feature separating customers who do from customers who do not have a mortgage

# ##### Reclassifying Mortgage_Value
# 
# Between customers who have and who do not have a mortgage

# In[53]:


df['Mortgage_Holder'] = df['Mortgage_Value'].apply(lambda x: 1 if x>0 else 0)


# In[54]:


#dropping Mortgage_Value since reclassified into Mortgage_Holder variable
df.drop(["Mortgage_Value"], axis=1, inplace=True)


# In[55]:


df["Mortgage_Holder"].value_counts()


# * Mortgage_Value successfully recategorized as Mortgage_Holder to indicate customers who have (1) and who do not have (0) a mortgage

# ## Categorical and Ordinal Features

# #### Function to create labeled barplots

# In[56]:


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


# In[57]:


labeled_barplot(df, "Education", perc=True)


# In[58]:


labeled_barplot(df, "Family_Size", perc=True)


# In[59]:


labeled_barplot(df, "Mortgage_Holder", perc=True)


# In[60]:


labeled_barplot(df, "Securities_Account", perc=True)


# In[61]:


labeled_barplot(df, "CD_Account", perc=True)


# In[62]:


labeled_barplot(df, "Other_CreditCard", perc=True)


# In[63]:


labeled_barplot(df, "Online_Banking", perc=True)


# In[64]:


labeled_barplot(df, "CA_Region", perc=True)


# #### Observations
# 
# * Most customers do not currently have a mortgage, securities account, CD account, or credit card from another bank
# * 40% of them have at least some college education and 30% have advanced degrees
# * In terms of family size, 30% of customers report a family size equal to 1
# * 60% of customers report using online banking
# * Most customers live in the San Francisco Bay area followed by Los Angeles county

# ## Observations on the Target Variable: Personal Loan

# In[65]:


labeled_barplot(df,"Personal_Loan", perc=True)


# #### Observations
# 
# * We can see here that 9.6% of customers accepted a personal loan offer after the previous marketing campaign
# * We will use the data here to train and test classification models aimed at predicting the probability that a given customer, with certain characteristics, is to sign up for a loan as a result of the marketing promo 

# ## Bivariate Analysis

# ### Examining Bivariate Relationships between Personal Loan and Continuous Features

# In[66]:


continuous_var = df[['Age', 'Experience', 'Income_sqrt', 'CreditCard_Spending_sqrt', 'Personal_Loan']]


# In[67]:


sns.pairplot(continuous_var, hue="Personal_Loan")
plt.show()


# #### Observations
# 
# * We do not see clear differences in terms of age and professional experience between customers who did not accept vs. accepted the loan
# * It looks like customers who opted for the loan have higher incomes and credit card spending levels than those who did not accept the offer for a loan

# In[68]:


plt.figure(figsize=(15, 7))
sns.heatmap(continuous_var.corr(), annot=True, vmin=-1, vmax=1, fmt=".2f", cmap="Blues")
plt.show()


# #### Observations
# 
# * Accepting the personal loan offer looks to be moderately correlated (positively) with income and credit card spending
# * As indicated above, it looks like age and experience have almost no correlation with accepting a personal loan offer

# ## Examining Relationships between Features and Target Variable

# In[69]:


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


# In[70]:


stacked_barplot(df, "Education", "Personal_Loan")


# #### Observation
# 
# * Approximately 14% and 13% of customers with advanced/professional and graduate degrees accepted the personal loan offer in contrast to 4% of customers with undergraduate education

# In[71]:


stacked_barplot(df, "Mortgage_Holder", "Personal_Loan")


# #### Observation
# 
# * A slightly greater proportion of mortgage holders vs. non-mortgage holders (11% compared to 9%) accepted the personal loan offer

# In[72]:


stacked_barplot(df, "Securities_Account", "Personal_Loan")


# #### Observation
# 
# * Customers with and without a securites account accepted the personal loan offer in almost the same proportion

# In[73]:


stacked_barplot(df, "Other_CreditCard", "Personal_Loan")


# #### Observation
# 
# * Customers with and without a other bank's credit card accepted the personal loan offer in almost the same proportion

# In[74]:


stacked_barplot(df, "Online_Banking", "Personal_Loan")


# #### Observation
# 
# * Customers who use or do not use online banking accepted the personal loan offer in almost the same proportion (both about 9%)

# In[75]:


stacked_barplot(df, "CD_Account", "Personal_Loan")


# #### Observation
# 
# * Approximately 46% of customers who have a CD account accepted the personal loan offer compared to 7% of customers without a CD account at the bank

# In[76]:


stacked_barplot(df, "Family_Size", "Personal_Loan")


# #### Observation
# 
# * Larger families accepted the personal loan offer in greater proportions

# In[77]:


stacked_barplot(df, "CA_Region", "Personal_Loan")


# #### Observations
# 
# * The region with the greatest proportion of personal loan opt ins was the North Coast, followed by Southern San Joaquin Valley
# * The region with least proportion of personal loan opt ins was Northern San Joaquin Valley

# In[78]:


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


# In[79]:


distribution_plot_wrt_target(df, "Age", "Personal_Loan")


# #### Observation
# 
# We hardly see any difference in terms of age between customers who opted in and out of a personal loan

# In[80]:


distribution_plot_wrt_target(df, "Experience", "Personal_Loan")


# #### Observation
# 
# As with age, it looks like personal loan opt ins and outs cannot be differentiated much in terms of professional experience

# In[81]:


distribution_plot_wrt_target(df, "Income_sqrt", "Personal_Loan")


# #### Observation
# 
# From the histograms and boxplots we observe a clear difference of income between personal loan opt in and opt out customers with the opt in group at higher income levels

# In[82]:


distribution_plot_wrt_target(df, "CreditCard_Spending_sqrt", "Personal_Loan")


# #### Observation
# 
# From the histograms and boxplots we observe a clear difference in terms of credit card spending between personal loan offer opt in and opt out customers with the opt in group at spending on average more

# ## <a id='link1'>Summary of EDA</a>
# **Data Description:**
# 
# * Dependent variable is Personal Loan which is of categorical data type.
# * ID is an identifier column and therefore will be dropped from modeling
# * ZIPCode column provided means for data extraction from uszipcode library: customer zip codes were associated with state of residence (all from California) and then further with county
# * Other than CA_Region feature which is coded as categorical, the remaning features in the dataset are numerical
# * There are no missing values
# 
# **Data Cleaning:**
# 
# * We observed that 52 customers had professional experience lower than 0 years, which is non-sensical. We replaced these with the lowest possible value for experience in the dataset, 0.
# * To handle the great number of categorical values in ZIPCode column, we imported data from uszipcode library and observed that all zipcodes were from California, US. We then imported the counties associated with each zip code and pulled data from the California census website to help segment customers into different California regions, thereby reducing the dimension of the zipcode data to 10 distinct values. 
# * We reduced the Mortgate_Value feature to two categories: mortgage holders and non-mortgage holders
# * Square root transformation was performed to handle skeweness in the distribution of values for Income and CreditCard_Spending 
# 
# **Observations from EDA:**
# 
# * `Age`: We did not observe clear differences in terms of age between customers who did not accept vs. accepted the personal loan offered in the previous marketing campaign
# * `Experience`: We did not observe clear differences in terms of professional experience between customers who did not accetp vs. accepted the personal loan offer
# * `Income`: we observed a clear difference of income between personal loan offer opt in and opt out customers with the opt in group at higher income levels              
# * `Credit Card Spending`: we observed a clear difference in terms of credit card spending between personal loan offer opt in and opt out customers with the opt-in group spending more monthly on credit cards on average
# * `Family Size`:  Larger families accepted the personal loan offer in greater proportions than families of smaller size         
# * `Education`: Approximately 14% and 13% of customers with advanced/professional and graduate degrees accepted the personal loan offer in contrast to 4% of customers with undergraduate education                        
# * `Securities Account`: Customers with and without a securities account opted in for the personal loan offer in almost the same proportion       
# * `CD Account`: Approximately 46% of customers who have a CD account opted in for the personal loan offer compared to 7% of customers without a CD account         
# * `Online Banking`: 60% of customers report using online banking. Customers who use or do not use online banking opted in for the personal loan offer in almost the same proportion (both about 9%)            
# * `Other Credit Card`: Customers who reported using and not using another bank's credit card opted in for the personal loan offer in almost the same proportion           
# * `CA Region`: Most customers reported in the dataset live in the San Francisco Bay area followed by Los Angeles county. The region with the greatest proportion of personal loan offer opt ins was the North Coast, followed by Southern San Joaquin Valley. The region with least proportion of personal loan offer opt ins was Northern San Joaquin Valley                 
# * `Mortgage Holder`: A slightly greater proportion of mortgage holders vs.non-mortgage holders (11% compared to 9%) accepted the personal loan offer
# * `Personal_Loan (target variable)`: 9.6% of customers accepted the personal loan offer after the previous marketing campaign
# 

# ### Outlier Treatment for Continuous Variables
# 
# Any value that is smaller than Q1 - 1.5IQR and greater than Q3 + 1.5IQR is defined as an outlier. We will treat outliers in the dataset by flooring and capping.
# 
# - All the values smaller than Q1 - 1.5IQR will be assigned the value equal to Q1 - 1.5IQR (flooring)
# - All the values greater than Q3 + 1.5IQR will be assigned the value equal to Q3 + 1.5IQR (capping)

# In[83]:


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


# ### Treating the outliers

# We observed the presence of outlier values in the Credit Card Spending feature

# In[84]:


df = treat_outliers_all(df, ["CreditCard_Spending_sqrt"])


# In[85]:


# let's look at box plot to see if outliers have been treated or not
plt.figure(figsize=(20, 30))

for i, variable in enumerate(["CreditCard_Spending_sqrt"]):
    plt.subplot(5, 4, i + 1)
    plt.boxplot(df[variable], whis=1.5)
    plt.tight_layout()
    plt.title(variable)

plt.show()


# ### Data Normalization
# 
# Normalizing scales of numeric features for better modeling performance.
# 
# Differences in scales across input variables may increase the difficulty of the problem being modeled - a model with large weight values is often unstable, meaning it might suffer from poor performance during learning as well as sensitivity to input values thereby resulting in higher generalization error

# In[86]:


df[["Age", "Experience", "Income_sqrt", "CreditCard_Spending_sqrt", "Family_Size"]].describe().T


# In[87]:


from sklearn.preprocessing import MinMaxScaler
min_max_scaler = MinMaxScaler()
df[["Age", "Experience", "Income_sqrt", "CreditCard_Spending_sqrt", "Family_Size"]] = min_max_scaler.fit_transform(df[["Age", "Experience", "Income_sqrt", "CreditCard_Spending_sqrt", "Family_Size"]])


# In[88]:


df[["Age", "Experience", "Income_sqrt", "CreditCard_Spending_sqrt", "Family_Size"]].describe().T


# #### Observation
# 
# * Non-categorical features normalized successfully - this was done primarily in order to stabilize logistic regression model performance

# # Building the Predictive Models

# ### Goal of the model
# 
# * To recap, the goal is to build a model that will help the marketing department identify the potential customers who have a higher probability of accepting a personal loan offer
# 
# For the predictive model it is important to remember the following in terms of target variable classification: 
# 
# - True Positive: customer classified by the model as “will accept personal loan offer” and who had in fact “accepted offer” in reality.
# - False Positive: customer classified by the model as “will accept personal loan offer” but who actually “did not accept offer” in reality.
# - True Negative: customer classified by the model as “will not accept personal loan offer” and who in fact “did not accept offer” in reality.
# - False Negative: customer classified by the model as “will not accept personal loan offer” but who had actually “accepted offer” in reality.
# 
# ###### Precision:
# 
# - Of those classified as “Will accept personal loan offer,” what proportion actually did?
# - True positive / (True positive + False positive)
# 
# 
# ###### Recall: 
# - Of those that in fact “accepted the personal loan offer,” what proportion were classified that way?
# - True positive / (True positive + False negative)
# 
# 
# ##### Better models have higher values for precision and recall.
# 
# For example, 94% precision would mean that almost all customers identified as “will accept the offer” do in fact and 97% recall would mean almost all customers who “accepted the offer” were identified as such.
# 
# If the model performed with 95% precision but 50% recall this would mean that when it identifies someone as “will accept offer,” it’s largely correct, but it mislabels as “will not accept the offer” half of those who did in fact later “accepted offer”.
# 
# 
# ### Precision, Recall, F1 scores
# 
# Within any one model, we can decide to emphasize precision, recall or a balance of the two which is provided by F1 scores - the decision comes down to context. 
# 
# 
# 
# ### Model can make wrong predictions as:
#  
# 1. Predicting a customer will accept the personal loan offer but customer will not.
# 
# 2. Predicting a customer will not accept the personal loan offer but customer will.
# 
# 
# ### Which case is more important? 
# 
# * Given that the marketing team commissioned a model to enhance target marketing and increase success ratio, we attempted to maximize precision (i.e., minimize false positives) while not losing sight of the F1 performance which can serve as a means of comparison and verification for the strongest predictors of the model

# ## Supervised Learning Model 1: Logistic Regression

# ### Logistic Regression (with Sklearn library)

# In[89]:


X = df.drop(["Personal_Loan", "ID", "ZIPCode"], axis=1)
Y = df["Personal_Loan"]

# creating dummy variables
X = pd.get_dummies(X, drop_first=True) #Converting categorical variable CA_Region into dummy variables.

# splitting in training and test set
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=1)


# In[90]:


#Making sure distribution of target variable values in the train and test data is the same as for the original dataset
print("Distribution of target variable values in original dataset:\n", df.Personal_Loan.value_counts(normalize=True))
print("Distribution of target variable values in train dataset:\n", y_train.value_counts(normalize=True))
print("Distribution of target variable values in test dataset:\n", y_test.value_counts(normalize=True))


# #### Observation:
# 
# * As desired, we observe that the target variable is distributed at similar levels in the original, train, and test datasets

# In[91]:


# There are different solvers available in Sklearn logistic regression: we will use the newton-cg solver
model = LogisticRegression(solver="newton-cg", random_state=1)
lg = model.fit(X_train, y_train)


# In[92]:


# predicting on training set
y_pred_train = lg.predict(X_train)


# In[93]:


print("Training set performance:")
print("Accuracy:", accuracy_score(y_train, y_pred_train))
print("Precision:", precision_score(y_train, y_pred_train))
print("Recall:", recall_score(y_train, y_pred_train))
print("F1:", f1_score(y_train, y_pred_train))


# In[94]:


# predicting on the test set
y_pred_test = lg.predict(X_test)


# In[95]:


print("Test set performance:")
print("Accuracy:", accuracy_score(y_test, y_pred_test))
print("Precision:", precision_score(y_test, y_pred_test))
print("Recall:", recall_score(y_test, y_pred_test))
print("F1:", f1_score(y_test, y_pred_test))


# **Observations**
# 
# * Accuracy gives us the fraction of times a prediction was right. In the case of training and testing data this prediction (both for true positives and true negatives) was 95.5% and 94.7% accurate, respectively
# * Precision gives us the percentage of true positives out of all values that were predicted to be positive. In the case of training data, this percentage was 88.2%; for the testing data it was 89.7%
# * Recall is also called the true positive rate, it calculates the percentage of true positives predicted out of all actual positive values (meaning out of true positives and false negatives). In this case, recall for the training data was 60.7% and for test data, 52.3%. 
# * F1 gives us a measure of both the goodness of precision and recall. When precision and recall are both 1, F1 will be 1 as well - no error in the model. In this case, the F1 percentage in the training data was 71.9% and in the test data 66.1%
# * Overall, the performance measures reflected in the training and test data are comparable. The greatest difference being between recall scores

# ### Logistic Regression (with statsmodels library)

# In[96]:


# defining a function to compute different metrics to check performance of a classification model built using statsmodels
def model_performance_classification_statsmodels(
    model, predictors, target, threshold=0.5
):
    """
    Function to compute different metrics to check classification model performance

    model: classifier
    predictors: independent variables
    target: dependent variable
    threshold: threshold for classifying the observation as class 1
    """

    # checking which probabilities are greater than threshold
    pred_temp = model.predict(predictors) > threshold
    # rounding off the above values to get classes
    pred = np.round(pred_temp)

    acc = accuracy_score(target, pred)  # to compute Accuracy
    recall = recall_score(target, pred)  # to compute Recall
    precision = precision_score(target, pred)  # to compute Precision
    f1 = f1_score(target, pred)  # to compute F1-score

    # creating a dataframe of metrics
    df_perf = pd.DataFrame(
        {"Accuracy": acc, "Recall": recall, "Precision": precision, "F1": f1,},
        index=[0],
    )

    return df_perf


# In[97]:


# defining a function to plot the confusion_matrix of a classification model


def confusion_matrix_statsmodels(model, predictors, target, threshold=0.5):
    """
    To plot the confusion_matrix with percentages

    model: classifier
    predictors: independent variables
    target: dependent variable
    threshold: threshold for classifying the observation as class 1
    """
    y_pred = model.predict(predictors) > threshold
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


# In[98]:


X = df.drop(["Personal_Loan", "ID", "ZIPCode"], axis=1)
Y = df["Personal_Loan"]


# creating dummy variables
X = pd.get_dummies(X, drop_first=True)

# adding constant
X = sm.add_constant(X)

# splitting in training and test set
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=1)


# In[99]:


logit = sm.Logit(y_train, X_train.astype(float))
lg = logit.fit(
    disp=False
)  # setting disp=False will remove the information on number of iterations

print(lg.summary())


# In[100]:


print("Training performance:")
model_performance_classification_statsmodels(lg, X_train, y_train)


# **Observations**
# 
# * Negative values of the coefficient shows that probability of customer signing up for a personal loan decreases with the increase of corresponding attribute value.
# 
# * Positive values of the coefficient show that that probability of customer signing up for a personal loan increases with the increase of corresponding attribute value.
# 
# * p-value of a variable indicates if the variable is significant or not. If we consider the significance level to be 0.05 (5%), then any variable with a p-value less than 0.05 would be considered significant.
# 
# * But these variables might contain multicollinearity, which will affect the p-values.
# 
# * We will have to remove multicollinearity from the data to get reliable coefficients and p-values.
# 
# * There are different ways of detecting (or testing) multi-collinearity, one such way is the Variation Inflation Factor.

# ### Additional Information on VIF
# 
# * **Variance  Inflation  factor**:  Variance  inflation  factors  measure  the  inflation  in  the variances of the regression coefficients estimates due to collinearity that exist among the  predictors.  It  is  a  measure  of  how  much  the  variance  of  the  estimated  regression coefficient βk is "inflated" by  the  existence  of  correlation  among  the  predictor variables in the model. 
# 
# * General Rule of thumb: If VIF is 1 then there is no correlation among the kth predictor and the remaining predictor variables, and  hence  the variance of β̂k is not inflated at all. Whereas if VIF exceeds 5, we say there is moderate VIF and if it is 10 or exceeding 10, it shows signs of high multi-collinearity. But the purpose of the analysis should dictate which threshold to use. 

# In[101]:


vif_series = pd.Series(
    [variance_inflation_factor(X_train.values, i) for i in range(X_train.shape[1])],
    index=X_train.columns,
    dtype=float,
)
print("Series before feature selection: \n\n{}\n".format(vif_series))


# #### Observations
# 
# * We observe a high mulitcollinearity in Age and Experience features, both VIF over 10
# * We will drop Age first since it has the highest VIF

# In[102]:


X_train1 = X_train.drop(["Age"], axis=1,)


# In[103]:


vif_series1 = pd.Series(
    [variance_inflation_factor(X_train1.values, i) for i in range(X_train1.shape[1])],
    index=X_train1.columns,
    dtype=float,
)
print("Series after feature removal: \n\n{}\n".format(vif_series1))


# #### Observations
# 
# * None of the variables exhibit high multicollinearity now, so the values in the summary are reliable.
# * Now let's move to removing the insignificant features (p-value>0.05).

# In[104]:


logit1 = sm.Logit(y_train, X_train1.astype(float))
lg1 = logit1.fit(disp=False)
print(lg1.summary())
print("Training performance:")
model_performance_classification_statsmodels(lg1, X_train1, y_train)


# #### Observations
# 
# * In the case of 'CA_Region', all the attributes have a high p-value which means these are not significant and therefore we can drop the complete variable.
# * For other attributes present in the data, the p-values are high for only a few - we will drop them iteratively as sometimes p-values change after dropping a variable.

# In[105]:


X_train2 = X_train1.drop([
    "CA_Region_Inland_Empire", 
    "CA_Region_Los_Angeles_County", 
    "CA_Region_North_Coast",
    "CA_Region_Northern_San_Joaquin_Valley",
    "CA_Region_Orange_County",
    "CA_Region_San_Diego_Imperial",
    "CA_Region_San_Francisco_Bay_Area",
    "CA_Region_Southern_San_Joaquin_Valley",
    "CA_Region_Superior_California",
    "CA_Region_Unknown"], axis=1)


# In[106]:


logit2 = sm.Logit(y_train, X_train2.astype(float))
lg2 = logit2.fit()
print(lg2.summary())
print("Training performance:")
model_performance_classification_statsmodels(lg2, X_train2, y_train)


# #### Observation
# 
# * We observe p-values higher than the 0.05 level for Experience and Mortgage_Holder. We will drop Experience next and then re-examine the model

# In[107]:


X_train3 = X_train2.drop(["Experience"], axis=1)


# In[108]:


logit3 = sm.Logit(y_train, X_train3.astype(float))
lg3 = logit3.fit()
print(lg3.summary())
print("Training performance:")
model_performance_classification_statsmodels(lg3, X_train3, y_train)


# #### Observation
# 
# * P-value for Mortgage Holder continues to be greater than 0.05. We will drop it next.

# In[109]:


X_train4 = X_train3.drop(["Mortgage_Holder"], axis=1)


# In[110]:


logit4 = sm.Logit(y_train, X_train4.astype(float))
lg4 = logit4.fit()
print(lg4.summary())
print("Training performance:")
model_performance_classification_statsmodels(lg4, X_train4, y_train)


# ##### Observation
# 
# **Now no feature has p-value greater than 0.05, so we'll consider the features in *X_train4* as the final ones and *lg4* as final model.**

# ### Coefficient interpretations
# 
# * Coefficients of  Family Size, Education, CD Account (having a certificate of deposit account at the bank), Income, and Credit Card Spending (average monthly credit card spending) are positive; an increase in these will lead to an increase in chances of a customer accepting the personal loan offered in the marketing campaign.
# * Coefficients of Securities Account (having a securities account at the bank), Online Banking (using the bank's online banking facilities), and Other Credit Card (using a credit card issued by any other bank) are negative; an increase in these will lead to a decrease in chances of a customer accepting the personal loan offered in the marketing campaign.

# ###  Converting coefficients to odds
# * The coefficients of the logistic regression model are in terms of log(odd), to find the odds we have to take the exponential of the coefficients. 
# * Therefore, **odds =  exp(b)**
# * The percentage change in odds is given as **odds = (exp(b) - 1) * 100**

# In[111]:


# converting coefficients to odds
odds = np.exp(lg4.params)

# finding the percentage change
perc_change_odds = (np.exp(lg4.params) - 1) * 100

# removing limit from number of columns to display
pd.set_option("display.max_columns", None)

# adding the odds to a dataframe
pd.DataFrame({"Odds": odds, "Change_odd%": perc_change_odds}, index=X_train4.columns).T


# ### Coefficient interpretations
# 
# * `Family Size`: Holding all other features constant a unit change in family size will increase the odds of a customer accepting a personal loan offer after the marketing campaign by 9.61 times or a 861.45% increase in odds.
# * `Education`: Holding all other features constant a unit change in education will increase the odds of a customer accepting a personal loan offer after the marketing campaign by 5.87 times or a 486.92% increase in odds.
# * `Securities Account`: The odds of a customer who has a securities account accepting the personal loan offer is 0.30 times less than a customer who does not have a securities account at the bank or 69.65% less odds of accepting the bank's personal loan offer than the customer who does not have a securities account. 
# * `CD Account`: The odds of a customer who has a CD account at the bank accepting the personal loan offer is 50.333804 times greater than a customer who does not have a CD account or 4933.38% greater odds of accepting the bank's personal loan offer than the customer who does not have a CD account.
# * `Online Banking`: The odds of a customer who uses the bank's online facilities accepting the personal loan offer is 0.51 times less than a customer who does not use online banking or 48.42% less odds of accepting the bank's personal loan offer than the customer who does not make use of online banking.
# * `Other Credit Card`: The odds of a customer who uses a credit card from any other bank accepting the personal loan offer is 0.30 times less than a customer who does not use another bank's credit card or 70.48% less odds of accepting the bank's personal loan offer than the customer who does not use another bank's credit card
# * `Income`: Holding all other features constant a unit change in income will increase the odds of a customer accepting a personal loan offer after the marketing campaign by 2.65 x 10^6 times or a 2.65 x 10^8% increase in odds.
# * `Credit Card Spending`: Holding all other features constant a unit change in credit card spending will increase the odds of a customer accepting a personal loan offer after the marketing campaign by 7.06 times or a 606.01% increase in odds.

# #### Checking model performance on the training set

# In[112]:


# creating confusion matrix
confusion_matrix_statsmodels(lg4, X_train4, y_train)


# In[113]:


log_reg_model_train_perf = model_performance_classification_statsmodels(
    lg4, X_train4, y_train
)

print("Training performance:")
log_reg_model_train_perf


# #### ROC-AUC

# * ROC-AUC on training set

# In[114]:


logit_roc_auc_train = roc_auc_score(y_train, lg4.predict(X_train4))
fpr, tpr, thresholds = roc_curve(y_train, lg4.predict(X_train4))
plt.figure(figsize=(7, 5))
plt.plot(fpr, tpr, label="Logistic Regression (area = %0.2f)" % logit_roc_auc_train)
plt.plot([0, 1], [0, 1], "r--")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver operating characteristic")
plt.legend(loc="lower right")
plt.show()


# ### Model Performance Improvement

# * Let's see if the precision score can be improved further, by changing the model threshold using AUC-ROC Curve.

# ### Optimal threshold using AUC-ROC curve

# In[115]:


# Optimal threshold as per AUC-ROC curve
# The optimal cut off would be where tpr is high and fpr is low
fpr, tpr, thresholds = roc_curve(y_train, lg4.predict(X_train4))

optimal_idx = np.argmax(tpr - fpr)
optimal_threshold_auc_roc = thresholds[optimal_idx]
print(optimal_threshold_auc_roc)


# #### Checking model performance on training set

# In[116]:


# creating confusion matrix
confusion_matrix_statsmodels(
    lg4, X_train4, y_train, threshold=optimal_threshold_auc_roc
)


# In[117]:


# checking model performance for this model
log_reg_model_train_perf_threshold_auc_roc = model_performance_classification_statsmodels(
    lg4, X_train4, y_train, threshold=optimal_threshold_auc_roc
)
print("Training performance:")
log_reg_model_train_perf_threshold_auc_roc


# #### Observations
# 
# * Precision of model has substantially decreased while recall greatly improved
# * Accuracy and F1 have also decreased
# * Substantial increase in type I error (False Positives): from 1.14% to 9.37% - not ideal since we are trying to minimize type I error to maximize precision

# #### Let's use Precision-Recall curve and see if we can find a better threshold

# In[118]:


y_scores = lg4.predict(X_train4)
prec, rec, tre = precision_recall_curve(y_train, y_scores,)


def plot_prec_recall_vs_tresh(precisions, recalls, thresholds):
    plt.plot(thresholds, precisions[:-1], "b--", label="precision")
    plt.plot(thresholds, recalls[:-1], "g--", label="recall")
    plt.xlabel("Threshold")
    plt.legend(loc="upper left")
    plt.ylim([0, 1])


plt.figure(figsize=(10, 7))
plot_prec_recall_vs_tresh(prec, rec, tre)
plt.show()


# * At threshold around 0.38 we will get equal precision and recall but taking a step back and selecting value around 0.40 will provide higher precision which we are trying to maximize

# In[119]:


# setting the threshold
optimal_threshold_curve = 0.40


# In[120]:


# creating confusion matrix
confusion_matrix_statsmodels(lg4, X_train4, y_train, threshold=optimal_threshold_curve)


# In[121]:


log_reg_model_train_perf_threshold_curve = model_performance_classification_statsmodels(
    lg4, X_train4, y_train, threshold=optimal_threshold_curve
)
print("Training performance:")
log_reg_model_train_perf_threshold_curve


# #### Observations
# 
# * Model is performing well on training set.
# * Compared to the default threshold of 0.50, the model with threshold set at 0.40 is performaing about the same in terms of accuracy though precision has decreased. Recall and F1 improved slightly with use of 0.40 threshold compared to 0.5 threshold

# #### Now let's try to go higher than the default threshold of 0.50 and examine the results

# In[122]:


# setting the threshold
higher_threshold_curve1 = 0.60
higher_threshold_curve2 = 0.70


# In[123]:


# creating confusion matrix
confusion_matrix_statsmodels(lg4, X_train4, y_train, threshold=higher_threshold_curve1)


# In[124]:


# creating confusion matrix
confusion_matrix_statsmodels(lg4, X_train4, y_train, threshold=higher_threshold_curve2)


# In[125]:


log_reg_model_train_perf_higher_threshold_curve1 = model_performance_classification_statsmodels(
    lg4, X_train4, y_train, threshold=higher_threshold_curve1
)
print("Training performance:")
log_reg_model_train_perf_higher_threshold_curve1


# In[126]:


log_reg_model_train_perf_higher_threshold_curve2 = model_performance_classification_statsmodels(
    lg4, X_train4, y_train, threshold=higher_threshold_curve2
)
print("Training performance:")
log_reg_model_train_perf_higher_threshold_curve2


# #### Observation
# 
# * Looking at thresholds higher than the 0.5 default precision keeps improving and recall decreasing

# ### Model Performance Summary

# In[127]:


# training performance comparison

models_train_comp_df = pd.concat(
    [
        log_reg_model_train_perf_threshold_auc_roc.T, #log_reg_model_train_perf_threshold_auc_roc
        log_reg_model_train_perf_threshold_curve.T, #log_reg_model_train_perf_threshold_curve
        log_reg_model_train_perf.T,#log_reg_model_train_perf
        log_reg_model_train_perf_higher_threshold_curve1.T,
        log_reg_model_train_perf_higher_threshold_curve2.T
        
    ],
    axis=1,
)
models_train_comp_df.columns = [
    "Logistic Regression-0.09 Threshold",
    "Logistic Regression-0.40 Threshold",
    "Logistic Regression-0.50 Threshold",
    "Logistic Regression-0.60 Threshold",
    "Logistic Regression-0.70 Threshold"
]

print("Training performance comparison:")
models_train_comp_df


# ### Let's check the performance on the test set

# In[128]:


X_test4 = X_test[X_train4.columns].astype(float)


# In[129]:


# creating confusion matrix
confusion_matrix_statsmodels(lg4, X_test4, y_test)


# In[130]:


log_reg_model_test_perf = model_performance_classification_statsmodels(
    lg4, X_test4, y_test
)

print("Test performance:")
log_reg_model_test_perf


# In[131]:


logit_roc_auc_train = roc_auc_score(y_test, lg4.predict(X_test4))
fpr, tpr, thresholds = roc_curve(y_test, lg4.predict(X_test4))
plt.figure(figsize=(7, 5))
plt.plot(fpr, tpr, label="Logistic Regression (area = %0.2f)" % logit_roc_auc_train)
plt.plot([0, 1], [0, 1], "r--")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver operating characteristic")
plt.legend(loc="lower right")
plt.show()


# **Using model with threshold=0.09** 

# In[132]:


# creating confusion matrix
confusion_matrix_statsmodels(lg4, X_test4, y_test, threshold=optimal_threshold_auc_roc)


# In[133]:


# checking model performance for this model
log_reg_model_test_perf_threshold_auc_roc = model_performance_classification_statsmodels(
    lg4, X_test4, y_test, threshold=optimal_threshold_auc_roc
)
print("Test performance:")
log_reg_model_test_perf_threshold_auc_roc


# **Using model with threshold = 0.40**

# In[134]:


# creating confusion matrix
confusion_matrix_statsmodels(lg4, X_test4, y_test, threshold=optimal_threshold_curve)


# In[135]:


log_reg_model_test_perf_threshold_curve = model_performance_classification_statsmodels(
    lg4, X_test4, y_test, threshold=optimal_threshold_curve
)
print("Test performance:")
log_reg_model_test_perf_threshold_curve


# **Using model with threshold = 0.60**

# In[136]:


# creating confusion matrix
confusion_matrix_statsmodels(lg4, X_test4, y_test, threshold=higher_threshold_curve1)


# In[137]:


log_reg_model_test_perf_higher_threshold_curve1 = model_performance_classification_statsmodels(
    lg4, X_test4, y_test, threshold=higher_threshold_curve1
)
print("Test performance:")
log_reg_model_test_perf_higher_threshold_curve1


# **Using model with threshold = 0.70**

# In[138]:


# creating confusion matrix
confusion_matrix_statsmodels(lg4, X_test4, y_test, threshold=higher_threshold_curve2)


# In[139]:


log_reg_model_test_perf_higher_threshold_curve2 = model_performance_classification_statsmodels(
    lg4, X_test4, y_test, threshold=higher_threshold_curve2
)
print("Test performance:")
log_reg_model_test_perf_higher_threshold_curve2


# ### Model performance summary

# In[140]:


# training performance comparison

models_train_comp_df = pd.concat(
    [
        log_reg_model_train_perf_threshold_auc_roc.T, 
        log_reg_model_train_perf_threshold_curve.T, 
        log_reg_model_train_perf.T,
        log_reg_model_train_perf_higher_threshold_curve1.T,
        log_reg_model_train_perf_higher_threshold_curve2.T
        
    ],
    axis=1,
)
models_train_comp_df.columns = [
    "Logistic Regression-0.09 Threshold",
    "Logistic Regression-0.40 Threshold",
    "Logistic Regression-0.50 Threshold",
    "Logistic Regression-0.60 Threshold",
    "Logistic Regression-0.70 Threshold"
]

print("Training performance comparison:")
models_train_comp_df


# In[141]:


# test performance comparison

models_test_comp_df = pd.concat(
    [
        log_reg_model_test_perf_threshold_auc_roc.T, 
        log_reg_model_test_perf_threshold_curve.T, 
        log_reg_model_test_perf.T,
        log_reg_model_test_perf_higher_threshold_curve1.T,
        log_reg_model_test_perf_higher_threshold_curve2.T
        
    ],
    axis=1,
)
models_test_comp_df.columns = [
    "Logistic Regression-0.09 Threshold",
    "Logistic Regression-0.40 Threshold",
    "Logistic Regression-0.50 Threshold",
    "Logistic Regression-0.60 Threshold",
    "Logistic Regression-0.70 Threshold"
]

print("Test set performance comparison:")
models_test_comp_df


# ### Conclusion
# 
# * To recap, the goal of this model building project is to develop a means to better predict, based on a set of characterisitcs of AllLife Bank's customers, which kind of customer would be more likely to accept the bank's marketing campaign offer for a personal loan. 
# 
# * Toward that end, we have been able to build a model that can be used by the marketing team to predict customers more likely to accept the personal loan offer with a precision score of 0.83 on the test set at default threshold 0.5.
# 
# * Perhaps most indicative of the robust quality of the predictive model is the fact that it performs similarly in the training and test modes, **the most closely alligned threshold between training and test datasets being at the 0.5 default level**.
# 
# * The model's coefficients indicate that income, credit card spending, family size, education, and having a CD account at the bank all increase the odds of a customer accepting the bank's personal loan offer.
# 
# * Though with small odds, the model indicates that having a securities account at the bank, using its online banking facilities, as well as using a credit card from another bank decrease the chances that a customer will accept the personal loan offer.
# 

# ### Business Recommendations
# 
# * Overall the bank should target customers with higher levels of income, credit card spending, education, larger family size (2 or more), and who hold a CD account at the bank
# * We also emphasize, based on the model's predictions, that income, having a CD account, education, and credit card spending have the strongest impact at increasing the odds that a customer will accept a loan offer; meaning that the greater the income, credit card spending, and education, the greater the odds that the customer will accept the offer. Having a CD account also exerts a strong positive effect.
# * More highly educated customers as well as those with higher income and credit card spending levels are also more likely to opt in for a loan - campaigns developed to specially address the needs of these customers, the model indicates, would yield higher conversion rates and gains for AllLife Bank
# * An idea would also be to develop offers tailored to customers opening a CD account at the bank as well as existing CD account holders
# * Another opportunity would be to tailor offers to customers who are married and/or have children

# ## Supervised Learning Model 2: Decision Trees

# ### Create dummies

# In[142]:


df1 = pd.get_dummies(df,columns=['CA_Region'], drop_first=True)


# In[143]:


#check dataset after dummies created
df1.info()


# In[144]:


# defining X and y variables
X = df1.drop(["Personal_Loan", "ID", "ZIPCode"], axis=1) 
y = df1[["Personal_Loan"]]

print(X.head())
print(y.head())


# In[145]:


#Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)


# In[146]:


print("Number of rows in train data =", X_train.shape[0])
print("Number of rows in test data =", X_test.shape[0])


# In[147]:


#Making sure distribution of target variable values in the train and test data is the same as for the original dataset
print("Distribution of target variable values in original dataset:\n", df1.Personal_Loan.value_counts(normalize=True))
print("Distribution of target variable values in train dataset:\n", y_train.Personal_Loan.value_counts(normalize=True))
print("Distribution of target variable values in test dataset:\n", y_test.Personal_Loan.value_counts(normalize=True))


# #### Observation:
# 
# * As desired, we observe that the target variable is distributed at similar levels in the original, train, and test datasets

# In[148]:


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
        {"Accuracy": acc, "Recall": recall, "Precision": precision, "F1": f1,},
        index=[0],
    )

    return df_perf


# In[149]:


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


# ### Build Decision Tree Model

# * We will build our model using the DecisionTreeClassifier function. Using default 'gini' criteria for measuring impurity to split. 

# In[150]:


#Build Decision Tree Model
model = DecisionTreeClassifier(
    criterion="gini", random_state=1
)
model.fit(X_train, y_train)


# In[151]:


#Checking model performance on training set
decision_tree_perf_train = model_performance_classification_sklearn(
    model, X_train, y_train
)
decision_tree_perf_train


# In[152]:


confusion_matrix_sklearn(model, X_train, y_train)


# #### Observations
# * Model is able to perfectly classify all the data points on the training set.
# * 0 errors on the training set, each sample has been classified correctly.
# * As we know a decision tree will continue to grow and classify each data point correctly if no restrictions are applied as the trees will learn all the patterns in the training set.
# * This generally leads to overfitting of the model as Decision Tree will perform well on the training set but will fail to replicate the performance on the test set.

# #### Checking model performance on test set

# In[153]:


decision_tree_perf_test = model_performance_classification_sklearn(
    model, X_test, y_test
)
decision_tree_perf_test


# In[154]:


confusion_matrix_sklearn(model, X_test, y_test)


# #### Observation
# 
# * We observe disparity in the performance of model on training set and test set, which suggests that the model is overfitting.

# ## Visualizing the Decision Tree

# In[155]:


column_names = list(X.columns)
feature_names = column_names
print(feature_names)


# In[156]:


plt.figure(figsize=(20, 30))

out = tree.plot_tree(
    model,
    feature_names=feature_names,
    filled=True,
    fontsize=9,
    node_ids=True,
    class_names=True,
)
for o in out:
    arrow = o.arrow_patch
    if arrow is not None:
        arrow.set_edgecolor("black")
        arrow.set_linewidth(1)
plt.show()


# In[157]:


# Text report showing the rules of a decision tree -

print(tree.export_text(model, feature_names=feature_names, show_weights=True))


# In[158]:


importances = model.feature_importances_
indices = np.argsort(importances)

plt.figure(figsize=(12, 12))
plt.title("Feature Importances")
plt.barh(range(len(indices)), importances[indices], color="orange", align="center")
plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
plt.xlabel("Relative Importance")
plt.show()


# ### Using GridSearch for Hyperparameter tuning of our tree model 
# * Let's see if we can improve our model performance even more.

# * Model is giving good and generalized results on training and test set.

# In[159]:


# Choose the type of classifier.
estimator = DecisionTreeClassifier(random_state=1)

# Grid of parameters to choose from

parameters = {
    "max_depth": [np.arange(2, 50, 5), None],
    "criterion": ["entropy", "gini"],
    "splitter": ["best", "random"],
    "min_impurity_decrease": [0.00001, 0.0001, 0.001],
}

# Type of scoring used to compare parameter combinations
acc_scorer = make_scorer(precision_score)

# Run the grid search
grid_obj = GridSearchCV(estimator, parameters, scoring=acc_scorer, cv=5)
grid_obj = grid_obj.fit(X_train, y_train)

# Set the clf to the best combination of parameters
estimator = grid_obj.best_estimator_

# Fit the best algorithm to the data.
estimator.fit(X_train, y_train)


# #### Checking performance on training set

# In[160]:


decision_tree_tune_perf_train = model_performance_classification_sklearn(
    estimator, X_train, y_train
)
decision_tree_tune_perf_train


# In[161]:


confusion_matrix_sklearn(estimator, X_train, y_train)


# #### Checking model performance on test set

# In[162]:


decision_tree_tune_perf_test = model_performance_classification_sklearn(
    estimator, X_test, y_test
)

decision_tree_tune_perf_test


# In[163]:


confusion_matrix_sklearn(estimator, X_test, y_test)


# * After hyperparameter tuning the model has performance has remained same and the model has become simpler.

# In[164]:


plt.figure(figsize=(15, 12))

tree.plot_tree(
    estimator,
    feature_names=feature_names,
    filled=True,
    fontsize=9,
    node_ids=True,
    class_names=True,
)
plt.show()


# In[165]:


# Text report showing the rules of a decision tree -

print(tree.export_text(estimator, feature_names=feature_names, show_weights=True))


# We are getting a more simplified tree after pre-pruning.

# **Observations from decision tree after pre-pruning:**
# 
# * The first split in the decision tree model happens at the income level between customers making more than approx 100k/year (remembering that we are dealing with normalized values for income square root, 0.66 equates to approximately 75th percentile in the dataset's income distribution or 100k - reference below)
# * The model predicts that, among customers who make equal to or less than approx 100k/year and have credit card spending over 2k/month (0.64 is the normalized value of credit card spending square root, which equates to approximately 65th percentile of the credit card spending distribution or 2k - reference below), if they make over 78k (please see reference below) and have a graduate and/or advanced/professional degree will accept the loan offer
# * The model predicts that, among customers who make equal to or less than approx 100k/year and have credit card spending over 2k/month, if they make over 78k and have an undergraduate degree and a CD account at the bank will accept the loan offer
# * The model predicts that, among customers make equal to or less than approx than 100k/year and have credit card spending over 2k/month, if they make below 78k/year and have a CD account at the bank will accept the loan offer
# * The model predicts that a customer who has annual income greater than approx 100k and a graduate or advanced/professional degree will accept the personal loan offer.
# * Customers making more than approx 100k/year without a graduate or advanced/professional degree but with family sizes greater than 2, the model predicts will accept the loan offer
# 
# 
# 
# **Reference**:
# * Income_sqrt = 0.66 ≈ 75th percentile ≈ 100k
# * Income_sqrt = 0.56 ≈ 60th percentile ≈ 78k
# * CreditCard_Spending_sqrt = 0.64 ≈ 65th percentile ≈ 2k

# ### Cost Complexity Pruning

# In[166]:


clf = DecisionTreeClassifier(random_state=1)
path = clf.cost_complexity_pruning_path(X_train, y_train)
ccp_alphas, impurities = path.ccp_alphas, path.impurities


# In[167]:


pd.DataFrame(path)


# In[168]:


fig, ax = plt.subplots(figsize=(15, 5))
ax.plot(ccp_alphas[:-1], impurities[:-1], marker="o", drawstyle="steps-post")
ax.set_xlabel("effective alpha")
ax.set_ylabel("total impurity of leaves")
ax.set_title("Total Impurity vs effective alpha for training set")
plt.show()


# Next, we train a decision tree using the effective alphas. The last value in ccp_alphas is the alpha value that prunes the whole tree, leaving the tree, clfs[-1], with one node.

# In[169]:


clfs = []
for ccp_alpha in ccp_alphas:
    clf = DecisionTreeClassifier(random_state=1, ccp_alpha=ccp_alpha)
    clf.fit(X_train, y_train)
    clfs.append(clf)
print(
    "Number of nodes in the last tree is: {} with ccp_alpha: {}".format(
        clfs[-1].tree_.node_count, ccp_alphas[-1]
    )
)


# For the remainder, we remove the last element in clfs and ccp_alphas, because it is the trivial tree with only one node. Here we show that the number of nodes and tree depth decreases as alpha increases.

# In[170]:


clfs = clfs[:-1]
ccp_alphas = ccp_alphas[:-1]

node_counts = [clf.tree_.node_count for clf in clfs]
depth = [clf.tree_.max_depth for clf in clfs]
fig, ax = plt.subplots(2, 1, figsize=(10, 7))
ax[0].plot(ccp_alphas, node_counts, marker="o", drawstyle="steps-post")
ax[0].set_xlabel("alpha")
ax[0].set_ylabel("number of nodes")
ax[0].set_title("Number of nodes vs alpha")
ax[1].plot(ccp_alphas, depth, marker="o", drawstyle="steps-post")
ax[1].set_xlabel("alpha")
ax[1].set_ylabel("depth of tree")
ax[1].set_title("Depth vs alpha")
fig.tight_layout()


# ### Precision vs alpha for training and testing sets

# In[171]:


precision_train = []
for clf in clfs:
    pred_train = clf.predict(X_train)
    values_train = precision_score(y_train, pred_train)
    precision_train.append(values_train)


# In[172]:


precision_test = []
for clf in clfs:
    pred_test = clf.predict(X_test)
    values_test = precision_score(y_test, pred_test)
    precision_test.append(values_test)


# In[173]:


fig, ax = plt.subplots(figsize=(15, 5))
ax.set_xlabel("alpha")
ax.set_ylabel("Precision")
ax.set_title("Precision vs alpha for training and testing sets")
ax.plot(ccp_alphas, precision_train, marker="o", label="train", drawstyle="steps-post")
ax.plot(ccp_alphas, precision_test, marker="o", label="test", drawstyle="steps-post")
ax.legend()
plt.show()


# In[174]:


# creating the model where we get highest train and test recall
index_best_model = np.argmax(precision_test)
best_model = clfs[index_best_model]
print(best_model)


# #### Checking model performance on training set

# In[175]:


decision_tree_postpruned_perf_train = model_performance_classification_sklearn(
    best_model, X_train, y_train
)
decision_tree_postpruned_perf_train


# In[176]:


confusion_matrix_sklearn(best_model, X_train, y_train)


# #### Checking model performance on test set

# In[177]:


decision_tree_postpruned_perf_test = model_performance_classification_sklearn(
    best_model, X_test, y_test
)
decision_tree_postpruned_perf_test


# In[178]:


confusion_matrix_sklearn(best_model, X_test, y_test)


# ### Visualizing the Decision Tree

# In[179]:


plt.figure(figsize=(10, 10))

out = tree.plot_tree(
    best_model,
    feature_names=feature_names,
    filled=True,
    fontsize=9,
    node_ids=True,
    class_names=True,
)
for o in out:
    arrow = o.arrow_patch
    if arrow is not None:
        arrow.set_edgecolor("black")
        arrow.set_linewidth(1)
plt.show()
plt.show()


# In[180]:


# Text report showing the rules of a decision tree -

print(tree.export_text(best_model, feature_names=feature_names, show_weights=True))


# **Observations from decision tree after post-pruning:**
# 
# * The model predicts that a customer who has annual income greater than approx 100k (reference below) and a graduate or advanced/professional degree will accept the personal loan offer
# * Customers making more than approx 100k/year without a graduate or advanced/professional degree but with family sizes greater than 2, the model predicts will accept the loan offer
# 
# **Reference**:
# * Income_sqrt = 0.66 ≈ 75th percentile ≈ 100k 

# In[181]:


# importance of features in the tree building ( The importance of a feature is computed as the
# (normalized) total reduction of the 'criterion' brought by that feature. It is also known as the Gini importance )

print(
    pd.DataFrame(
        best_model.feature_importances_, columns=["Feature Importance"], index=X_train.columns
    ).sort_values(by="Feature Importance", ascending=False)
)


# In[182]:


importances = best_model.feature_importances_
indices = np.argsort(importances)

plt.figure(figsize=(12, 12))
plt.title("Feature Importances")
plt.barh(range(len(indices)), importances[indices], color="orange", align="center")
plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
plt.xlabel("Relative Importance")
plt.show()


# ### Comparing all the decision tree models

# In[183]:


# training performance comparison

models_train_comp_df = pd.concat(
    [
        decision_tree_perf_train.T,
        decision_tree_tune_perf_train.T,
        decision_tree_postpruned_perf_train.T,
    ],
    axis=1,
)
models_train_comp_df.columns = [
    "Decision Tree sklearn",
    "Decision Tree (Pre-Pruning)",
    "Decision Tree (Post-Pruning)",
]
print("Training performance comparison:")
models_train_comp_df


# In[184]:


# test performance comparison

models_test_comp_df = pd.concat(
    [
        decision_tree_perf_test.T,
        decision_tree_tune_perf_test.T,
        decision_tree_postpruned_perf_test.T,
    ],
    axis=1,
)
models_test_comp_df.columns = [
    "Decision Tree sklearn",
    "Decision Tree (Pre-Pruning)",
    "Decision Tree (Post-Pruning)",
]
print("Test set performance comparison:")
models_test_comp_df


# #### Observations
# 
# * Decision tree with post-pruning is giving the highest precision on the test set - the model had 0 instances of misclassified positives or false positives
# * The pre-prunning model still performs well in terms of precision while doing better than the post-prunning model in recall
# * The tree with post pruning is not complex and easy to interpret
# 
# 

# ## Business Insights
# 
# * Education, income and family size (in that order) are the most important variables in determining if a customer will accept the personal loan offer 
# * The model indicates that, to minimize likelihood of extending loan offer to customers who will not accept it, the core group of customers to be targeted by the campaign should consist of customers with graduate and/or advanced/professional education (perhaps due to this group having incurred more personal debt over the course of their lives), income level over 100k (though the pre-pruning decision tree model predicts, so long as the customer's credit card spending is over 2k/month and customer has a CD account at the bank, that customer will accept the offer), and family size between 2 and 4 (4 being the maximum family size present in this dataset)
# * Main criteria therefore to offer a personal loan should depend on three factors - income, education, and family size. Secondarily, the marketing team could take into account additional 2 factors: credit card spending and customer having a CD account at the bank. 
# * We recommend the marketing campaign to focus on these core customers in order to maximize conversion rates for the personal loan offer

# ## Optimizing Decision Tree Model with F1 scores

# ### Using GridSearch for Hyperparameter tuning of our tree model 
# * Let's see if we can improve our model performance even more.

# * Model is giving good and generalized results on training and test set.

# In[185]:


# Choose the type of classifier.
estimator = DecisionTreeClassifier(random_state=1)

# Grid of parameters to choose from

parameters = {
    "max_depth": [np.arange(2, 50, 5), None],
    "criterion": ["entropy", "gini"],
    "splitter": ["best", "random"],
    "min_impurity_decrease": [0.0001, 0.001, 0.01],
}

# Type of scoring used to compare parameter combinations
acc_scorer = make_scorer(f1_score)

# Run the grid search
grid_obj = GridSearchCV(estimator, parameters, scoring=acc_scorer, cv=5)
grid_obj = grid_obj.fit(X_train, y_train)

# Set the clf to the best combination of parameters
estimator = grid_obj.best_estimator_

# Fit the best algorithm to the data.
estimator.fit(X_train, y_train)


# #### Checking performance on training set

# In[186]:


decision_tree_tune_perf_train_f1 = model_performance_classification_sklearn(
    estimator, X_train, y_train
)
decision_tree_tune_perf_train_f1


# In[187]:


confusion_matrix_sklearn(estimator, X_train, y_train)


# #### Checking model performance on test set

# In[188]:


decision_tree_tune_perf_test_f1 = model_performance_classification_sklearn(
    estimator, X_test, y_test
)

decision_tree_tune_perf_test_f1


# In[189]:


confusion_matrix_sklearn(estimator, X_test, y_test)


# #### Observations:
# 
# * No changes observed from model using precision score to compare parameter combinations

# In[190]:


plt.figure(figsize=(15, 12))

tree.plot_tree(
    estimator,
    feature_names=feature_names,
    filled=True,
    fontsize=9,
    node_ids=True,
    class_names=True,
)
plt.show()


# In[191]:


# Text report showing the rules of a decision tree -

print(tree.export_text(estimator, feature_names=feature_names, show_weights=True))


# We are getting a more simplified tree after pre-pruning.

# **Observations from decision tree after pre-pruning:**
# 
# * No changes observed from model using precision score to compare parameter combinations

# ### Cost Complexity Pruning

# In[192]:


clf = DecisionTreeClassifier(random_state=1)
path = clf.cost_complexity_pruning_path(X_train, y_train)
ccp_alphas, impurities = path.ccp_alphas, path.impurities


# In[193]:


pd.DataFrame(path)


# In[194]:


fig, ax = plt.subplots(figsize=(15, 5))
ax.plot(ccp_alphas[:-1], impurities[:-1], marker="o", drawstyle="steps-post")
ax.set_xlabel("effective alpha")
ax.set_ylabel("total impurity of leaves")
ax.set_title("Total Impurity vs effective alpha for training set")
plt.show()


# Next, we train a decision tree using the effective alphas. The last value in ccp_alphas is the alpha value that prunes the whole tree, leaving the tree, clfs[-1], with one node.

# In[195]:


clfs = []
for ccp_alpha in ccp_alphas:
    clf = DecisionTreeClassifier(random_state=1, ccp_alpha=ccp_alpha)
    clf.fit(X_train, y_train)
    clfs.append(clf)
print(
    "Number of nodes in the last tree is: {} with ccp_alpha: {}".format(
        clfs[-1].tree_.node_count, ccp_alphas[-1]
    )
)


# For the remainder, we remove the last element in clfs and ccp_alphas, because it is the trivial tree with only one node. Here we show that the number of nodes and tree depth decreases as alpha increases.

# In[196]:


clfs = clfs[:-1]
ccp_alphas = ccp_alphas[:-1]

node_counts = [clf.tree_.node_count for clf in clfs]
depth = [clf.tree_.max_depth for clf in clfs]
fig, ax = plt.subplots(2, 1, figsize=(10, 7))
ax[0].plot(ccp_alphas, node_counts, marker="o", drawstyle="steps-post")
ax[0].set_xlabel("alpha")
ax[0].set_ylabel("number of nodes")
ax[0].set_title("Number of nodes vs alpha")
ax[1].plot(ccp_alphas, depth, marker="o", drawstyle="steps-post")
ax[1].set_xlabel("alpha")
ax[1].set_ylabel("depth of tree")
ax[1].set_title("Depth vs alpha")
fig.tight_layout()


# ### F1 vs alpha for training and testing sets

# In[197]:


f1_train = []
for clf in clfs:
    pred_train = clf.predict(X_train)
    values_train = f1_score(y_train, pred_train)
    f1_train.append(values_train)


# In[198]:


f1_test = []
for clf in clfs:
    pred_test = clf.predict(X_test)
    values_test = f1_score(y_test, pred_test)
    f1_test.append(values_test)


# In[199]:


fig, ax = plt.subplots(figsize=(15, 5))
ax.set_xlabel("alpha")
ax.set_ylabel("F1")
ax.set_title("F1 vs alpha for training and testing sets")
ax.plot(ccp_alphas, f1_train, marker="o", label="train", drawstyle="steps-post")
ax.plot(ccp_alphas, f1_test, marker="o", label="test", drawstyle="steps-post")
ax.legend()
plt.show()


# In[200]:


# creating the model where we get highest train and test recall
index_best_model = np.argmax(f1_test)
best_model = clfs[index_best_model]
print(best_model)


# #### Checking model performance on training set

# In[201]:


decision_tree_postpruned_perf_train_f1 = model_performance_classification_sklearn(
    best_model, X_train, y_train
)
decision_tree_postpruned_perf_train_f1


# In[202]:


confusion_matrix_sklearn(best_model, X_train, y_train)


# #### Checking model performance on test set

# In[203]:


decision_tree_postpruned_perf_test_f1 = model_performance_classification_sklearn(
    best_model, X_test, y_test
)
decision_tree_postpruned_perf_test_f1


# In[204]:


confusion_matrix_sklearn(best_model, X_test, y_test)


# ### Visualizing the Decision Tree

# In[205]:


plt.figure(figsize=(10, 10))

out = tree.plot_tree(
    best_model,
    feature_names=feature_names,
    filled=True,
    fontsize=9,
    node_ids=True,
    class_names=True,
)
for o in out:
    arrow = o.arrow_patch
    if arrow is not None:
        arrow.set_edgecolor("black")
        arrow.set_linewidth(1)
plt.show()
plt.show()


# In[206]:


# Text report showing the rules of a decision tree -

print(tree.export_text(best_model, feature_names=feature_names, show_weights=True))


# **Observations from decision tree after post-pruning:**
# 
# * The main differences between F1 and precision models is that in the post-pruning F1 model, Experience and Age (which in the logistic regression model were dropped due to multicollinearity (Age) and p-value being greater than 0.05 (Experience)) enter the picture as predictors and the model gets more complex: the model predicts that customers who make between 100k and 85k/year with credit card spending equal to or less than 2k/month, with family size greater than 3, with over 4 years of professional experience and equal to or less than 60 years of age are predicted to accept the loan offer
# 
# **Reference**:
# * Income_sqrt = 0.66 ≈ 75th percentile ≈ 100k
# * Income_sqrt = 0.62 ≈ 70th percentile ≈ 85k
# * CreditCard_Spending_sqrt = 0.64 = 65th percentile = 2k
# * Family_Size = 0.83 ≈ 80th percentile ≈ 3 people
# * Experience = 0.08 ≈ 10th percentile ≈ 4 years
# * Age = 0.84 ≈ 90th percentile ≈ 60 years

# In[207]:


# importance of features in the tree building ( The importance of a feature is computed as the
# (normalized) total reduction of the 'criterion' brought by that feature. It is also known as the Gini importance )

print(
    pd.DataFrame(
        best_model.feature_importances_, columns=["Feature Importance"], index=X_train.columns
    ).sort_values(by="Feature Importance", ascending=False)
)


# In[208]:


importances = best_model.feature_importances_
indices = np.argsort(importances)

plt.figure(figsize=(12, 12))
plt.title("Feature Importances")
plt.barh(range(len(indices)), importances[indices], color="orange", align="center")
plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
plt.xlabel("Relative Importance")
plt.show()


# ### Comparing all the decision tree models

# In[209]:


# training performance comparison

models_train_comp_df_f1 = pd.concat(
    [
        decision_tree_perf_train.T,
        decision_tree_tune_perf_train_f1.T,
        decision_tree_postpruned_perf_train_f1.T,
    ],
    axis=1,
)
models_train_comp_df_f1.columns = [
    "Decision Tree sklearn F1",
    "Decision Tree (Pre-Pruning) F1",
    "Decision Tree (Post-Pruning) F1",
]
print("Training performance comparison:")
models_train_comp_df_f1


# In[210]:


# test performance comparison

models_test_comp_df_f1 = pd.concat(
    [
        decision_tree_perf_test.T,
        decision_tree_tune_perf_test_f1.T,
        decision_tree_postpruned_perf_test_f1.T,
    ],
    axis=1,
)
models_test_comp_df_f1.columns = [
    "Decision Tree sklearn",
    "Decision Tree (Pre-Pruning) F1",
    "Decision Tree (Post-Pruning) F1",
]
print("Test set performance comparison:")
models_test_comp_df_f1


# #### Observation
# 
# * Except for the added complexity and overall better performance in the test set for the post-prunning model, using F1 score as a means for comparing parameter combinations yielded similar results
# * The model still points out to the same features, namely education, income, and credit card spending, as the main predictors of likelihood that loan offer will be accepted

# In[238]:


# test performance comparison: Precision and F1 benchmarks

models_test_comp_df_f1_precision = pd.concat([
    log_reg_model_test_perf.T,
    decision_tree_tune_perf_test.T, 
    decision_tree_postpruned_perf_test.T, 
    decision_tree_tune_perf_test_f1.T,
    decision_tree_postpruned_perf_test_f1.T], 
    axis=1,
)
models_test_comp_df_f1_precision.columns = [
    "Logistic Regression-0.50 Threshold",
    "Decision Tree (Pre-Pruning) Precision",
    "Decision Tree (Post-Pruning) Precision",
    "Decision Tree (Pre-Pruning) F1",
    "Decision Tree (Post-Pruning) F1"
]
print("Test set performance comparison:")
models_test_comp_df_f1_precision


# In[239]:


row_precision = models_test_comp_df_f1_precision.iloc[2,0:5]


# In[240]:


row_precision.plot(kind='bar', figsize=(8,8),title="Classification Performance - Precision Scores")


# In[241]:


row_f1 = models_test_comp_df_f1_precision.iloc[3,0:5]


# In[242]:


row_f1.plot(kind='bar', figsize=(8,8),title="Classification Performance - F1 Scores")


# #### Observations:
# 
# * The graphs show us a comparison of precision scores and F1 scores for the 0.5 threshold Logistic Regression model and all pre and post-pruning trees
# * They show us that all models performed well
# * The logistic regression model was the lowest performing one with 0.82 precision score and 0.70 F1 score
# * The decision tree models were the best performing with Precision scores of at least 0.93 and F1 scores of at least 0.85
# * The findings indicate that all pruning methods worked well at simplifying the original decision-tree model while maintaining robust classification performance

# # Insights and Final Business Recommendations

# ##### Insights:
# 
# After testing and tuning different classification models using Logistic Regression and Decision Trees we
# have come to key takeaways.
# The models paint the following profile of the ultimate AllLife Bank personal loan customer:
# * Highly educated - educated over undergraduate level
# * High income - annual income over 100k
# * Family size consisting of 2 people or more
# * High credit card spending - customers who spend over 2k/month are more likely to opt in for a personal
# loan
# * CD Account - having a CD account at the bank also improves chances customer will accept loan offer

# ##### Recommendations to help the business grow:
#     
# Based on the models’ predictions, our recommendations are as follows:
# 
# * The core group to be targeted by the campaign should consist of customers with annual income of 100k or above,
# with education attainment over the undergraduate level (perhaps due to this group having incurred more educationrelated
# personal debt), and family sizes between 2 and 4 (4 being the maximum family size tested in this dataset)
# * Campaigns developed to specially address the needs of these customers are predicted to yield higher conversion
# rates and gains for AllLife Bank
# * Another recommendation would be to develop offers tailored to customers with the above characteristics and who
# are also opening a CD account at the bank or who are already existing CD account holders
# * Extending personal loan offers geared toward high earning, highly educated customers who are married and/or
# have children can also lead to higher conversion rates for the campaign

# In[ ]:




