#!/usr/bin/env python
# coding: utf-8

# ####  Code and analysis by Gracieli Scremin
# 
# # E-News Express Project

# ## Description
# 
# 
# 
# ### Background: 
# 
# An online news portal aims to expand its business by acquiring new subscribers. Every visitor to the website takes certain actions based on their interest. The company plans to analyze these interests and wants to determine whether a new feature will be effective or not. Companies often analyze users' responses to two variants of a product to decide which of the two variants is more effective. This experimental technique is known as a/b testing that is used to determine whether a new feature attracts users based on a chosen metric.
# 
# Suppose you are hired as a Data Scientist in E-news Express. The design team of the company has created a new landing page. You have been assigned the task to decide whether the new landing page is more effective to gather new subscribers. Suppose you randomly selected 100 users and divided them equally into two groups. The old landing page is served to the first group (control group) and the new landing page is served to the second group (treatment group). Various data about the customers in both groups are collected in 'abtest.csv'. Perform the statistical analysis to answer the following questions using the collected data.
# 
# 
# ### Objective:
# 
# Statistical analysis of business data. Explore the dataset and extract insights from the data. The idea is for you to get comfortable with doing statistical analysis in Python.
# 
# You are expected to perform the statistical analysis to answer the following questions:
# 
# * Explore the dataset and extract insights using Exploratory Data Analysis.
# #### (Project Questions)
# * Do the users spend more time on the new landing page than the old landing page?
# * Is the conversion rate (the proportion of users who visit the landing page and get converted) for the new page greater than the conversion rate for the old page?
# * Does the converted status depend on the preferred language? [Hint: Create a contingency table using the pandas.crosstab() function]
# * Is the mean time spent on the new page same for the different language users?
# 
# 
# ** Consider a significance level of 0.05 for all tests.
# 
# 
# ### Data Dictionary:
# 
# * user_id - This represents the user ID of the person visiting the website.
# * group - This represents whether the user belongs to the first group (control) or the second group (treatment).
# * landing_page - This represents whether the landing page is new or old.
# * time_spent_on_the_page - This represents the time (in minutes) spent by the user on the landing page.
# * converted - This represents whether the user gets converted to a subscriber of the news portal or not.
# * language_preferred - This represents the language chosen by the user to view the landing page.

# ### Importing the necessary libraries - pandas, numpy, seaborn, matplotlib.pyplot:

# In[62]:


# install the scipy version 1.6.1. and restart the kernel after the successful installation
get_ipython().system('pip install scipy==1.6.1')


# In[63]:


# import the scipy and check the version to be sure that the version is 1.6.1.
import scipy
scipy.__version__


# In[64]:


# Library to suppress warnings or deprecation notes 
import warnings
warnings.filterwarnings('ignore')

#import the important packages
import pandas as pd #library used for data manipulation and analysis
import numpy as np # library used for working with arrays.
import matplotlib.pyplot as plt # library for plots and visualisations
import seaborn as sns # library for visualisations
get_ipython().run_line_magic('matplotlib', 'inline')

import scipy.stats as stats # this library contains a large number of probability distributions a growing library of statistical functions.


# ### Loading the dataset

# In[65]:


data=pd.read_csv('abtest.csv')


# ### Looking at the first few rows of the dataset

# In[66]:


data.head()


# ### Checking shape (total rows and columns) of dataset

# In[67]:


data.shape


# ### Copying dataset in order to run analyses while maintaining original intact

# In[68]:


df = data.copy()


# ### Checking first rows and shape to make sure copy was successful

# In[9]:


df.head()


# In[69]:


df.shape


# ### Checking column names, count and dtypes

# In[70]:


df.info()


# #### Observations
# 
# * All 6 columns have 100 observations which seems to indicate there are no missing values but we will check on this below
# * There are four columns coded as "object" indicating qualitative variables: group, landing_page, converted, language_preferred
# * user_id and time_spent_on_the_page are listed as quantitative variables/features and their values categorized as interger and float, respectively

# ### Changing "object" data type to "category" for group, landing_page, converted, language_preferred

# In[71]:


df.group = df.group.astype('category')
df.landing_page = df.landing_page.astype('category')
df.converted = df.converted.astype('category')
df.language_preferred = df.language_preferred.astype('category')


# In[72]:


df.info()


# Memory usage drop to 2.5KB

# ### Checking for any missing values

# In[73]:


df.isnull().sum()


# No missing values

# ### Renaming long name columns to facilitate coding

# In[74]:


df.rename(columns = {'time_spent_on_the_page': 'time', 'language_preferred': 'language'}, inplace = True)


# ## Exploratory Data Analysis

# ### Descriptive statistical summary of categorical and numerical variables

# In[75]:


df.describe(include='all')


# #### Observations
# 
# * group variable is split evenly 50/50 between website visitors in treatment condition and website visitors in control condition
# * following logically from group variable, landing_page is also split evenly 50/50 between website visitors in "old" landing page treatment condition and website visitors in "new" landing page condition
# * for time spent on page, the average time was 5.38, just slightly lower than the median of 5.42. The 50% mid-range of the distribution indicate visitors spent between 3.88 and 7.02 minutes on the landing page  
# * Most visitors, 54 of them, opted to become subscribers
# * The language preferred by most visitors (34 of them) was Spanish (to be looked into further below)

# In[76]:


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
    sns.boxplot(feature, ax=ax_box2, showmeans=True, color='orange') # boxplot will be created and a star will indicate the mean value of the column
    sns.distplot(feature, kde=F, ax=ax_hist2, bins=bins) if bins else sns.distplot(feature, kde=False, ax=ax_hist2) # For histogram
    ax_hist2.axvline(np.mean(feature), color='g', linestyle='--') # Add mean to the histogram
    ax_hist2.axvline(np.median(feature), color='black', linestyle='-') # Add median to the histogram


# #### Boxplot and Histogram for Time Spent on Page variable

# In[77]:


histogram_boxplot(df.time)


# In[78]:


#Function to calculate IQR
def IQR(feature):
    return feature.quantile(0.75) - feature.quantile(0.25)


# In[79]:


IQR(df.time)


# #### Observations
# 
# * For time spent on page, the average was 5.38 minutes, just slightly lower than the median of 5.42
# * The 50% mid-range of the distribution indicate visitors spent between 3.88 and 7.02 minutes on the landing page
# * Distribution resembling a normal distribution

# ### Visually exploring categorical variables

# In[80]:


# Function to create barplots that indicate percentage for each category.
def bar_perc(plot, feature):
    '''
    plot
    feature: 1-d categorical feature array
    '''
    total = len(feature) # length of the column
    for p in ax.patches:
        percentage = '{:.1f}%'.format(100 * p.get_height()/total) # percentage of each class of the category
        x = p.get_x() + p.get_width() / 2 - 0.05 # width of the plot
        y = p.get_y() + p.get_height()           # hieght of the plot
        ax.annotate(percentage, (x, y), size = 12) # annotate the percantage


# #### Experimental Condition

# In[81]:


plt.figure(figsize=(10,7))
ax = sns.countplot(df.group)
plt.xlabel('Experimental Condition')
plt.ylabel('Count')
bar_perc(ax,df.group)


# #### Landing Page

# In[82]:


plt.figure(figsize=(10,7))
ax = sns.countplot(df.landing_page)
plt.xlabel('Landing Page')
plt.ylabel('Count')
bar_perc(ax,df.landing_page)


# #### Confirming all users in Treatment group visited new landing page

# In[24]:


df[(df['group'] == 'treatment')].landing_page.value_counts()


# The table above indicates that 100% of users (n=50) in treatment group landed on new landing page, thereby confirming that intended experimental condition was delivered.

# #### Conversion Status

# In[83]:


plt.figure(figsize=(10,7))
ax = sns.countplot(df.converted)
plt.xlabel('Subscription Conversion Status - Yes-converted, No-unconverted')
plt.ylabel('Count')
bar_perc(ax,df.converted)


# #### User-Preferred Language

# In[84]:


plt.figure(figsize=(10,7))
ax = sns.countplot(df.language)
plt.xlabel('Preferred Language')
plt.ylabel('Count')
bar_perc(ax,df.language)


# #### Observations
# 
# * Sample is distributed 50/50 between control/old landing page and treatment/new landing page visitor groups
# * Most visitors, 54%, opted to become subscribers
# * Spanish and French tied as language most preferred visitors
# 

# ## Project Question:
# 
# ### Do users spend more time on the new landing page than the old landing page?

# #### Defining null and alternate hypotheses

# * $H_0$: User time spent on new and old landing pages is the same
# * $H_A$: User time spent on the new landing page is greater than on the old landing page

# #### Selecting Appropriate test

# T-test for comparing means (with unknown variance) of a continuous variable (time spent on landing page) will be used here to test the hypothesis concerning 2 independent samples - users of new landing page (treatment condition) and users of old landing page (control condition).

# #### Pre-defined significance level

# Here, we select α= 0.05

# In[46]:


#Defining dataframe with only new landing page values
df_newpage = df[(df['landing_page'] == 'new')]


# In[47]:


#Defining dataframe with only old landing page values
df_oldpage = df[(df['landing_page'] == 'old')]


# In[29]:


# find the sample means and sample standard deviations for the two samples
print('The mean time spent on landing page for new landing page group is ' + str(df_newpage['time'].mean()))
print('The mean time spent on landing page for old landing page group is ' + str(df_oldpage['time'].mean()))
print('The std deviation time spent on landing page for new landing page group is ' + str(round(df_newpage['time'].std(), 2)))
print('The std deviation time spent on landing page for old landing page group is ' + str(round(df_oldpage['time'].std(), 2)))


# In[57]:


sns.distplot(df_oldpage["time"], color="grey")
sns.distplot(df_newpage["time"], color="red")
plt.show()


# In[90]:


#import the required functions
from scipy.stats import ttest_ind

# find the p-value
test_stat, p_value = ttest_ind(df_newpage['time'], df_oldpage['time'].dropna(), equal_var = False, alternative = 'greater')
print('The test statistic value is ', str(round(test_stat, 2)))
print('The p-value is ', str(round(p_value, 5)))


# ### Observation
# 
# The p-value being less than the 5% significance level indicates that the difference of means between new landing page (treatment condition) and old landing page (control condition) in terms of time spent on landing page is significantly different and that users that landed on new landing page spent on average more time.
# 
# 
# ### Insight
# 
# The data indicates that users spend more time on new versus old landing page, providing support for adoption of new landing page

# ## Project Question:
# ### Is the mean time spent on the new page same for the different language users?

# #### Defining null and alternate hypotheses

# * $H_0$: The mean time spent on the new landing page is the same for the different language user groups
# * $H_A$: The mean time spent on the new landing page is different for the different language user groups

# #### Selecting Appropriate test

# One-way ANOVA F-test will be used here to test the hypothesis 
# concerning user-preferred language among three different groups English, French, Spanish, and a continous variable (time spent on new landing page).

# #### Pre-defined significance level

# Here, we select α= 0.05

# In[48]:


# mean of time spent on new landing page by user-preferred language (English, Spanish, French)
print(df_newpage.groupby("language")["time"].mean())

# draw the boxplot for visualization 
fig, ax = plt.subplots(figsize = (6,6))
a = sns.boxplot(x= "language", y = 'time' , data = df_newpage, hue = 'language')
a.set_title("Time Spent on New Landing Page v User-Preferred Language (English, Spanish, French)", fontsize=15)
plt.show()


# In[93]:


test_stat, p_value = f_oneway(df_newpage.loc[df['language']=='Spanish', 'time'], df_newpage.loc[df['language']=='English', 'time'],
                             df_newpage.loc[df['language']=='French', 'time'])
print('The test statistic value is ', str(round(test_stat, 2)))
print('The p-value is ', str(round(p_value, 5)))


# In[50]:


#checking differences pairwise by user-preferred language
from statsmodels.stats.multicomp import pairwise_tukeyhsd


# In[51]:


m_comp = pairwise_tukeyhsd(endog=df_newpage['time'], groups=df_newpage['language'], alpha=0.05)
print(m_comp)


# ### Observation
# 
# The p-value being greater than the 5% significance level indicates that the difference of means between the different user-preferred language groups in terms of time spent on new landing page is not significantly different. Therefore we fail to reject the null hypothesis.
# 
# 
# ### Insight
# 
# The new landing page effectiveness in terms of time spent on page does not seem to impact users differently based on their language preferrence

# ## Project Question:
# ### Is the conversion rate (the proportion of users who visit the landing page and get converted) for the new page greater than the conversion rate for the old page?
# 

# #### Defining null and alternate hypotheses

# * $H_0$: The conversion rate (the proportion of users who visit the landing page and get converted) for new and old landing pages is the same
# * $H_A$: The conversion rate (the proportion of users who visit the landing page and get converted) for new and old landing pages is different

# #### Selecting Appropriate test

# This is a problem of Chi-square test of independence, 
# concerning the two independent categorical variables, experimental condition 
# (control-old landing page and treatment-new landing page) and conversion (subscribed/did not subscribe to website).

# #### Pre-defined significance level

# Here, we select α= 0.05

# In[94]:


#Importing the appropriate statistical analysis method
from scipy.stats import chi2_contingency


# In[95]:


#creating contingency table for group (treatment/control) and conversion (yes/no)
contingency1 = pd.crosstab(index=df['group'], columns=df['converted'])
contingency1


# In[96]:


# use chi2_contingency() to find the p-value
chi2, p_value, dof, exp_freq = chi2_contingency(contingency1)
# print the p-value
print('The test statistic value is ', str(round(chi2, 2)))
print('The p-value is ', str(round(p_value, 5)))


# In[97]:


#creating contingency table for group (treatment/control) and conversion (yes/no)
contingency_1 = pd.crosstab(index=df['landing_page'], columns=df['converted'])
contingency_1


# In[98]:


# use chi2_contingency() to find the p-value
chi2, p_value, dof, exp_freq = chi2_contingency(contingency_1)
# print the p-value
print('The test statistic value is ', str(round(chi2, 2)))
print('The p-value is ', str(round(p_value, 5)))


# ### Observation
# 
# There is enough evidence to reject the null hypothesis that there is no difference in conversion rates between users who were exposed to old vs. users exposed to new landing page. Users who were exposed to new landing page subscribed to the site a statistically significant higher rate than users exposed to old landing page. 
# 
# ### Insight
# The data suggests that the new landing page is more effective than the old landing page at converting users into new  subscribers.

# ## Project Question
# ### Does the converted status depend on the preferred language?

# #### Defining null and alternate hypotheses

# * $H_0$: The conversion rate (the proportion of users who visit the landing page and get converted) for different preferred language user groups is the same
# * $H_A$: The conversion rate (the proportion of users who visit the landing page and get converted) for different preferred language user groups is the different

# #### Selecting Appropriate test

# This is a problem of Chi-square test of independence, 
# concerning the two independent categorical variables, user-preferred language (English, Spanish, or French) and conversion (subscribed/did not subscribe to website).

# #### Pre-defined significance level

# Here, we select α= 0.05

# In[44]:


#creating contingency table for treatment groups and conversions
contingency2 = pd.crosstab(index=df['language'], columns=df['converted'])
contingency2


# In[99]:


# use chi2_contingency() to find the p-value
chi2, p_value, dof, exp_freq = chi2_contingency(contingency2)
# print the p-value
print('The test statistic value is ', str(round(chi2, 2)))
print('The p-value is ', str(round(p_value, 5)))


# ### Observation
# 
# We fail to reject the null hypothesis - there is not enough evidence to indicate a significant difference in conversion status between English, Spanish, French preferred language users.
# 
# ### Insight
# 
# User language preferrence does not seem to be a significant indicator in terms of how likely a user is to subscribe to the website

# ## Managerial Implications
# 
# Through the statistical analyses performed we suggest E-News Express management to adopt the new landing page.
# 
# - Is the new landing page more effective?
# Yes. Users who were served the new landing page spent more time and were more likely to subscribe to E-News Express 
# 
# - Does preferred language influence desired outcomes?
# User-preferred language is not a significant factor in terms of how likely a user is to subscribe to E-News or time spent on the site’s landing page
# 
