#!/usr/bin/env python
# coding: utf-8

# #### Code and analysis by Gracieli Scremin
# 
# # CARS4U
# 
# ## Background & Context
# 
# There is a huge demand for used cars in the Indian Market today. As sales of new cars have slowed down in the recent past, the pre-owned car market has continued to grow over the past years and is larger than the new car market now. Cars4U is a budding tech start-up that aims to find footholes in this market.
# 
# In 2018-19, while new car sales were recorded at 3.6 million units, around 4 million second-hand cars were bought and sold. There is a slowdown in new car sales and that could mean that the demand is shifting towards the pre-owned market. In fact, some car sellers replace their old cars with pre-owned cars instead of buying new ones. Unlike new cars, where price and supply are fairly deterministic and managed by OEMs (Original Equipment Manufacturer / except for dealership level discounts which come into play only in the last stage of the customer journey), used cars are very different beasts with huge uncertainty in both pricing and supply. Keeping this in mind, the pricing scheme of these used cars becomes important in order to grow in the market.
# 
# As a senior data scientist at Cars4U, you have to come up with a pricing model that can effectively predict the price of used cars and can help the business in devising profitable strategies using differential pricing. For example, if the business knows the market price, it will never sell anything below it. 
# 
# ## Objective
# 
# Explore and visualize the dataset.
# Build a linear regression model to predict the prices of used cars.
# Generate a set of insights and recommendations that will help the business.
# Data Dictionary 
# 
# - S.No.: Serial Number
# - Name: Name of the car which includes Brand name and Model name
# - Location: The location in which the car is being sold or is available for purchase Cities
# - Year: Manufacturing year of the car
# - Kilometers_driven: The total kilometers driven in the car by the previous owner(s) in KM.
# - Fuel_Type: The type of fuel used by the car. (Petrol, Diesel, Electric, CNG, LPG)
# - Transmission: The type of transmission used by the car. (Automatic / Manual)
# - Owner: Type of ownership
# - Mileage: The standard mileage offered by the car company in kmpl or km/kg
# - Engine: The displacement volume of the engine in CC.
# - Power: The maximum power of the engine in bhp.
# - Seats: The number of seats in the car.
# - New_Price: The price of a new car of the same model in INR Lakhs.(1 Lakh = 100, 000)
# - Price: The price of the used car in INR Lakhs (1 Lakh = 100, 000)

# ### Importing libraries

# In[1]:


import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
# Removes the limit from the number of displayed columns and rows.
# This is so I can see the entire dataframe when I print it
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)


# ### Loading and exploring the data
# 
# In this section the goals are to load the data into python and then to check its basic properties. This will include the dimension, column types and names, and missingness counts.

# In[2]:


data = pd.read_csv("used_cars_data.csv", index_col=0)
print(f'There are {data.shape[0]} rows and {data.shape[1]} columns.')  # f-string


# In[3]:


# Copying dataset to maintain integrity of original
# I'm now going to look at 10 random rows
# I'm setting the random seed via np.random.seed so that
# I get the same random results every time
df = data.copy()
np.random.seed(1)
df.sample(n=10)


# ##### Observations
# Looking at these 10 random rows, we can see there are some columns represented as strings but that we really will want to be numeric. This includes columns like Mileage, Engine, Power, New_Price which need to be turned from strings ending in "kmpl", "CC", "bhp", and "Lakh" respectively into just a number.
# 
# This preview also shows that New_Price column has missing values so we'll want to make sure to look into that.

# In[4]:


df.info()


# ##### Observations
# 
# We see columns (Mileage, Engine, Power, New_Price) that we would expect to have numeric values as strings. We will need to address this.

# - It looks like the column with the greatest number of missing values is New_Price with 6247 missing values. That might be because the used cars in question might have been discontinued and no longer be sold as new models in the market. 
# 
# - We also see a high number of missing variables in the Price column. Price will be treated here as the target variable since it is what we are trying to predict. 
# 
# - Dependent variable is "Price"

# ## Processing columns

# #### Columns containing white spaces
# There are some columns that contain values with white spaces followed by units of measure. For example, the values in 'Mileage' end with space followed by 'kmpl'. Before removing the units of measure, I opted to strip the white space. The user-defined function below does that.

# In[5]:


def strip_space(column_val):
    """For each string column value, strip all white space."""
    if isinstance(column_val, str):
        return str(column_val.replace(" ",""))
    else:
        return np.nan

position_cols = ['Mileage','Engine','Power','New_Price']

for colname in position_cols:
    df[colname] = df[colname].apply(strip_space)


# #### Looking for any unexpected values in Mileage, Engine, Power, New_Price columns

# In[6]:


print(df['Mileage'].value_counts().sample(30))


# In[7]:


print(df['New_Price'].value_counts().sample(30))


# In[8]:


print(df['Power'].value_counts().sort_values(ascending=False))


# ##### Observations
# 
# - We see that Mileage is recorded as both "km/kg" and "kmpl": both units refer to the distance covered (in km) per unit of fuel. So, there is no need to convert between them. We will simply drop the units with no conversion.
# - We can see that Power column has 129 "nullbhp" values which will have to be replaced by NaN.
# - We see also that "New_Price" has "Cr" and "Lakh" as units of measure. As 1 Cr = 100 Lakh, we will have to make the conversion accordingly.

# #### Replacing "nullbhp" values in Power column as NaN

# In[9]:


df['Power'] = df['Power'].replace('nullbhp', np.nan)


# In[10]:


df.Power.isnull().sum()


# ##### Observations
# 
# Power had 46 null values and now, after the replacement of 129 "nullbhp" values, it has correctly a total of 175 null values.

# In[11]:


def unit_drop(column_val):
    """This function takes in a string representing Mileage in kmpl or km/kg, 
    Engine in CC, Power in bhp, and New_Price in Cr and Lakh
    and converts it to a number. In addition, it coverts the Cr value to Lakh 
    by multiplying the Cr value by 100.
    If the input is already numeric, which probably means it's NaN,
    this function just returns np.nan."""
    if isinstance(column_val, str):  # checks if `column_val` is a string
        multiplier = 1 # we will need to use a multiplier to convert Cr into Lakh in New_Price column
        if column_val.endswith('Cr'):
            multiplier = 100
        return float(column_val.replace('kmpl', '').replace('km/kg', '').replace('CC', '').replace('bhp', '').replace('Cr', '').replace('Lakh', '')) * multiplier
    else:  # this happens when the current value is np.nan
        return np.nan

for colname in position_cols:
    df[colname] = df[colname].apply(unit_drop)
    
df[position_cols].head()  #let's see if function worked


# In[12]:


df.info()


# ##### Observations
# We now have Features where we would expect to see numeric values coded, as intended, as float numbers.

# ## Missing values

# ### Examining missing values

# In[13]:


df['Mileage'].describe()


# ##### Observations
# 
# It looks like Mileage (The standard mileage offered by the car company in kmpl or km/kg) has some zero values which we could treat as missing values - no car would have standard mileage equal to zero. We will examine further to determine how many of these values there are and use the previous method of substitute these values by the mean as standard value if needed.

# In[14]:


df1 = df[df['Mileage']==0]


# In[15]:


df1['Mileage'].value_counts()


# ##### Observations
# 
# There are 81 instances in the dataset where standard mileage value is zero.

# In[16]:


df.isnull().sum().sort_values(ascending=False).reset_index()


# In[17]:


#This will give us the number of rows with different number of missing values.
#For example, how many rows have 1 missing value, which the output tells us in this case: 5232
num_missing = df.isnull().sum(axis=1)
num_missing.value_counts()


# In[18]:


df[num_missing == 1].sample(n=5)


# In[19]:


df[num_missing == 2].sample(n=5)


# In[20]:


df[num_missing == 3].sample(n=5)


# In[21]:


df[num_missing == 4].sample(n=5)


# In[22]:


df[num_missing == 5].sample(5)


# In[23]:


for n in num_missing.value_counts().sort_index().index:
    if n > 0:
        print(f'For the rows with exactly {n} missing values, NAs are found in:')
        n_miss_per_col = df[num_missing == n].isnull().sum()
        print(n_miss_per_col[n_miss_per_col > 0])
        print('\n\n')


# ### Handling missing values

# In[24]:


# plotting histogram of all numerical variables
col = ["New_Price", "Power", "Seats", "Engine","Mileage"]
plt.figure(figsize=(17,75))

for i in range(len(col)):
    plt.subplot(18, 3, i+1)
    plt.hist(df[col[i]])
    plt.tight_layout()
    plt.title(col[i], fontsize=25)
    
plt.show()


# In[25]:


#descriptives before any changes
df.describe().T


# ##### Observations
# 
# It looks like Mileage (The standard mileage offered by the car company in kmpl or km/kg) 
# has some zero values which seem nonsensical 
# as no car would have standard mileage equal to zero. 
# Since the distribution of values in Mileage
# tends to normality we will substitute zeros (81 instances as seen in above section) for the mean

# In[26]:


df['Mileage'] = df['Mileage'].replace(0,df['Mileage'].mean())


# In[27]:


df['Mileage'].describe()


# ##### Observations
# 
# The mean and standard deviation of the distribution before and after changing Mileage values equal to zero for the mean are similar, indicating that the change did not alter the distribution in a biasing manner.
# 
# Mileage mean and standard deviation before change:
# - mean       18.141580
# - std         4.562197
# 
# Mileage descriptives after change:
# 
# - mean       18.344238
# - std         4.134674

# 
# 
# The distribution of the Mileage and Seats looks to be centered around the mean more so than New_Price, Engine, and Power which have right-skewed distributions. We will therefore substitude missing values for the mean for Mileage and Seats and for the median for New Price, Engine and Power.

# In[28]:


#using `fillna` with numeric columns Mileage and Seats to fill in missing values with the mean
mean_fillna = ['Mileage','Seats']
for colname in mean_fillna:
    print(df[colname].isnull().sum())
    df[colname].fillna(df[colname].mean(), inplace=True) 
    print(df[colname].isnull().sum())


# In[29]:


#using `fillna` to fill in New Price, Engine, and Power with their median value
mean_fillna2 = ['New_Price','Engine', 'Power']
for colname in mean_fillna2:
    print(df[colname].isnull().sum())
    df[colname].fillna(df[colname].median(), inplace=True) 
    print(df[colname].isnull().sum())


# Substitution for mean and median values successfully completed. Descriptives provided below.

# In[30]:


# Examining distribution after filling missing values
df.describe().T


# ##### Observations
# Comparing means and standard deviations of Mileage, Engine, Power and Seats before and after filling NA values with the means of each feature we see only slight changes to these values, as desired. 
# 
# Change to New Price though, given the large number of missing values (6247), was drastic.
# 
# Mileage
# - Mean, Std Dev (before): 18.142, 4.562
# - Mean, Std Dev (after): 18.344238, 4.134104
# 
# Engine
# - Mean, Std Dev (before): 1616.573, 595.285137
# - Mean, Std Dev (after): 1615.790, 593.475
# 
# Power
# - Mean, Std Dev (before): 112.765214, 53.493553
# - Mean, Std Dev (after): 112.312, 52.923
# 
# Seats
# - Mean, Std Dev (before): 5.280, 0.811660
# - Mean, Std Dev (after): 5.280, 0.809
# 
# New Price
# - Mean, Std Dev (before): 22.780, 27.759
# - Mean, Std Dev (after): 13.125, 11.036

# In[31]:


df.isnull().sum().sort_values(ascending=False)


# We are still left with a large number of missing values in the target variable, *Price*

# In[32]:


# plotting histogram of all numerical variables
col = ["New_Price", "Power", "Seats", "Engine","Mileage"]
plt.figure(figsize=(17,75))

for i in range(len(col)):
    plt.subplot(18, 3, i+1)
    plt.hist(df[col[i]])
    plt.tight_layout()
    plt.title(col[i], fontsize=25)
    
plt.show()


# ##### Observations
# As 6247 out of 7253 values for *New_Price* were imputed with the distribution's median, we observe a reduction/narrowing of the distribution around the median value.

# #### Dropping missing values in the target variable

# As it is generally preferred to impute missing values in the independent variables, we opt to impute missing New_Price values with the median of the distribution, since it is highly skewed.

# In[33]:


#Dropping the remaining missing values which are now only left in the target variable, Price
df = df.dropna()


# In[34]:


df.isnull().sum().sort_values(ascending=False)


# In[35]:


df.info()


# After dropping na values for price new total rows for the dataset: 6019

# ### Extracting car brand from 'Name' column

# We know that brands play a role in determining car prices. Let's pull that valuable information from the data

# In[36]:


# function below takes a string, splits it, and returns the first word
def take_first(name):
    if isinstance(name, str):
        splt = name.split(" ")
        return str(splt[0])

df['Brand'] = df['Name'].apply(take_first)


# In[37]:


#We notice different "Isuzu" written in all caps thereby being read by Python
#as a different brand name. Let's change so that only one brand stands as Isuzu
df["Brand"].replace({"ISUZU": "Isuzu"}, inplace=True)


# In[38]:


#Let's take a look at the frequency of observations for each brand
df["Brand"].value_counts(normalize=True)


# In[39]:


#Number of unique brand values
df.Brand.nunique()


# ##### Observations
# We observe frequency lower than 5% of total observations for 12 brands: Mitsubishi, Mini, Volvo, Porsche, Jeep, Datsun, Isuzu, Force, Smart, Lamborghini, Ambassador, and Bentley. As such, we will consider these brands outliers and will integrate them in the data by examining similar brands in terms of country/region of origin and price level.

# In[40]:


#Let's divide brands into country/region of origin categories 
#(european category further divided into popular and luxury - since luxury brands make such a big portion of this group)

brand_cats = []

for brand in df["Brand"]:
    if brand == 'Maruti' or brand == 'Tata' or brand == 'Mahindra' or brand == 'Ambassador' or brand == 'Force':
        brand_cats.append("Indian")
    elif brand == 'Ford' or brand == 'Chevrolet' or brand == 'Jeep':
        brand_cats.append("American")
    elif brand == 'Hyundai' or brand == 'Honda' or brand == 'Nissan' or brand == 'Toyota' or brand == 'Mitsubishi' or brand == 'Datsun'or brand == 'Isuzu':
        brand_cats.append("Japanese-Korean")
    elif brand == 'Volkswagen' or brand == 'Renault' or brand == 'Skoda' or brand == 'Fiat' or brand == 'Smart':
        brand_cats.append("European_popular")
    else:
        brand_cats.append("European_luxury")
        
df["Brand_cats"] = brand_cats


# In[41]:


df["Brand_cats"].value_counts()


# In[42]:


jp_k = df[(df["Brand_cats"] == "Japanese-Korean")]
indian = df[(df["Brand_cats"] == "Indian")]
euro_lux = df[(df["Brand_cats"] == "European_luxury")]
euro_pop = df[(df["Brand_cats"] == "European_popular")]
american = df[(df["Brand_cats"] == "American")]


# Examining price distributions by brand origin:

# In[43]:


jp_k.groupby("Brand")["Price"].describe()


# In[44]:


indian.groupby("Brand")["Price"].describe()


# In[45]:


euro_lux.groupby("Brand")["Price"].describe()


# In[46]:


euro_pop.groupby("Brand")["Price"].describe()


# In[47]:


american.groupby("Brand")["Price"].describe()


# We examined brand origin and price levels to determine the best matches for brands with low frequency observations. Replacement for low frequency brands as follows
# 
# * Lamborghini' replaced as 'Land Rover' 
# * 'Bentley': 'Jaguar' 
# * 'Smart': 'Fiat' 
# * 'Ambassador': 'Tata'
# * 'Mitsubishi': 'Toyota'
# * 'Isuzu': 'Toyota'
# * 'Datsun':'Nissan'
# * 'Volvo': 'BMW'
# * 'Porsche': 'Jaguar'
# * 'Mini': 'Audi'
# * 'Jeep': 'BMW'
# * 'Force': 'Mahindra'

# In[48]:


df["Brand"].replace({'Lamborghini': 'Land', 
                     'Bentley': 'Jaguar', 
                     'Smart': 'Fiat', 
                     'Ambassador': 'Tata',
                    'Mitsubishi': 'Toyota',
                    'Isuzu': 'Toyota',
                    'Datsun':'Nissan',
                    'Volvo': 'BMW',
                    'Porsche': 'Jaguar',
                    'Mini': 'Audi',
                    'Jeep': 'BMW',
                    'Force': 'Mahindra'}, inplace=True)


# In[49]:


#After replacement we end up with this number of unique brands in the dataset
df.Brand.nunique()


# After replacement we end up with 18 unique brand values 

# In[50]:


#dropping added column
df.drop("Brand_cats", axis=1, inplace=True)


# In[51]:


#dropping added dataframes
del jp_k
del indian
del euro_lux
del euro_pop
del american


# ### Making categoricals into categorical types

# In[52]:


cat_cols = ['Name', 'Location', 'Fuel_Type',
            'Transmission', 'Owner_Type', 'Brand']

for colname in cat_cols:
    df[colname] = df[colname].astype('category')
    
df.info()


# ### Basic Summary Statitics

# In[53]:


pd.set_option(
    "display.float_format", lambda x: "%.3f" % x
)  # to display numbers in digits
df.describe(include="all").T  # quick summary of numeric features


# In[54]:


# looking at value counts for non-numeric features

num_to_display = 15  # defining up to how many values to display
for colname in df.dtypes[df.dtypes == 'category'].index:
    val_counts = df[colname].value_counts(dropna=False)  # "dropna=False" to see NA counts
    print(val_counts[:num_to_display])
    if len(val_counts) > num_to_display:
        print(f'Only displaying first {num_to_display} of {len(val_counts)} values.')
    print('\n\n') # just for more space between 


# ##### Observations:
# 
# - The dataset contains 5 categorical variables
# - The *Location* column has 11 unique values, i.e., data on used car sales is captured for 11 different locations in India. The most popular location of used car sales being Mumbai
# - Diesel is the most popular fuel type
# - Most cars being sold reported in this dataset are manual
# - *Year* ranges from 1998 to 2019.
# - We see a highly right-skewed distribution for *Kilometers_Driven*	
# - Mileage's distribution appears to be close to normal: mean and median are very close.
# - Engine and Power look to be right skewed - we will look at these distributions more closely later on to examine if they are good candidates for log transformation
# - Maruti is the most popular brand in the dataset, followed by Hyundai and Honda

# ### Exploring Numeric Variables through Univariate Analysis

# In[55]:


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


# ### Examining need for data transformations of numeric variables

# #### Kilometers Driven

# In[56]:


df['Kilometers_Driven'].describe()


# In[57]:


histogram_boxplot(df['Kilometers_Driven'])


# ##### Observations
# 
# 'Kilometers_Driven' feature looks to be highly skewed. 
# To handle the skewness and help data 
# behave more like a normal distribution, we will look at different transformations: square root, log and arcsinh log 

# In[58]:


# looking at data distribution when transformed by taking the square root of Kilometers Driven values
histogram_boxplot(np.sqrt(df['Kilometers_Driven']))


# In[59]:


# looking at data distribution when transformed by taking the log of Kilometers Driven values
histogram_boxplot(np.log(df['Kilometers_Driven']))


# In[60]:


# looking at data distribution when transformed by taking the arcsinh log of Kilometers Driven values
histogram_boxplot(np.arcsinh(df['Kilometers_Driven']))


# ##### Observations
# All three transformations have helped handle skewness in the data pertaining to Kilometers Driven, however, square root looked as best toward getting data distribution closest to normality - mean and median more closely alligned than in the log and arcsinh log distributions.

# In[61]:


df['Kilometers_Driven' + '_sqrt'] = np.sqrt(df['Kilometers_Driven'])
df.drop(['Kilometers_Driven'], axis=1, inplace=True)


# In[62]:


print(df['Kilometers_Driven_sqrt'].describe())
histogram_boxplot(df['Kilometers_Driven_sqrt'])


# ##### Observations
# 
# Although we still see the presence of outliers but the distribution alligns much more closely to normality after the transformation.

# #### Mileage

# In[63]:


df['Mileage'].describe()


# In[64]:


# Visually examining the distribution of values in mileage
histogram_boxplot(df['Mileage'])


# ##### Observations
# 
# Mileage looks to be close to normally distributed. No transformations to handle skewness needed.

# #### Engine

# In[65]:


histogram_boxplot(df['Engine'])


# In[66]:


df['Engine'].describe()


# In[67]:


histogram_boxplot(np.sqrt(df['Engine']))
histogram_boxplot(np.log(df['Engine']))
histogram_boxplot(np.arcsinh(df['Engine']))


# ##### Observations
# 
# We see that log and arcsinh log transformations handle skewness in *Engine* very similarly. We will opt to transform each x in Engine by log of x since it is the simpler of the two transformations.

# In[68]:


df['Engine' + '_log'] = np.log(df['Engine'])
df.drop(['Engine'], axis=1, inplace=True)


# In[69]:


print(df['Engine_log'].describe())
histogram_boxplot(df['Engine_log'])


# #### Power

# In[70]:


histogram_boxplot(df['Power'])


# In[71]:


df['Power'].describe()


# ##### Observations
# 
# We see the *Power* distribution as right-skewed, with the presence of large value outliers. To help handle the skeweness and reduce the scale of the data distribution we will examine the impact of square root, log and arcsinh log transformations below

# In[72]:


histogram_boxplot(np.sqrt(df['Power']))
histogram_boxplot(np.log(df['Power']))
histogram_boxplot(np.arcsinh(df['Power']))


# ##### Observations
# 
# We see that log and arcsinh log transformations handle skewness in *Power* very similarly. We will opt to transform each x in Power by log of x since it is the simpler of the two transformations.

# In[73]:


df['Power' + '_log'] = np.log(df['Power'])
df.drop(['Power'], axis=1, inplace=True)


# In[74]:


print(df['Power_log'].describe())


# In[75]:


histogram_boxplot(df['Power_log'])


# #### Seats

# In[76]:


df.Seats.describe()


# In[77]:


df.Seats.value_counts()


# ##### Observations
# 
# The value zero for Seats seems nonsensical, we will treat it as a missing value and replace it with the mean value 5.280

# In[78]:


df['Seats'] = df['Seats'].replace(0,df['Seats'].mean())


# In[79]:


histogram_boxplot(df['Seats'])


# In[80]:


histogram_boxplot(np.sqrt(df['Seats']))
histogram_boxplot(np.log(df['Seats']))
histogram_boxplot(np.arcsinh(df['Seats']))


# ##### Observations
# 
# Transformations did not work to reshape the distribution so we opt to leave Seats feature unaltered. 

# #### New_Price

# In[81]:


histogram_boxplot(df['New_Price'])


# In[82]:


df['New_Price'].describe()


# In[83]:


histogram_boxplot(np.sqrt(df['New_Price']))
histogram_boxplot(np.log(df['New_Price']))
histogram_boxplot(np.arcsinh(df['New_Price']))


# ##### Observations
# 
# Given the narrowness of the New_Price data distribution (remembering that most values were imputed with the median to handle missing data) we will opt not to handle the feature with transformations.

# ### Examining Numerical Variables through Histograms

# In[84]:


# plotting histogram of all numerical variables


all_col = df.select_dtypes(include=np.number).columns.tolist()
all_col.remove("Year")
plt.figure(figsize=(17,75))

for i in range(len(all_col)):
    plt.subplot(18, 3, i+1)
    plt.hist(df[all_col[i]])
    plt.tight_layout()
    plt.title(all_col[i], fontsize=25)
    
plt.show()


# #### Observations

# - Mileage looks to be normally distributed
# - Seats, New_Price, Price, Kilometers Driven look to be right skewed
# - The log transformations of Engine and Power managed to make these distributions more normally distributed

# ## Outlier Treatment for Continuous Variables

# Any value that is smaller than Q1 - 1.5IQR and greater than Q3 + 1.5IQR is defined as an outlier. We will treat outliers in the dataset by flooring and capping.
# 
# - All the values smaller than Q1 - 1.5IQR will be assigned the value equal to Q1 - 1.5IQR (flooring)
# - All the values greater than Q3 + 1.5IQR will be assigned the value equal to Q3 + 1.5IQR (capping)

# In[85]:


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

# In[86]:


numerical_col = df.select_dtypes(include=np.number).columns.tolist()
numerical_col.remove("Year")
df = treat_outliers_all(df, numerical_col)


# In[87]:


# let's look at box plot to see if outliers have been treated or not
plt.figure(figsize=(20, 30))

for i, variable in enumerate(numerical_col):
    plt.subplot(5, 4, i + 1)
    plt.boxplot(df[variable], whis=1.5)
    plt.tight_layout()
    plt.title(variable)

plt.show()


# In[88]:


print("Value counts for Seats variable:","\n", df["Seats"].value_counts())
print("\n")
print("Descriptives for Seats variable:","\n", df["Seats"].describe())


# In[89]:


sns.distplot(df["Seats"], kde=False);


# In[90]:


print("Value counts for New_Price variable:","\n", df["New_Price"].value_counts())
print("\n")
print("Descriptives for New_Price variable:","\n", df["New_Price"].describe())


# In[91]:


sns.distplot(df["New_Price"], kde=False);


# ##### Observations
# 
# After treatment for missing values and outliers, the features *Seats* and *New_Price* ended up with very narrow distributions which render these inadequate for predictions. They will therefore be excluded from the model.
# 
# - Seats: out of 6019 possible values, 83% have 5 as number of seats.
# - New_Price: out of 6019 possible values, 86% have price as 11.570.

# ### Looking at Correlations

# In[92]:


col = ["Mileage", "Engine_log", "Power_log", "Kilometers_Driven_sqrt", "Price"]
corr = df[col].corr().sort_values(by=['Price'], ascending=False)

f, ax = plt.subplots(figsize=(28,15))

sns.heatmap(corr, cmap='seismic', annot=True, fmt=".1f", vmin=-1,vmax=1,center=0,square=False, linewidth=0.7, cbar_kws={"shrink": 0.5},)


# #### Observations

# - The highest correlations with Price observed were for Engine (0.7) and Power (0.8) - both correlated positively with Price
# - We observe negative correlations between Price and Mileage and Price and Kilometers Driven
# - Strong correlation present between Power and Engine: 0.9
# 

# ### Examining Bivariate Relationships through Scatterplots

# In[93]:


plt.figure(figsize=(10,7))
sns.scatterplot(y="Price", x="Power_log", data=df);


# In[94]:


plt.figure(figsize=(10,7))
sns.scatterplot(y="Price", x="Engine_log", data=df);


# In[95]:


plt.figure(figsize=(10,7))
sns.scatterplot(y="Price", x="Mileage", data=df);


# In[96]:


plt.figure(figsize=(10,7))
sns.scatterplot(y="Price", x="Kilometers_Driven_sqrt", data=df);


# ##### Observations
# 
# * The graphs above indicate a positive relationship between Engine (the displacement volume of the engine) and Price as well as between Power (the maximum power of the engine) and Price
# 
# * The scatterplots for Mileage and Kilometers Driven indicate there is not a strong correlation between these and the target variable Price

# In[97]:


# Function to create barplots that indicate count.
def count_on_bar(data, z):
    """
    plot
    feature: categorical feature
    the function won't work if a column is passed in hue parameter
    """

    total = len(data[z])  # length of the column
    plt.figure(figsize=(15, 5))
    plt.xticks(rotation=45)
    ax = sns.countplot(data=data, x=z, palette="Paired")
    for p in ax.patches:
        percentage = p.get_height()  # count of each class of the category
        x = p.get_x() + p.get_width() / total + 0.2  # width of the plot
        y = p.get_y() + p.get_height()  # height of the plot

        ax.annotate(percentage, (x, y), size=12)  # annotate the percentage
    plt.show()  # show the plot


# In[98]:


df.groupby("Year")["Price"].describe()


# In[99]:


count_on_bar(df, "Year")


# In[100]:


plt.figure(figsize=(25, 10))

plt.subplot(1, 2, 1)
sns.barplot(data=df, y="Price", x="Year")
plt.xticks(rotation=45)

plt.show()


# ##### Observations
# 
# * The oldest cars reported in the dataset were built in 1998, the newest in 2019
# 
# * We observe an increasing volume of used cars for every new year from 1998 to 2014 and a downward trend from 2015 to 2019
# 
# * In terms of price, as expected, the clear trend observed is for newer cars to be sold at increasingly higher price points 

# # Examining categorical variables

# In[101]:


#creating a list of the categorical features
cat_cols = []
for colname in df.columns[df.dtypes == 'category']:  # only need to consider string columns
    cat_cols.append(colname)

print(cat_cols) 


# #### Location

# In[102]:


df.groupby("Location")["Price"].describe()


# In[103]:


count_on_bar(df, "Location")


# In[104]:


plt.figure(figsize=(15, 5))

plt.subplot(1, 2, 1)
sns.barplot(data=df, y="Price", x="Location")
plt.xticks(rotation=45)

plt.subplot(1, 2, 2)
sns.boxplot(data=df, y="Price", x="Location")
plt.xticks(rotation=45)

plt.show()


# ##### Observations
# 
# * It looks like most sales have been reported from Mumbai and Hyderabad
# * We also observe the lowest used car prices in Kolkata and Jaipur (means and mid 50% price distributions at the lower end of the scale).
# * We see the widest range of prices in Bangalore and Coimbatore, the highest price means observed in these two cities as well.

# #### Fuel Type

# In[105]:


df.groupby("Fuel_Type")["Price"].mean().sort_values(ascending=False)


# In[106]:


count_on_bar(df, "Fuel_Type")


# In[107]:


df.Fuel_Type.value_counts(normalize=True, ascending=True)


# We could consider here Electric, CNG, and LPG outlier categories given their very low frequency. We will then substitute these for the most popular fuel type: Diesel.

# In[108]:


df.groupby("Fuel_Type")["Price"].describe()


# In[109]:


df["Fuel_Type"] = df["Fuel_Type"].str.upper()
not_petrol = []

df["Fuel_Type"].replace({"CNG": "DIESEL", "LPG": "DIESEL", "ELECTRIC": "DIESEL"}, inplace=True)

df["Fuel_Type"] = df["Fuel_Type"].str.title()


# In[110]:


count_on_bar(df, "Fuel_Type")


# In[111]:


plt.figure(figsize=(15, 5))

plt.subplot(1, 2, 1)
sns.barplot(data=df, y="Price", x="Fuel_Type")
plt.xticks(rotation=45)

plt.subplot(1, 2, 2)
sns.boxplot(data=df, y="Price", x="Fuel_Type")
plt.xticks(rotation=45)

plt.show()


# ##### Observations
# 
# * The most popular fuel type is Diesel followed by Petrol*
# * Petrol cars have a lower mean price (5.14) as well as a narrower range of mid 50% distributions with presence of outliers on the high/left end of the distribution
# * The IQR range for the Diesel distribution is wider and we observe a mean price, 9.86, considerably higher than Petrol cars

# #### Transmission

# In[112]:


df.groupby("Transmission")["Price"].describe()


# In[113]:


count_on_bar(df, "Transmission")


# In[114]:


plt.figure(figsize=(15, 5))

plt.subplot(1, 2, 1)
sns.barplot(data=df, y="Price", x="Transmission")
plt.xticks(rotation=45)

plt.subplot(1, 2, 2)
sns.boxplot(data=df, y="Price", x="Transmission")
plt.xticks(rotation=45)

plt.show()


# ##### Observations
# 
# With 71.4% of cars in the dataset, the most popular transmission type is Manual
# 
# Manual cars are considerably cheaper than Automatic, with a mean price of 5.29 compared to 13.75
# 
# The IQR price range for Automatic cars is wider and spans a higher range of prices, 7.97 for Q1 and 19.63 for Q3
# 
# Manual car prices’ IQR range: Q1 = 3, Q3 = 6.55. We also observe the presence of outliers: manual cars with prices above the top whisker value (Q3 + 1.5 IQR) of 11.86

# #### Owner Type

# In[115]:


df.groupby("Owner_Type")["Price"].describe()


# In[116]:


count_on_bar(df, "Owner_Type")


# In[117]:


df.Owner_Type.value_counts(normalize=True, ascending=True)


# ##### Observations
# 82% of used cars in this sample are classified as "First" owner_type. To account for the skewness here, we can split the category into 2 types, First and Second & Above. This will not solve the skewness in the distribution of values in the category but will attenuate it.

# In[118]:


df["Owner_Type"] = df["Owner_Type"].str.upper()

df["Owner_Type"].replace({"SECOND": "SECOND & ABOVE", "THIRD": "SECOND & ABOVE", "FOURTH & ABOVE": "SECOND & ABOVE"}, inplace=True)

df["Owner_Type"] = df["Owner_Type"].str.title()


# In[119]:


df.groupby("Owner_Type")["Price"].describe()


# In[120]:


count_on_bar(df, "Owner_Type")


# In[121]:


df.Owner_Type.value_counts(normalize=True, ascending=True)


# In[122]:


plt.figure(figsize=(15, 5))

plt.subplot(1, 2, 1)
sns.barplot(data=df, y="Price", x="Owner_Type")
plt.xticks(rotation=45)

plt.subplot(1, 2, 2)
sns.boxplot(data=df, y="Price", x="Owner_Type")
plt.xticks(rotation=45)

plt.show()


# ##### Observations
# * Nearly 81% of cars in the dataset reported to have been owned previously by only one owner (“First”)
# 
# * As expected, cars sold after First owner have a mean price higher than cars sold after Second & Above, 8.04 compared to 6.19
# 
# * We observe the presence of price outliers among the Second & Above group

# In[123]:


df.groupby("Brand")["Price"].describe()


# In[124]:


count_on_bar(df, "Brand")


# In[125]:


plt.figure(figsize=(30, 15))

plt.subplot(1, 2, 1)
sns.barplot(data=df, y="Price", x="Brand")
plt.xticks(rotation=45)

plt.show()


# ##### Observations:
# 
# Hyundai and Maruti are the most popular brands
# - European luxury brands like Land Rover, Jaguar, and Audi show up as the most
# expensive
# - Chevrolet has the lowest mean price (3.04 Lakh), followed by Fiat (3.26) and Tata (3.55)

# In[126]:


df.Fuel_Type = df.Fuel_Type.astype('category')
df.Owner_Type = df.Owner_Type.astype('category')


# In[127]:


df.info()


# ## Creating Dummy Variables

# In[128]:


df.Name.nunique()


# In[129]:


#creating a list of the categorical features
cat_cols = []
for colname in df.columns[df.dtypes == 'category']:  # only need to consider string columns
    cat_cols.append(colname)

cat_cols.remove("Name")
cat_cols.remove("Brand")
print(cat_cols)


# In[130]:


df = pd.get_dummies(df,columns=cat_cols,drop_first=True)


# In[131]:


df = pd.get_dummies(df,columns=["Brand"],drop_first=False) # will drop one of the dummies manually - see below


# ##### Observations
# 
# We observe too large of a number of unique values for the Name category to encode into dummy variables. We will use one-hot encoding for the remaining categorical variables to assist us in model building. 
# 
# Note that no dummy has yet been dropped for the ones created for Brand variable. We will manually drop one of the dummies, the one with the lowest mean price ("Brand_Chevrolet"), to help later on with the interpretation of the linear equation coefficients.

# ## Data Preparation for Modeling

# In[132]:


# defining X and y variables
#choosing to drop manually brand chevrolet to help with model interpretation since Chevrolet has the lowest
#mean price of all the brands
X = df.drop(["Price", "Name", "New_Price", "Seats", "Brand_Chevrolet"], axis=1) 
y = df[["Price"]]

print(X.head())
print(y.head())


# In[133]:


print(X.shape)
print(y.shape)


# In[134]:


# split the data into train and test
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)


# In[135]:


X_train.head()


# # <a id='link4'>Choose, train and evaluate the model</a>

# In[136]:


# To build linear regression_model
from sklearn.linear_model import LinearRegression

# To check model performance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

#additional libraries to be used
import statsmodels.stats.api as sms
from statsmodels.compat import lzip
import statsmodels.api as sm


# In[137]:


# fitting the model on the train data (70% of the whole data)
linearregression = LinearRegression()
linearregression.fit(X_train, y_train)


# In[138]:


# let us check the coefficients and intercept of the model

coef_df = pd.DataFrame(
    np.append(linearregression.coef_[0], linearregression.intercept_[0]),
    index=X_train.columns.tolist() + ["Intercept"],
    columns=["Coefficients"],
)

coef_df


# **Let's check the performance of the model using different metrics (MAE, MAPE, RMSE, $R^2$).**
# 
# * We will be using metric functions defined in sklearn for RMSE, MAE, and $R^2$.
# * We will define a function to calculate MAPE.
# * We will create a function which will print out all the above metrics in one go.

# In[139]:


# defining function for MAPE
def mape(targets, predictions):
    return np.mean(np.abs((targets - predictions)) / targets) * 100


# defining common function for all metrics
def model_perf(model, inp, out):
    """
    model: model
    inp: independent variables
    out: dependent variable
    """
    y_pred = model.predict(inp).flatten()
    y_act = out.values.flatten()

    return pd.DataFrame(
        {
            "MAE": mean_absolute_error(y_act, y_pred),
            "MAPE": mape(y_act, y_pred),
            "RMSE": np.sqrt(mean_squared_error(y_act, y_pred)),
            "R^2": r2_score(y_act, y_pred),
        },
        index=[0],
    )


# In[140]:


# Checking model performance on train set (seen 70% data)
print("Train Performance\n")
model_perf(linearregression, X_train, y_train)


# In[141]:


# Checking model performance on test set (unseen 30% data)
print("Test Performance\n")
model_perf(linearregression, X_test, y_test)


# - The mean absolute percentage error (MAPE) measures the accuracy of predictions as a percentage, and can be calculated as the average absolute percent error for each predicted value minus actual values divided by actual values. It works best if there are no extreme values in the data and none of the actual values are 0.
# 
# 
# - The mean absolute error (MAE) is the simplest regression error metric to understand. We'll calculate the residual for every data point, taking only the absolute value of each so that negative and positive residuals do not cancel out. We then take the average of all these residuals. Effectively, MAE describes the typical magnitude of the residuals.
# 
# 
# - The root mean square error (RMSE) is just like the MAE, but squares the difference before summing them all instead of using the absolute value, and then takes the square root of the value.
# 
# 
# - $R^2$ (or coefficient of determination) represents the proportion of variance (of y) that has been explained by the independent variables in the model. It provides an indication of goodness of fit of the model. Best possible score is 1.0 and it can be negative (because the model can be arbitrarily worse). A constant model that always predicts the expected value (or mean) of y, disregarding the input features, would get a $R^2$ of 0.0.
# 
# 
# **Observations**
# 
# - The training and testing scores are 89.3% and 88.6% respectively, and both the scores are comparable. Hence, the model is a good fit.
# 
# - R-squared is 0.886 on the test set, i.e., the model explains 88.6.3% of total variation in the test dataset. So, overall the model is very satisfactory.
# 
# - MAE indicates that our current model is able to predict price within a mean error of 1.410 Lakh on the test data.
# 
# - MAPE on the test set suggests we can predict within 32.78% of the price.

# # <a id='link5'>Linear Regression using statsmodels</a>

# - We have build a linear regression model which shows good performance on the train and test sets.
# 
# - But this is **not** the final model.
# 
# - We have to check the statistical validity of the model, and also make sure it satisfies the assumptions of linear regression.
# 
# - We will now perform linear regression using *statsmodels*, a Python module that provides functions for the estimation of many statistical models, as well as for conducting statistical tests, and statistical data exploration.
# 
# - Using statsmodels, we will be able to check the statistical validity of the model.

# In[142]:


# Let's build linear regression model using statsmodel
# unlike sklearn, statsmodels does not add a constant to the data on its own
# we have to add the constant manually
X = sm.add_constant(X)
X_train1, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

olsmod1 = sm.OLS(y_train, X_train1)
olsres1 = olsmod1.fit()
print(olsres1.summary())


# ## Interpreting the Regression Results:
# 
# 1. **Adjusted. R-squared**: It reflects the fit of the model.
#     - Adjusted R-squared values generally range from 0 to 1, where a higher value generally indicates a better fit, assuming certain conditions are met.
#     - In our case, the value for Adj. R-squared is **0.892**, which is good
# 
# 
# 2. ***const* coefficient**: It is the Y-intercept.
#     - It means that if all the independent variable coefficients are zero, then the expected output (i.e., Y) would be equal to the *const* coefficient.
# 
# 
# 
# 3. **Coefficient of an independent variable**: It represents the change in the output Y due to a change in the independent variable (everything else held constant).
# 
# 
# 5. **std err**: It reflects the level of accuracy of the coefficients.
#     - The lower it is, the higher is the level of accuracy.
# 
# 
# 6. **P>|t|**: It is p-value.
#    
#     * For each independent feature, there is a null hypothesis and an alternate hypothesis. Here $\beta_i$ is the coefficient of the $i$th independent variable.
# 
#         - $H_o$ : Independent feature is not significant ($\beta_i = 0$)
#         - $H_a$ : Independent feature is that it is significant ($\beta_i \neq 0$)
# 
#     * (P>|t|) gives the p-value for each independent feature to check that null hypothesis. We are considering 0.05 (5%) as significance level.
#         
#         - A p-value of less than 0.05 is considered to be statistically significant.
# 
# 
# 7. **Confidence Interval**: It represents the range in which our coefficients are likely to fall (with a likelihood of 95%).

# **Observations**
# 
# - Negative values of the coefficient show that *Price* decreases with the increase of corresponding attribute value.
# 
# - Positive values of the coefficient show that *Price* increases with the increase of corresponding attribute value.
# 
# - p-value of a variable indicates if the variable is significant or not. If we consider the significance level to be 0.05 (5%), then any variable with a p-value less than 0.05 would be considered significant.
# 
# - But these variables might contain multicollinearity, which will affect the p-values.
# 
# - So, we need to deal with multicollinearity and check the other assumptions of linear regression first, and then look at the p-values.

# # <a id='link6'>Checking Linear Regression Assumptions</a>

# We will be checking the following Linear Regression assumptions:
# 
# **1. No Multicollinearity**
# 
# **2. Mean of residuals should be 0**
# 
# **3. No Heteroscedasticity**
# 
# **4. Linearity of variables**
# 
# **5. Normality of error terms**

# ### TEST FOR MULTICOLLINEARITY
# 
# * Multicollinearity occurs when predictor variables in a regression model are correlated. This correlation is a problem because predictor variables should be independent. If the correlation between variables is high, it can cause problems when we fit the model and interpret the results. When we have multicollinearity the linear model, The coefficients that the model suggests are unreliable.
# 
# * There are different ways of detecting (or testing) multicollinearity. One such way is by using Variation Inflation Factor.
# 
# * **Variance  Inflation  factor**:  Variance inflation factors measure the inflation in the variances of the regression parameter estimates due to collinearities that exist among the  predictors.  It  is  a  measure  of  how  much  the  variance  of  the  estimated  regression coefficient $\beta_k$ is "inflated" by  the  existence  of  correlation  among  the  predictor variables in the model. 
# 
# * **General Rule of thumb**: If VIF is 1 then there is no correlation among the $k$th predictor and the remaining predictor variables, and  hence, the variance of $\beta_k$ is not inflated at all. Whereas, if VIF exceeds 5 or is close to exceeding 5, we say there is moderate multicollinearity, and if it is 10 or exceeding 10, it shows signs of high multicollinearity.

# In[143]:


from statsmodels.stats.outliers_influence import variance_inflation_factor

vif_series1 = pd.Series(
    [variance_inflation_factor(X.values, i) for i in range(X.shape[1])], index=X.columns
)
print("VIF Scores: \n\n{}\n".format(vif_series1))


# #### Observartions

# * VIF for Engine_log is higher than 10 (VIF: 11.269) so we will drop it from the model and examine the results

# ### Removing Multicollinearity
# 
# To remove multicollinearity
# 
# 1. Drop every column one by one that has VIF score greater than 5.
# 2. Look at the adjusted R-squared of all these models.
# 3. Drop the variable that makes least change in adjusted R-squared.
# 4. Check the VIF scores again.
# 5. Continue till you get all VIF scores under 5.

# In[144]:


# we drop the one with the highest vif values and check the adjusted R-squared
X_train2 = X_train1.drop("Engine_log", axis=1)
vif_series2 = pd.Series(
    [variance_inflation_factor(X_train2.values, i) for i in range(X_train2.shape[1])],
    index=X_train2.columns,
)
print("VIF Scores: \n\n{}\n".format(vif_series2))


# * That seemed to have helped, VIF has come down to quite a good limit, now we can say features are not correlated.

# In[145]:


olsmod2 = sm.OLS(y_train, X_train2)
olsres2 = olsmod2.fit()
print(olsres2.summary())


# ##### Observations
# 
# * Earlier adj. R-squared was 0.892, now it is 0.892 and multicolinearity resolved. Now let's look at features with p-value greater than 0.05 and remove them from the model
# 
# 

# - *Location_Chennai*, *Location_Jaipur*, *Location_Pune*, *Location_Kochi*, *Brand_Fiat*, and *Brand_Tata* have p-value greater than 0.05, so they are not significant, we'll drop them.
# 
# - But sometimes p-values change after dropping a variable. So, we'll not drop all variables at once, instead will drop them one by one, starting with the feature with the highest p-value
# 
# - After first feature is dropped we will re-evaluate and drop the next highest as needed, until only features with p-value lower than 0.05 remain in the model

# **Let's drop the feature *Location_Jaipur* since it has highest p-value among all with p-value greater than 0.05**

# In[146]:


X_train3 = X_train2.drop("Location_Jaipur", axis=1)


# In[147]:


olsmod3 = sm.OLS(y_train, X_train3)
olsres3 = olsmod3.fit()
print(olsres3.summary())


# Next with highest p-value over 0.05 is *Location_Chennai* (p-value: 0.787) so we'll drop it.

# In[148]:


X_train4 = X_train3.drop("Location_Chennai", axis=1)


# In[149]:


olsmod4 = sm.OLS(y_train, X_train4)
olsres4 = olsmod4.fit()
print(olsres4.summary())


# Next with highest p-value over 0.05 is *Brand_Tata* (p-value: 0.550) so we'll drop it.

# In[150]:


X_train5 = X_train4.drop("Brand_Tata", axis=1)


# In[151]:


olsmod5 = sm.OLS(y_train, X_train5)
olsres5 = olsmod5.fit()
print(olsres5.summary())


# Next with highest p-value over 0.05 is *Brand_Fiat* (p-value: 0.313) so we'll drop it.

# In[152]:


X_train6 = X_train5.drop("Brand_Fiat", axis=1)


# In[153]:


olsmod6 = sm.OLS(y_train, X_train6)
olsres6 = olsmod6.fit()
print(olsres6.summary())


# ##### Observations
# 
# **Now no feature has p-value greater than 0.05, so we'll consider the features in *X_train6* as the final ones and *olsres6* as final model.**

# * Adjusted R-squared is 0.892, i.e., our model is able to explain 89.2% of variance. This shows that the model is good.
# * The adjusted R-squared in *olsres1* (where we considered all the variables) was 0.892. This shows that the variables dropped were not affecting the model much.

# ### Now we'll check the rest of the assumptions on model *olsres6*
# 
# 2. Mean of residuals should be 0 
# 3. Linearity of variables
# 4. Normality of error terms
# 5. No Heteroscedasticity

# ### MEAN OF RESIDUALS SHOULD BE 0

# In[154]:


residual = olsres6.resid
np.mean(residual)


# * Mean of redisuals is very close to 0.

# ### TEST FOR LINEARITY 
# 
# **Why the test?**
# * Linearity describes a straight-line relationship between two variables, predictor variables must have a linear relation with the dependent variable.
# 
# **How to check linearity?**
# 
# * Make a plot of fitted values vs residuals, if they don't follow any pattern, they we say the model is linear, otherwise model is showing signs of non-linearity.
# 
# **How to fix if this assumption is not followed?**
# 
# * We can try to transform the variables and make the relationships linear.

# In[155]:


residual = olsres6.resid
fitted = olsres6.fittedvalues  # predicted values


# In[156]:


sns.set_style("whitegrid")
sns.residplot(fitted, residual, color="orange", lowess=True)
plt.xlabel("Fitted Values")
plt.ylabel("Residuals")
plt.title("Residual vs Fitted plot")
plt.show()


# * Scatter plot shows the distribution of residuals (errors) vs fitted values (predicted values).
# 
# * If there exist any pattern in this plot, we consider it as signs of non-linearity in the data and a pattern means that the model doesn't capture non-linear effects.
# 
# * **The relationship between residuals and fitted values is not entirely clear as the graph below fails to display a pattern. Hence, the assumption is satisfied.**

# ### TEST FOR NORMALITY
# 
# **What is the test?**
# 
# * Error terms/Residuals should be normally distributed
# 
# * If the error terms are non- normally distributed, confidence intervals may become too wide or narrow. Once confidence interval becomes unstable, it leads to difficulty in estimating coefficients based on minimization of least squares.
# 
# **What do non-normality indicate?**
# 
# * It suggests that there are a few unusual data points which must be studied closely to make a better model.
# 
# **How to Check the Normality?**
# 
# * It can be checked via QQ Plot, Residuals following normal distribution will make a straight line plot otherwise not.
# 
# * Other test to check for normality : Shapiro-Wilk test.
# 
# **What is the residuals are not-normal?**
# 
# * We can apply transformations like log, exponential, arcsinh, etc. as per our data.

# In[157]:


sns.distplot(residual)
plt.title("Normality of residuals")
plt.show()


# #### The QQ plot of residuals can be used to visually check the normality assumption. The normal probability plot of residuals should approximately follow a straight line.

# In[158]:


import pylab
import scipy.stats as stats

stats.probplot(residual, dist="norm", plot=pylab)
plt.show()


# In[159]:


stats.shapiro(residual)


# * The residuals are not normal as per shapiro test, but as per QQ plot they are approximately normal.
# * The issue with shapiro test is when dataset is big, even for small deviations, it shows data as not normal.
# * Hence we go with the QQ plot and say that residuals are normal.

# ### TEST FOR HOMOSCEDASTICITY
# 
# * Test - goldfeldquandt test
# 
# * **Homoscedacity**: If the variance of the residuals are symmetrically distributed across the regression line, then the data is said to homoscedastic.
# 
# * **Heteroscedacity**: If the variance is unequal for the residuals across the regression line, then the data is said to be heteroscedastic. In this case the residuals can form an arrow shape or any other non symmetrical shape.

# For goldfeldquandt test, the null and alternate hypotheses are as follows:
# 
# - Null hypothesis : Residuals are homoscedastic
# - Alternate hypothesis : Residuals have heteroscedasticity

# In[160]:


import statsmodels.stats.api as sms
from statsmodels.compat import lzip

name = ["F statistic", "p-value"]
test = sms.het_goldfeldquandt(residual, X_train6)
lzip(name, test)


# **Since p-value > 0.05, we can say that residuals are homoscedastic. This assumption is therefore valid in the data.**

# **Now we have checked all the assumptions and they are satisfied, so we can move towards the prediction part.**

# ### Predicting on the test data

# In[161]:


X_train6.columns


# In[162]:


# Selecting columns from test data that we used to create our final model
X_test_final = X_test[X_train6.columns]


# In[163]:


X_test_final.head()


# In[164]:


# Checking model performance on train set
print("Train Performance\n")
model_perf(olsres6, X_train6.values, y_train)


# In[165]:


# Checking model performance on test set
print("Test Performance\n")
model_perf(olsres6, X_test_final.values, y_test)


# ##### Observations
# 
# * Now we can see that the model has low test and train RMSE and MAE, and both the errors are comparable. So, the model is not suffering from overfitting.
# 
# * The model is able to explain 88.6% of the variation on the test set, which is very good.
# 
# * The MAPE on the test set suggests we can predict within 32.7% of price. 
# 
# 
# **Hence, we can conclude *olsres6* is a good predictive model.**

# In[166]:


# let us print the model summary

olsmod6 = sm.OLS(y_train, X_train6)
olsres6 = olsmod6.fit()
print(olsres6.summary())


# # <a id='link7'>Conclusions</a>

# ***olsres6* is our final model which follows all the assumptions, and can be used for interpretations.**
# 
# 1. Brand turns out to be a significant predictor of used car prices. The coefficients (slope values of the regression line) for each brand compare to the base line brand Chevrolet (the brand with the lowest mean price of all the brands in the dataset). According to the model, we see that, for instance, compared to Chevrolet, an Audi used car will result in a market price on average 7.80 Lakh higher
# 
# 
# 2. Power is another significant predictor, 1 unit increase in power value (the maximum power of the engine in bhp) leads to an average increase in price by 4.28 Lakh
# 
# 
# 3. According to the model, fuel and transmission type also influence price. Petrol cars tend to be cheaper than Diesel such that running on Petrol, the model predicts, will lead to an average decrease of 1.97 Lakh in the price of the vehicle compared to Diesel fuel type
# 
# 
# 4. Manual cars tend to be cheaper than automatic: a manual transmission car will result in a car on average 1.27 Lakh cheaper compared to automatic
# 
# 
# 5. Location also has an influence on used car prices. Compared to the baseline city of Ahmedabad, a car sold in Kolkata is predicted to be 1.48 Lakh cheaper, for example
# 
# 
# 6. Year has a positive coefficient: newer used cars result in higher sale prices
# 
# 
# 7. Cars that are more mileage efficient then to cost less than more powerful but also more fuel consuming cars
# 
# 
# 8. As expected, cars owned by more than one previous owner tend to cost less than cars owned previously by only one owner
# 
# 
# 9. Kilometers driven also plays a role in predicting used car prices - the more kilometers driven the cheaper the car tends to be

# # <a id='link7'>Insights and Recommendations</a>

# Our used car pricing model points to the importance of the following:
# 
# 
# * Power: the maximum power of the car engine - the more power, the more costly the car
# * Car brand: overall, used cars by luxury brands tend to maintain their high status and be sold at a premium compared to more popular brands
# * Location: used car prices are influenced by the location where cars are sold
# * Car fuel type: Petrol cars are cheaper than cars run on Diesel
# * Mileage: more fuel efficient cars, that run more km on less fuel (higher kmpl), tend to cost less than cars with lower kmpl mileage
# * Car transmission: used cars with manual transmission tend to cost less than automatic
# * Ownership: used cars previously owned by more than one owner tend to cost less
# * Year: not surprisingly though vouched for by the model, newer cars tend to be sold at higher prices
# * Kilometers driven: kilometers driven also plays a role in predicting used car prices - the more kilometers driven the cheaper the car tends to be
# 
# 
# 
# 
# Every car has a collection of characteristics which make it unique. We only considered a subset of these in our model. Nonetheless, we were able to determine several important predictors that influence used car prices and should be leveraged by Cars4U in its pricing strategy. 
# 
# 
# 
# Given the model’s predictors we believe that an effective pricing strategy should:
# 
# 
# 
# 1. Take in consideration several characteristics of a car, namely - as tested by this model - its engine power, brand, transmission type, ownership history, fuel efficiency, make year, and kilometers driven to come up with a price that would be fair and not below market value
# 
# 
# 2. Be adapted to the car’s location of sale
# 
# 
# 3. Use the model’s predictors as leverage in developing special sales strategies - especially in market locations in which car prices seem to be inflated or others in which perhaps due to more fierce competition cars seem to be sold below market value - with a clever location buy/sell strategy, for example, Cars4U could leverage such price fluctuations and sell cars at optimum prices 
# 
