#!/usr/bin/env python
# coding: utf-8

# #### Code and analysis by Gracieli Scremin
# 
# ## Description
# 
# #### Context
# AllLife Bank wants to focus on its credit card customer base in the next financial year. They have been advised by their marketing research team, that the penetration in the market can be improved. Based on this input, the Marketing team proposes to run personalized campaigns to target new customers as well as upsell to existing customers. Another insight from the market research was that the customers perceive the support services of the back poorly. Based on this, the Operations team wants to upgrade the service delivery model, to ensure that customer queries are resolved faster. Head of Marketing and Head of Delivery both decide to reach out to the Data Science team for help
# 
#  
# 
# #### Objective
# To identify different segments in the existing customer, based on their spending patterns as well as past interaction with the bank, using clustering algorithms, and provide recommendations to the bank on how to better market to and service these customers.
# 
#  
# 
# #### Data Description
# The data provided is of various customers of a bank and their financial attributes like credit limit, the total number of credit cards the customer has, and different channels through which customers have contacted the bank for any queries (including visiting the bank, online and through a call center).
# 
# 
# #### Data Dictionary
# * Sl_No: Primary key of the records
# * Customer Key: Customer identification number
# * Average Credit Limit: Average credit limit of each customer for all credit cards
# * Total credit cards: Total number of credit cards possessed by the customer
# * Total visits bank: Total number of Visits that customer made (yearly) personally to the bank
# * Total visits online: Total number of visits or online logins made by the customer (yearly)
# * Total calls made: Total number of calls made by the customer to the bank or its customer service department (yearly)

# ### Importing libraries

# In[1]:


# this will help in making the Python code more structured automatically
#%load_ext nb_black

# Library to suppress warnings or deprecation notes
import warnings

warnings.filterwarnings("ignore")

# Libraries to help with reading and manipulating data
import numpy as np
import pandas as pd

# Libraries to help with data visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Removes the limit for the number of displayed columns
pd.set_option("display.max_columns", None)
# Sets the limit for the number of displayed rows
pd.set_option("display.max_rows", 200)

# to scale the data using z-score
from sklearn.preprocessing import StandardScaler

# to perform k-means clustering and compute silhouette scores
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# to visualize the elbow curve and silhouette scores
from yellowbrick.cluster import KElbowVisualizer, SilhouetteVisualizer

# to compute distances
from scipy.spatial.distance import pdist

# to compute distances
from scipy.spatial.distance import cdist

# to perform hierarchical clustering, compute cophenetic correlation, and create dendrograms
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage, cophenet

# to perform PCA
from sklearn.decomposition import PCA

# to time algorithms
import time


# In[2]:


data = pd.read_excel("Credit Card Customer Data.xlsx")


# In[3]:


data.shape


# #### Observation
# * The data has 660 rows and 7 columns

# In[4]:


data.info()


# #### Observations
# * The table summarizing dataframe info above indicates there are no missing values in dataset and all column data types are numerical
# * Sl_No and Customer Key are id columns - these will need to be excluded from clustering process

# In[5]:


# making a copy of the original dataset in order to maintain original intact
df = data.copy()


# In[6]:


# viewing a random sample of the dataset
df.sample(n=10, random_state=1)


# #### Observation
# * Dataset has two identifier columns ("Sl_No" and "Customer Key") which are not suitable for model building purposes. The columns will therefore be dropped

# In[236]:


# dropping ID columns
df.drop(["Sl_No", "Customer Key"], axis=1, inplace=True)


# In[237]:


# checking first 5 rows - dataframe should have only 5 columns
df.head()


# #### Observation
# * ID columns successfully dropped

# #### Checking for duplicates

# In[238]:


df.duplicated().sum()


# #### Observation
# * There are 11 duplicate observations. We will remove them from the data.

# In[239]:


df = df[(~df.duplicated())].copy()


# #### Looking at data summary

# In[240]:


df.describe().T


# **Observations**
# 
# - With 11 duplicates dropped the total count of rows for each column is now 649
# - The average value for average credit limit is  34,878 and median of 18,000 indicating the presence of high value outliers (right skewed distribution)
# - The average number of credit cards per customer is 4.7 and median of 5 indicating a slight left skew to the distribution
# - The average number of total yearly visits to the bank's physical location is 2.39 and median of 2. The distribution of number of visits ranges from 0 to 5
# - The average number of yearly visits online (i.e. number of online logins) is 2.62 with a median of 2. Number of online visits per year range from 0 to 15
# - The average number of total calls made by the customer to the bank and customer service department in a year is 3.59 with a median of 3. The values distribution ranges from 0 calls to 10
# - Overall, it looks like value distribution are within ranges of what we would expect - no non-sensical values observed so far. We will look at each feature's unique values below for a more thorough assessment

# In[241]:


# Let's see unique values
cols = df.columns

for col in cols:
    print("Unique values in the column '{}' are \n\n".format(col), df[col].unique())
    print("-" * 100)


# In[242]:


# Let's also run a check for missing values
df.isnull().sum()


# #### Observation
# * No missing or non-sensical values observed

# ## Exploratory Data Analysis (EDA)

# ### Univariate Analysis

# In[243]:


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


# In[244]:


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


# In[245]:


histogram_boxplot(df, "Avg_Credit_Limit")


# #### Observation
# * The graphs below clearly indicate a right-skew in the distribution of average credit card limit, driven by large value outliers

# In[246]:


histogram_boxplot(df, "Total_Credit_Cards")


# In[247]:


labeled_barplot(df, "Total_Credit_Cards")


# #### Observations
# 
# * Most customers in the dataset, 147 of them, own 4 credit cards
# * On average, customers own about 5 credit cards

# In[248]:


histogram_boxplot(df, "Total_visits_bank")


# In[249]:


labeled_barplot(df, "Total_visits_bank")


# #### Observations
# 
# * Most customers have visited the bank 2 times in the span of a year
# * We observe a slight left-skew to the distribution but no presence of outliers

# In[250]:


histogram_boxplot(df, "Total_visits_online")


# In[251]:


labeled_barplot(df, "Total_visits_online")


# #### Observations
# 
# * In terms of online visits per year, most customers also visit the bank's website about 2 times
# * We see the presence of outliers at the higher end of the distribution, with some customers having visited the bank online 15 times a year

# In[252]:


histogram_boxplot(df, "Total_calls_made")


# In[253]:


labeled_barplot(df, "Total_calls_made")


# #### Observations
# 
# * Most customers contact the bank's by phone between 0 to 4 times a year
# * At the higher end of the distribution we have customers who have contacted the bank up to 10 times in a year

# ### Bivariate Analysis

# #### Checking for correlations

# In[254]:


plt.figure(figsize=(15, 7))
sns.heatmap(df[cols].corr(), annot=True, vmin=-1, vmax=1, fmt=".2f", cmap="Spectral")
plt.show()


# **Observations**
# 
# * We observe the highest correlations between total number of credit cards and total number of calls made to the bank (-0.65 -- negative correlations mean that values for each variable are inversely correlated), total number of credit cards and average credit card limit (0.61), average credit card limit and total visits online (0.55), and total visits to the bank and total visits online (-0.55)
# * We also observe moderate correlations between total calls made and total visits to the bank (-0.50), average credit card limit and total number of calls made (-0.42) and total bank visits and total number of credit cards (0.31)

# In[255]:


sns.pairplot(data=df[cols], diag_kind="kde")
plt.show()


# #### Observations
# 
# Visualizing data distributions can be helpful in suggesting clustering patterns that might emerge from the data
# * We observe a positive correlation between average credit limit and total visits online as well as total credit cards
# * We also observe a positive correlation between total visits online and total credit cards
# * Average credit limit and total calls made are shown here to be inversely correlated
# 
# 
# In terms of further univariate analysis insights provided by the curves in the graph:
# * For average credit limit we observe a peak followed by a medium bump in the curve and ending in a tail at higher average credit limit values
# * For the distribution of total credit card values we observe 4 peaks in the curve, 2 at mid value range
# * For total bank visits, we see a single steep peak with the rest of values falling at about the 5 yearly visits mark
# * In terms of online bank visits, we observe a single steep peak, a slight curve at mid point and a long tail of values at the higher end of the distribution
# * For total calls made to bank, we observe a peak of values between 0 and 4 calls and then something of a mid hump in the curve followed by a sharp fall at the 10 calls mark
# 
# 
# The overall pattern of these curves (peaks and valleys) suggest anywhere between 2 to 5 customer segments/clusters - further analysis will be conducted to determine their optimal number

# ### Handling outliers

# In[256]:


# let's look at box plot to see all features with outliers
plt.figure(figsize=(20, 30))

for i, variable in enumerate(cols):
    plt.subplot(5, 4, i + 1)
    plt.boxplot(df[variable], whis=1.5)
    plt.tight_layout()
    plt.title(variable)

plt.show()


# In[257]:


outlier_features = ["Avg_Credit_Limit", "Total_visits_online"]


# In[258]:


# Let's treat outliers by capping
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


# In[259]:


#creating a new dataframe for data with no outliers
#we will run clustering algoriths on data with and without outliers 
#and compare to yield best clustering results

df_no_outliers = df.copy()
treat_outliers_all(df_no_outliers, outlier_features)


# In[260]:


# let's look at box plot to see if outliers have been treated or not
plt.figure(figsize=(20, 30))

for i, variable in enumerate(cols):
    plt.subplot(5, 4, i + 1)
    plt.boxplot(df_no_outliers[variable], whis=1.5)
    plt.tight_layout()
    plt.title(variable)

plt.show()


# #### Observation
# * Outliers successfully treated

# # <a id='link1'>Summary of Data Cleaning and EDA</a>
# **Data Description:**
# 
# * 649 rows of data (660 original with 11 duplicate rows removed) with 5 features (Sl_No and Customer Key are ID columns so they were dropped from the data)
# * We observed no missing values in the dataset
# * 11 duplicates found and removed from the dataset for a total of 649 rows
# 
# 
# **Data Cleaning:**
# 
# * Sl_No and Customer Key are ID columns so were dropped from the dataset.
# * As mentioned above, 11 rows with duplicate data were found and removed
# * Outliers observed for "Avg_Credit_Limit" and "Total_visits_online" which were capped the upper whisker (Q3 + 1.5 IQR) value of the distribution
# 
# 
# 
# 
# **Observations from EDA:**
# 
# ###### Univariate Analysis
# * `Avg_Credit_Limit`: average value for average credit limit is  34,878 and median of 18,000 indicating the presence of high value outliers (right skewed distribution)
# * `Total_Credit_Cards`: average number of credit cards per customer is 4.7 and median of 5 indicating a slight left skew to the distribution
# * `Total_visits_bank`: average number of total yearly visits to the bank's physical location is 2.39 and median of 2. The distribution of number of visits ranges from 0 to 5
# * `Total_visits_online`: average number of yearly visits online (i.e. number of online logins) is 2.62 with a median of 2. Number of online visits per year range from 0 to 15 which indicate presence of high value outliers (right skew to the distribution)
# * `Total_calls_made`: average number of total calls to the bank and customer service department in a year is 3.59 with a median of 3. The values distribution range from 0 calls to 10
# 
# * In terms of further univariate analysis insights provided by curve visualizations:
#     - Average credit limit, we observed a peak followed by a medium bump in the curve and ending in a tail at higher average credit limit values
#     - For the distribution of total credit card values we observed 4 peaks in the curve, 2 at mid value range
#     - For total bank visits, we saw a single steep peak with the rest of values falling at about the 5 yearly visits mark
#     - In terms of online bank visits, we observed a single steep peak, a slight ondulation at mid point and a long tail of values at the higher end of the distribution
#     - For total calls made to bank, we observed a peak of values between 0 and 4 calls and then something of a mid hump followed by a sharp fall at the 10 calls mark
# 
# 
# ###### Bivariate Analysis
# 
# **Correlations** between features:
# * We observe the highest correlations between total number of credit cards and total number of calls made to the bank (-0.65 -- negative correlations mean that values for each variable are inversely correlated) followed by the total number of credit cards and average credit card limit (0.61), average credit card limit and total visits online (0.55), and total visits to the bank and total visits online (-0.55)
# * We also observe moderate correlations between total calls made and total visits to the bank (-0.50), average credit card limit and total number of calls made (-0.42) and total bank visits and total number of credit cards (0.31)

# # <a id='link1'>Customer Segmentation: Applying K-Means and Hierarchical Clustering</a>

# ## K-Means Clustering - data with no outliers

# ### Scaling data

# In[261]:


# copying unscaled dataset with no outliers for k-means cluster analysis
df1 = df_no_outliers.copy()


# In[262]:


# scaling the dataset before clustering
sc = StandardScaler()
subset_scaled_df1 = pd.DataFrame(sc.fit_transform(df1), columns=cols)


# In[263]:


subset_scaled_df1.head()


# In[264]:


start = time.process_time()#for counting time elapsed for processing code below

clusters = range(1, 9)
meanDistortions = []

for k in clusters:
    model = KMeans(n_clusters=k)
    model.fit(subset_scaled_df1)
    prediction = model.predict(subset_scaled_df1)
    distortion = (
        sum(
            np.min(
                cdist(subset_scaled_df1, model.cluster_centers_, "euclidean"), axis=1
            )
        )
        / subset_scaled_df1.shape[0]
    )

    meanDistortions.append(distortion)

    print("Number of Clusters:", k, "\tAverage Distortion:", distortion)

plt.plot(clusters, meanDistortions, "bx-")
plt.xlabel("k")
plt.ylabel("Average Distortion")
plt.title("Selecting k with the Elbow Method", fontsize=20)

elapsed = (time.process_time() - start)
print("\nElapsed time (in fractional seconds):", elapsed)


# #### Observation
# 
# * The appropriate value of k from the elbow curve seems to be 3 or 4

# ### Checking the silhouette scores

# In[265]:


start = time.process_time()#for counting time elapsed for processing code below

sil_score = []
cluster_list = list(range(2, 10))
for n_clusters in cluster_list:
    clusterer = KMeans(n_clusters=n_clusters)
    preds = clusterer.fit_predict((subset_scaled_df1))
    # centers = clusterer.cluster_centers_
    score = silhouette_score(subset_scaled_df1, preds)
    sil_score.append(score)
    print("For n_clusters = {}, silhouette score is {}".format(n_clusters, score))

plt.plot(cluster_list, sil_score)
plt.xlabel("k")
plt.ylabel("Silhouette Score")
plt.title("Selecting k with Silhouette Scores", fontsize=20)

elapsed = (time.process_time() - start)
print("\nElapsed time (in fractional seconds):", elapsed)


# #### Observations
# 
# * Highest silhouette score at 3 clusters indicating greatest difference between clusters compared to within clusters. 
# * 1 value indicates that an observation is far from its neighbouring cluster and close to its own whereas -1 denotes that an observation is closer to neighbouring cluster than its own cluster

# In[266]:


start = time.process_time()#for counting time elapsed for processing code below

# finding optimal no. of clusters with silhouette coefficients
visualizer = SilhouetteVisualizer(KMeans(3, random_state=1))
visualizer.fit(subset_scaled_df1)
visualizer.show()

elapsed = (time.process_time() - start)
print("\nElapsed time (in fractional seconds):", elapsed)


# In[267]:


start = time.process_time()#for counting time elapsed for processing code below

# finding optimal no. of clusters with silhouette coefficients
visualizer = SilhouetteVisualizer(KMeans(4, random_state=1))
visualizer.fit(subset_scaled_df1)
visualizer.show()

elapsed = (time.process_time() - start)
print("\nElapsed time (in fractional seconds):", elapsed)


# #### Observation
# 
# * Highest average silhouette scores at 3 clusters compared to 4, indicating optimal number of clusters to be 3

# In[268]:


start = time.process_time()#for counting time elapsed for processing code below

# let's take 3 as number of clusters
kmeans = KMeans(n_clusters=3, random_state=1)
kmeans.fit(subset_scaled_df1)

elapsed = (time.process_time() - start)
print("\nElapsed time (in fractional seconds):", elapsed)


# In[269]:


# adding kmeans cluster labels to unscaled and scaled dataframes
df1["K_means_segments"] = kmeans.labels_
subset_scaled_df1["K_means_segments"] = kmeans.labels_


# ###  Cluster Profiling: K-Means - data with no outliers

# In[270]:


#grouping K-means cluster results
cluster_profile_km_no_outliers = df1.groupby("K_means_segments").mean()

cluster_profile_km_no_outliers["count_in_each_segment"] = (
    df1.groupby("K_means_segments")["Avg_Credit_Limit"].count().values
)


# In[271]:


#displaying K-means cluster profile
cluster_profile_km_no_outliers.style.highlight_max(color="lightgreen", axis=0)


# #### Observations
# 
# * Optimal number of clusters obtained through K-means clustering was 3
# * With observe the count of values for each cluster as follows: 
#     - cluster 0: 50 customers
#     - cluster 1: 221 customers
#     - cluster 2: 378 customers
# * Average credit limit, total credit cards, and total visits online for customers in cluster 0 are highest
# * Total calls made average is highest among cluster 1 customers while average credit limit is lowest among customers in this cluster
# * Cluster 2 customers have highest average number of total bank visits. Cluster 2 is also the largest cluster with 378 customers

# ## K-Means Clustering - data with outliers

# ### Scaling data

# In[272]:


# copying unscaled dataset with outliers for k-means cluster analysis
df1 = df.copy()


# In[273]:


# scaling the dataset before clustering
sc = StandardScaler()
subset_scaled_df1 = pd.DataFrame(sc.fit_transform(df1), columns=cols)


# In[274]:


subset_scaled_df1.head()


# In[275]:


start = time.process_time()#for counting time elapsed for processing code below

clusters = range(1, 9)
meanDistortions = []

for k in clusters:
    model = KMeans(n_clusters=k)
    model.fit(subset_scaled_df1)
    prediction = model.predict(subset_scaled_df1)
    distortion = (
        sum(
            np.min(
                cdist(subset_scaled_df1, model.cluster_centers_, "euclidean"), axis=1
            )
        )
        / subset_scaled_df1.shape[0]
    )

    meanDistortions.append(distortion)

    print("Number of Clusters:", k, "\tAverage Distortion:", distortion)

plt.plot(clusters, meanDistortions, "bx-")
plt.xlabel("k")
plt.ylabel("Average Distortion")
plt.title("Selecting k with the Elbow Method", fontsize=20)

elapsed = (time.process_time() - start)
print("\nElapsed time (in fractional seconds):", elapsed)


# #### Observation
# 
# * The appropriate value of k from the elbow curve seems to be 3 or 4

# ### Checking the silhouette scores

# In[276]:


start = time.process_time()#for counting time elapsed for processing code below

sil_score = []
cluster_list = list(range(2, 10))
for n_clusters in cluster_list:
    clusterer = KMeans(n_clusters=n_clusters)
    preds = clusterer.fit_predict((subset_scaled_df1))
    # centers = clusterer.cluster_centers_
    score = silhouette_score(subset_scaled_df1, preds)
    sil_score.append(score)
    print("For n_clusters = {}, silhouette score is {}".format(n_clusters, score))

plt.plot(cluster_list, sil_score)
plt.xlabel("k")
plt.ylabel("Silhouette Score")
plt.title("Selecting k with Silhouette Scores", fontsize=20)

elapsed = (time.process_time() - start)
print("\nElapsed time (in fractional seconds):", elapsed)


# #### Observations
# 
# * Highest silhouette score at 3 clusters indicating greatest difference between clusters compared to within clusters. 
# * 1 value indicates that an observation is far from its neighbouring cluster and close to its own whereas -1 denotes that an observation is closer to neighbouring cluster than its own cluster

# In[277]:


start = time.process_time()#for counting time elapsed for processing code below

# finding optimal no. of clusters with silhouette coefficients
visualizer = SilhouetteVisualizer(KMeans(3, random_state=1))
visualizer.fit(subset_scaled_df1)
visualizer.show()

elapsed = (time.process_time() - start)
print("\nElapsed time (in fractional seconds):", elapsed)


# In[278]:


start = time.process_time()#for counting time elapsed for processing code below

# finding optimal no. of clusters with silhouette coefficients
visualizer = SilhouetteVisualizer(KMeans(4, random_state=1))
visualizer.fit(subset_scaled_df1)
visualizer.show()

elapsed = (time.process_time() - start)
print("\nElapsed time (in fractional seconds):", elapsed)


# #### Observation
# 
# * Highest average silhouette scores at 3 clusters compared to 4, indicating optimal number of clusters to be 3

# In[279]:


start = time.process_time()#for counting time elapsed for processing code below

# let's take 3 as number of clusters
kmeans = KMeans(n_clusters=3, random_state=1)
kmeans.fit(subset_scaled_df1)

elapsed = (time.process_time() - start)
print("\nElapsed time (in fractional seconds):", elapsed)


# In[280]:


# adding kmeans cluster labels to unscaled and scaled dataframes
df1["K_means_segments"] = kmeans.labels_
subset_scaled_df1["K_means_segments"] = kmeans.labels_


# ###  Cluster Profiling: K-Means - data with outliers

# In[281]:


#grouping K-means cluster results
cluster_profile_km_with_outliers = df1.groupby("K_means_segments").mean()

cluster_profile_km_with_outliers["count_in_each_segment"] = (
    df1.groupby("K_means_segments")["Avg_Credit_Limit"].count().values
)


# In[282]:


#displaying K-means cluster profile
cluster_profile_km_with_outliers.style.highlight_max(color="lightgreen", axis=0)


# #### Observations
# 
# * Like K-means applied to scaled data with no outliers, optimal number of clusters with outliers in the data was 3
# * With observe the count of values for each cluster as follows - notice how count breakdowns are the same as K-means applied to data with no outliers: 
#     - cluster 0: 221 customers
#     - cluster 1: 378 customers
#     - cluster 2: 50 customers
# * Averages are nearly identical to K-means conducted on data with no outliers indicating that the influence of outliers on clustering algorithm was not substantial

# ## Hierarchical Clustering - data with no outliers

# ### Scaling data

# In[283]:


# copying original unscaled dataset for hierarchical cluster analysis
df2 = df_no_outliers.copy()


# In[284]:


# scaling dataset before clustering
sc = StandardScaler()
subset_scaled_df2 = pd.DataFrame(sc.fit_transform(df2), columns=cols)


# In[285]:


subset_scaled_df2.head()


# In[286]:


start = time.process_time()#for counting time elapsed for processing code below

# list of distance metrics
distance_metrics = ["euclidean", "chebyshev", "mahalanobis", "cityblock"]

# list of linkage methods
linkage_methods = ["single", "complete", "average", "weighted"]

high_cophenet_corr = 0
high_dm_lm = [0, 0]

for dm in distance_metrics:
    for lm in linkage_methods:
        Z = linkage(subset_scaled_df2, metric=dm, method=lm)
        c, coph_dists = cophenet(Z, pdist(subset_scaled_df2))
        print(
            "Cophenetic correlation for {} distance and {} linkage is {}.".format(
                dm.capitalize(), lm, c
            )
        )
        if high_cophenet_corr < c:
            high_cophenet_corr = c
            high_dm_lm[0] = dm
            high_dm_lm[1] = lm
            
elapsed = (time.process_time() - start)
print("\nElapsed time (in fractional seconds):", elapsed)


# In[287]:


# printing the combination of distance metric and linkage method with the highest cophenetic correlation
print(
    "Highest cophenetic correlation is {}, which is obtained with {} distance and {} linkage.".format(
        high_cophenet_corr, high_dm_lm[0].capitalize(), high_dm_lm[1]
    )
)


# In[288]:


start = time.process_time()#for counting time elapsed for processing code below

# list of linkage methods
linkage_methods = ["single", "complete", "average", "centroid", "ward", "weighted"]

high_cophenet_corr = 0
high_dm_lm = [0, 0]

for lm in linkage_methods:
    Z = linkage(subset_scaled_df2, metric="euclidean", method=lm)
    c, coph_dists = cophenet(Z, pdist(subset_scaled_df2))
    print("Cophenetic correlation for {} linkage is {}.".format(lm, c))
    if high_cophenet_corr < c:
        high_cophenet_corr = c
        high_dm_lm[0] = "euclidean"
        high_dm_lm[1] = lm
        
elapsed = (time.process_time() - start)
print("\nElapsed time (in fractional seconds):", elapsed)


# In[289]:


# printing the combination of distance metric and linkage method with the highest cophenetic correlation
print(
    "Highest cophenetic correlation is {}, which is obtained with {} linkage.".format(
        high_cophenet_corr, high_dm_lm[1]
    )
)


# #### Observations
# 
# * We see that the cophenetic correlation is maximum with Euclidean distance and average linkage
# 
# * Let's see the dendrograms for the different linkage methods

# In[290]:


start = time.process_time()#for counting time elapsed for processing code below

# list of linkage methods
linkage_methods = ["single", "complete", "average", "centroid", "ward", "weighted"]

# lists to save results of cophenetic correlation calculation
compare_cols = ["Linkage", "Cophenetic Coefficient"]

# to create a subplot image
fig, axs = plt.subplots(len(linkage_methods), 1, figsize=(15, 30))

# We will enumerate through the list of linkage methods above
# For each linkage method, we will plot the dendrogram and calculate the cophenetic correlation
for i, method in enumerate(linkage_methods):
    Z = linkage(subset_scaled_df2, metric="euclidean", method=method)

    dendrogram(Z, ax=axs[i])
    axs[i].set_title(f"Dendrogram ({method.capitalize()} Linkage)")

    coph_corr, coph_dist = cophenet(Z, pdist(subset_scaled_df2))
    axs[i].annotate(
        f"Cophenetic\nCorrelation\n{coph_corr:0.2f}",
        (0.80, 0.80),
        xycoords="axes fraction",
    )
    
elapsed = (time.process_time() - start)
print("\nElapsed time (in fractional seconds):", elapsed)


# #### Observations
# 
# * Although cophenetic correlations are highest for Average, Centroid, and Weighted linkage methods (at approximately 0.89), the dendogram for Complete linkage has a cophenetic correlation of 0.85 and more distinct and balanced clusters
# * We will move ahead with Complete linkage
# * 3 appears to be the appropriate number of clusters from the dendrogram

# In[291]:


start = time.process_time()#for counting time elapsed for processing code below

# modeling Hierarchical cluster on 3 clusters, euclidean distance and complete linkage criterion
HCmodel = AgglomerativeClustering(
    n_clusters=3, affinity="euclidean", linkage="complete"
)
HCmodel.fit(subset_scaled_df2)

elapsed = (time.process_time() - start)
print("\nElapsed time (in fractional seconds):", elapsed)


# In[292]:


# adding hierarchical cluster labels to unscaled and scaled dataframes
subset_scaled_df2["HC_Clusters"] = HCmodel.labels_
df2["HC_Clusters"] = HCmodel.labels_


# ###  Cluster Profiling: Hierarchical Clustering - data with no outliers

# In[293]:


#grouping Hierarchical clustering results
cluster_profile_hc = df2.groupby("HC_Clusters").mean()

cluster_profile_hc["count_in_each_segment"] = (
    df2.groupby("HC_Clusters")["Avg_Credit_Limit"].count().values
)


# In[294]:


#displaying K-means cluster profile
cluster_profile_hc.style.highlight_max(color="lightgreen", axis=0)


# #### Observations
# 
# * Hierarchical clustering also points out that the optimal number of clusters for this dataset is 3
# * Analysis with data with no outliers reveals three clusters with the following counts:
#     - cluster 0: 381 customers
#     - cluster 1: 50 customers
#     - cluster 2: 218 customers
# * We observe a great level of similarity between K-means and Hierarchical clustering in terms of cluster distributions and customer characteritics

# ## Hierarchical Clustering - data with outliers

# ### Scaling data

# In[295]:


# copying original unscaled dataset for hierarchical cluster analysis
df2 = df.copy()


# In[296]:


# scaling dataset before clustering
sc = StandardScaler()
subset_scaled_df2 = pd.DataFrame(sc.fit_transform(df2), columns=cols)


# In[297]:


subset_scaled_df2.head()


# In[298]:


start = time.process_time()#for counting time elapsed for processing code below

# list of distance metrics
distance_metrics = ["euclidean", "chebyshev", "mahalanobis", "cityblock"]

# list of linkage methods
linkage_methods = ["single", "complete", "average", "weighted"]

high_cophenet_corr = 0
high_dm_lm = [0, 0]

for dm in distance_metrics:
    for lm in linkage_methods:
        Z = linkage(subset_scaled_df2, metric=dm, method=lm)
        c, coph_dists = cophenet(Z, pdist(subset_scaled_df2))
        print(
            "Cophenetic correlation for {} distance and {} linkage is {}.".format(
                dm.capitalize(), lm, c
            )
        )
        if high_cophenet_corr < c:
            high_cophenet_corr = c
            high_dm_lm[0] = dm
            high_dm_lm[1] = lm
            
elapsed = (time.process_time() - start)
print("\nElapsed time (in fractional seconds):", elapsed)


# In[299]:


# printing the combination of distance metric and linkage method with the highest cophenetic correlation
print(
    "Highest cophenetic correlation is {}, which is obtained with {} distance and {} linkage.".format(
        high_cophenet_corr, high_dm_lm[0].capitalize(), high_dm_lm[1]
    )
)


# In[300]:


start = time.process_time()#for counting time elapsed for processing code below

# list of linkage methods
linkage_methods = ["single", "complete", "average", "centroid", "ward", "weighted"]

high_cophenet_corr = 0
high_dm_lm = [0, 0]

for lm in linkage_methods:
    Z = linkage(subset_scaled_df2, metric="euclidean", method=lm)
    c, coph_dists = cophenet(Z, pdist(subset_scaled_df2))
    print("Cophenetic correlation for {} linkage is {}.".format(lm, c))
    if high_cophenet_corr < c:
        high_cophenet_corr = c
        high_dm_lm[0] = "euclidean"
        high_dm_lm[1] = lm
        
elapsed = (time.process_time() - start)
print("\nElapsed time (in fractional seconds):", elapsed)


# In[301]:


# printing the combination of distance metric and linkage method with the highest cophenetic correlation
print(
    "Highest cophenetic correlation is {}, which is obtained with {} linkage.".format(
        high_cophenet_corr, high_dm_lm[1]
    )
)


# #### Observations
# 
# * We see that the cophenetic correlation is maximum with Euclidean distance and average linkage
# 
# * Let's see the dendrograms for the different linkage methods

# In[302]:


start = time.process_time()#for counting time elapsed for processing code below

# list of linkage methods
linkage_methods = ["single", "complete", "average", "centroid", "ward", "weighted"]

# lists to save results of cophenetic correlation calculation
compare_cols = ["Linkage", "Cophenetic Coefficient"]

# to create a subplot image
fig, axs = plt.subplots(len(linkage_methods), 1, figsize=(15, 30))

# We will enumerate through the list of linkage methods above
# For each linkage method, we will plot the dendrogram and calculate the cophenetic correlation
for i, method in enumerate(linkage_methods):
    Z = linkage(subset_scaled_df2, metric="euclidean", method=method)

    dendrogram(Z, ax=axs[i])
    axs[i].set_title(f"Dendrogram ({method.capitalize()} Linkage)")

    coph_corr, coph_dist = cophenet(Z, pdist(subset_scaled_df2))
    axs[i].annotate(
        f"Cophenetic\nCorrelation\n{coph_corr:0.2f}",
        (0.80, 0.80),
        xycoords="axes fraction",
    )
    
elapsed = (time.process_time() - start)
print("\nElapsed time (in fractional seconds):", elapsed)


# #### Observations
# 
# * Differently from Hierarchical clustering applied to the no outlier dataset, here we see a clearer distinction between clusters for the linkage method yielding the highest cophenetic correlation: average linkage (0.90)
# * We will move ahead with average linkage
# * 3 appears to be the appropriate number of clusters from the dendrogram

# In[303]:


start = time.process_time()#for counting time elapsed for processing code below

# modeling Hierarchical cluster on 3 clusters, euclidean distance and complete linkage criterion
HCmodel = AgglomerativeClustering(
    n_clusters=3, affinity="euclidean", linkage="average"
)
HCmodel.fit(subset_scaled_df2)

elapsed = (time.process_time() - start)
print("\nElapsed time (in fractional seconds):", elapsed)


# In[304]:


# adding hierarchical cluster labels to unscaled and scaled dataframes
subset_scaled_df2["HC_Clusters"] = HCmodel.labels_
df2["HC_Clusters"] = HCmodel.labels_


# ###  Cluster Profiling: Hierarchical Clustering - data with outliers

# In[305]:


#grouping Hierarchical clustering results
cluster_profile_hc = df2.groupby("HC_Clusters").mean()

cluster_profile_hc["count_in_each_segment"] = (
    df2.groupby("HC_Clusters")["Avg_Credit_Limit"].count().values
)


# In[306]:


#displaying K-means cluster profile
cluster_profile_hc.style.highlight_max(color="lightgreen", axis=0)


# #### Observations
# 
# * Hierarchical clustering conducted with data which includes outlier values also points out that the optimal number of clusters for this dataset is 3
# * Unlike hierarchical clustering with data with no outliers, here the best clustering results were yielded from use of average linkage - linkage criterion with the highest cophenetic correlation and distinct clusters 
# * Customer counts for each of the clusters still yielded similar results to both K-means and Hierarchical clustering performed on data with no outliers

# ## Performance comparison between K-means and Hierarchical clustering

# * We took measure of the processing times for running K-means and Hierarchical clustering algorithms and, as expected, Hierarchical clustering took longer to process than K-means, given that it is more computationally expensive, scaling quadratically in terms of distances computation

# ## Cluster Profiling: K-Means and Hierarchical Clustering
# 
# * We now turn to a more in-depth comparison between results yielded by the 2 clustering techniques applied: K-means and Hierarchical clustering
# * Given that no substantial difference in clustering results was observed between algorithms applied to scaled data with and without the presence of outliers and the better results yielded from data with outliers in hierarchical clustering we opted to conduct a comparison between results of K-means and Hierarchical clustering done on data which includes outliers

# In[307]:


#grouping K-means cluster results - unscaled data
cluster_profile_km = df1.groupby("K_means_segments").mean()

cluster_profile_km["count_in_each_segment"] = (
    df1.groupby("K_means_segments")["Avg_Credit_Limit"].count().values
)

#grouping Hierarchical clustering results - unscaled data
cluster_profile_hc = df2.groupby("HC_Clusters").mean()

cluster_profile_hc["count_in_each_segment"] = (
    df2.groupby("HC_Clusters")["Avg_Credit_Limit"].count().values
)


# In[308]:


#grouping K-means cluster results - scaled data
scaled_cluster_profile_km = subset_scaled_df1.groupby("K_means_segments").mean()

scaled_cluster_profile_km["count_in_each_segment"] = (
    subset_scaled_df1.groupby("K_means_segments")["Avg_Credit_Limit"].count().values
)

#grouping Hierarchical clustering results - scaled data
scaled_cluster_profile_hc = subset_scaled_df2.groupby("HC_Clusters").mean()

scaled_cluster_profile_hc["count_in_each_segment"] = (
    subset_scaled_df2.groupby("HC_Clusters")["Avg_Credit_Limit"].count().values
)


# In[309]:


#displaying K-means cluster profile - unscaled data
cluster_profile_km.style.highlight_max(color="lightgreen", axis=0)


# In[310]:


#displaying Hierarchical cluster profile - unscaled data
cluster_profile_hc.style.highlight_max(color="lightgreen", axis=0)


# #### Observations
# 
# * Both K-means and Hierarchical clusters count breakdown are nearly the same
# * With only small average and count differences, it appears that cluster 0 in Hierarchical clustering (HC) is equivalent to cluster 0 in K-Means (KM), cluster 1 in HC equivalent to cluster 2 in KM, and cluster 2 in HC equivalent to cluster 1 in KM

# #### Creating a single dataframe to hold KM and HC clustering results and changing HC cluster labels to match KM and assist with analysis

# In[311]:


#relabeling HC Clusters to match KM labels
df2["HC_Clusters"].replace({1: 2, 2: 1}, inplace=True)
subset_scaled_df2["HC_Clusters"].replace({1: 2, 2: 1}, inplace=True)

#extracting "HC_Clusters" column
extracted_col_unscaled = df2["HC_Clusters"]
extracted_col_scaled = subset_scaled_df2["HC_Clusters"]

#creating new unscaled and scaled dataframes by adding HC_Clusters values to K-means dataframes
df_all = df1.join(extracted_col_unscaled)
subset_scaled_df_all = subset_scaled_df1.join(extracted_col_scaled)


# In[312]:


#regrouping K-means cluster results
cluster_profile_km = df_all.groupby("K_means_segments").mean()

cluster_profile_km["count_in_each_segment"] = (
    df_all.groupby("K_means_segments")["Avg_Credit_Limit"].count().values
)

#regrouping Hierarchical clustering results
cluster_profile_hc = df_all.groupby("HC_Clusters").mean()

cluster_profile_hc["count_in_each_segment"] = (
    df_all.groupby("HC_Clusters")["Avg_Credit_Limit"].count().values
)


# In[322]:


#displaying K-means cluster profile
cols_count = cols.insert(5,"count_in_each_segment")
cluster_profile_km[cols_count].style.highlight_max(color="lightgreen", axis=0)


# In[323]:


#displaying Hierarchical cluster profile
cluster_profile_hc[cols_count].style.highlight_max(color="lightgreen", axis=0)


# #### Observation
# 
# * From the segments breakdown above we can see that the number of customers assigned to each of the 3 clusters was nearly identical for K-means (KM) and Hierarchical clustering (HC)

# #### Plotting KM and HC cluster distributions

# In[324]:


#Plotting KM cluster scaled distributions
fig, axes = plt.subplots(1, 5, figsize=(16, 6))
fig.suptitle("Boxplot of scaled numerical variables for each cluster - K-Means Clusters", fontsize=20)
counter = 0
for ii in range(5):
    sns.boxplot(
        ax=axes[ii],
        y=subset_scaled_df_all[cols[counter]],
        x=subset_scaled_df_all["K_means_segments"],
    )
    counter = counter + 1

fig.tight_layout(pad=2.0)


# In[325]:


#Plotting HC cluster scaled distributions
fig, axes = plt.subplots(1, 5, figsize=(16, 6))
fig.suptitle("Boxplot of scaled numerical variables for each cluster - Hierarchical Clustering", fontsize=20)
counter = 0
for ii in range(5):
    sns.boxplot(
        ax=axes[ii],
        y=subset_scaled_df_all[cols[counter]],
        x=subset_scaled_df_all["HC_Clusters"],
    )
    counter = counter + 1

fig.tight_layout(pad=2.0)


# In[326]:


#Plotting KM cluster unscaled distributions
fig, axes = plt.subplots(1, 5, figsize=(16, 6))
fig.suptitle("Boxplot of original numerical variables for each cluster - K-Means Clusters", fontsize=20)
counter = 0
for ii in range(5):
    sns.boxplot(
        ax=axes[ii],
        y=df_all[cols[counter]],
        x=df_all["K_means_segments"],
    )
    counter = counter + 1

fig.tight_layout(pad=2.0)


# In[327]:


#Plotting HC cluster unscaled distributions
fig, axes = plt.subplots(1, 5, figsize=(16, 6))
fig.suptitle("Boxplot of original numerical variables for each cluster - Hierarchical Clustering", fontsize=20)
counter = 0
for ii in range(5):
    sns.boxplot(
        ax=axes[ii],
        y=df_all[cols[counter]],
        x=df_all["HC_Clusters"],
    )
    counter = counter + 1

fig.tight_layout(pad=2.0)


# #### Observations
# 
# * As the boxplots illustrate, the value distributions for average credit limit, total credit cards, total bank visits, total visits online, and total calls made are nearly the same for segments that emerged from K-means and Hierarchical clustering 
# * Cluster 0 customers have low to mid average credit limit and own anywhere between 2 and 7 credit cards. They stand out for being the segment with highest number of visits to the bank
# * Cluster 1 customers are at the lowest end of average credit limit values and total number of credit cards whilst having the highest average for total number of calls made yearly to the bank
# * Cluster 2 customers stand out for having the highest average credit limit, total number of credit cards, and online bank visits as well as the lowest number of calls made to the bank

# In[330]:


subset_scaled_df_all.groupby("K_means_segments")[cols].mean().plot.bar(figsize=(15, 6))


# In[329]:


subset_scaled_df_all.groupby("HC_Clusters")[cols].mean().plot.bar(figsize=(15, 6))


# #### Observations
# 
# * The bar charts which plot scaled (z-score) means for each feature help us to visualize the characteristics of cluster/customer segments in terms of their average credit limit, total number of credit cards, physical bank visits, online bank visits, and calls made to the bank
# 
# * Once again, we observe that customer segment assignments yielded nearly the same results for K-means and Hierarchical clustering algorithms
# 
# The general pattern observed is the following:
# 
# * Customers in cluster 0 stand out for having the highest average number of yearly visits to the bank’s branches/physical locations and lowest level of online bank visits
# * Customers in cluster 1 stand out for having the lowest average credit card limit and number of credit cards and highest number of calls to the bank
# * Customers in cluster 2 stand out for having in comparison to the other clusters by far the highest average credit limit, total number of credit cards, and online bank visits. Customers in this cluster also have the lowest average visits the bank branches and number of calls

# # <a id='link1'>Insights and Business Recommendations</a>

# #### Through the application of K-means and Hierarchical clustering algorithms, 3 distinct customers segments emerged. We could name and describe them as follows

# **Branch believers**
# Branch believers are customers grouped in cluster 0 - the largest of the 3 clusters, with 58% of the customers in the dataset. As our findings indicate, these customers make visits to the bank’s physical locations more often than other customers and are less inclined to visit the bank’s website. We would recommend conducting further research to understand more about this segment. We would guess that they are more traditional in their approach to banking and credit, perhaps more conservative with their spending. It would also be important to find out how loyal these customers are to the AllLife Bank brand. Given the in-person contact factor, there is a special opportunity here to strengthen the probably already well forged relationship between the bank and this customer segment. As it takes more time and effort to visit the bank in person, positive and negative experiences during bank visits are both likely to carry more weight, therefore we recommend the development of marketing strategies to enhance the experience of these customers during their bank visits to help improve customer satisfaction and retention.
# 
# **Help seekers**
# Help seekers are customers grouped in cluster 1 - the second to largest segment, with 34% of the customers in the dataset. They stand out for contacting the bank more often over the phone than other customers. They also have the lowest average credit limit among the three segments. These customers are more inclined to use the bank’s website than branch believers. We would recommend conducting research to better understand who they are - are help seekers younger customers who are less experienced with banking tools and therefore more inclined to need customer service assistance? What are the main reasons for them to contact the bank? This can be an opportunity to improve, for instance, the bank’s online customer service tools. We would recommend a well-developed Frequently Asked Questions page to address this segment’s common questions and concerns. With that, we also recommend an in-depth analysis to unveil the main causes of help seekers’ calls to the bank. By listening carefully to these customers, we believe the marketing team can gain valuable insight into areas that the bank might be falling short and help to improve overall customer experience.
# 
# **Affluent online patrons**
# Affluent online patrons are customers grouped in cluster 2 - the smallest segment in the dataset, with 8% of the customers. This is the segment with by far the highest average credit limit among AllLife Bank’s customers. They are also the bank’s most avid online users and least likely to visit the bank’s branches and contact the bank over the phone. Affluent online patrons are likely to have busy lives and to prize convenience, as such we recommend developing a special online marketing strategy tailored to their needs. Further research would be beneficial to understand what offers would most likely compel them to expand share of use of the bank’s credit cards. Given the high number of credit cards these customers are likely to possess (8 to 10), affluent online patrons might not be merely interested in using credit cards as a spending instrument but also as a tool to help them achieve other financial goals - perhaps developing a special credit card product, like one with special low interest rate offers for big purchases, would yield promising results.
# 

# In[ ]:




