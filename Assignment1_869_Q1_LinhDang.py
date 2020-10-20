#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# [Linh, Dang]
# [20195426]
# [MMA]
# [Winter 2021]
# [MMA 869]
# [15 August 2020]


# Answer to Question [1]


# ## IMPORT LIBRARIES AND SET SYSTEM OPTIONS

# In[665]:


from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

from IPython.display import DisplayObject, display

# In[633]:

import pandas as pd
pd.set_option('display.max_rows', 270000)
pd.set_option('display.max_columns', 500)
pd.options.display.float_format = '{:.5f}'.format

import os

import pandas_profiling
from pandas_profiling import ProfileReport

from statistics import mean
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, silhouette_samples

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})

from yellowbrick.cluster import SilhouetteVisualizer, InterclusterDistance, KElbowVisualizer
import itertools
import scipy

from scipy.spatial import distance


# ## Read the data

# In[468]:


# Read in data from personal Github
df_org = pd.read_csv("https://raw.githubusercontent.com/LDang47/jewelry-dataset/master/jewelry_customers.csv")

# print info
df_org.shape
df_org.columns
df_org.head()
df_org.dtypes


# ## Plot the data

# In[469]:


# Plot the data using 2 features: Income and Spending Score
plt.style.use('default');

plt.figure(figsize=(12, 10));
plt.grid(True);

plt.scatter(df_org.iloc[:, 0], df_org.iloc[:, 1], c="black", s=100);
plt.title("Jewelry Data", fontsize=20);
plt.xlabel('Age', fontsize=22);
plt.ylabel('Income', fontsize=22);
plt.xticks(fontsize=18);
plt.yticks(fontsize=18);


# ## Data pre-processing and Feature engineering

# In[470]:


# Check for Nulls and missing value
df_org.isnull().sum()


# In[471]:


# Feature engineering to create new features
df_org["Spendings"] = df_org["Income"] - df_org["Savings"]
df_org["Save_Spend_ratio"] = df_org["Savings"]/df_org["Spendings"]
df_org["Save_Income_ratio"] = df_org["Savings"]/df_org["Income"]
df_org["Spend_Income_ratio"] = df_org["Spendings"]/df_org["Income"]

# print info
df_org.shape
df_org.columns
df_org.head()
df_org.dtypes


# In[472]:


# Have a look at summary statistics of the entire dataset 
profile = pandas_profiling.ProfileReport(df_org, html={'style':{'full_width':True}}, minimal=True)
profile.to_notebook_iframe()


# In[473]:


#Standard Scaler
float_feat = ['Age', 'Income', 'SpendingScore', 'Savings', 'Spendings', 'Save_Spend_ratio', 'Save_Income_ratio', 'Spend_Income_ratio']

scaler = StandardScaler()
df_org[float_feat] = scaler.fit_transform(df_org[float_feat])

df_org.shape
df_org.columns
df_org.head()


# ## K-Means

# ### Elbow method

# In[474]:


# Create copy of the orginial dataset
df = df_org.copy()


# In[475]:


# Using the elbow method and silhouettes score to find the optimal number of clusters
inertias = {}
silhouettes = {}
for k in range(2, 15):
    kmeans = KMeans(n_init=10, init='k-means++', n_clusters=k, max_iter=1000, random_state=40).fit(df)
    inertias[k] = kmeans.inertia_ # Inertia: Sum of distances of samples to their closest cluster center
    silhouettes[k] = silhouette_score(df, kmeans.labels_, metric='euclidean')
    

plt.figure();
plt.grid(True);
plt.plot(list(inertias.keys()), list(inertias.values()), '-bx');
plt.title('K-Means, Elbow Method')
plt.xlabel("Number of clusters, K");
plt.ylabel("Inertia");


plt.figure();
plt.grid(True);
plt.plot(list(silhouettes.keys()), list(silhouettes.values()), '-bx');
plt.title('K-Means, Elbow Method')
plt.xlabel("Number of clusters, K");
plt.ylabel("Silhouette");


# ### Clustering: K-Means

# In[476]:


#Run the KMeans Algorithm with k = 4
k_means = KMeans(init='k-means++', n_clusters= 4, n_init=10, max_iter=1000, random_state=40)
k_means.fit(df)

k_means.labels_


# In[477]:


# Let's look at the centers
k_means.cluster_centers_


# In[478]:


# Reverse he scaler to reveal true centers
means = scaler.inverse_transform(k_means.cluster_centers_)
means


# In[479]:


plt.style.use('default');

plt.figure(figsize=(12, 10));
plt.grid(True);

sc = plt.scatter(df.iloc[:, 0], df.iloc[:, 1], s=200, c=k_means.labels_);
plt.title("K-Means (K=4)", fontsize=20);
plt.xlabel('Age', fontsize=24);
plt.ylabel('Income', fontsize=24);
plt.xticks(fontsize=20);
plt.yticks(fontsize=20);

for label in k_means.labels_:
    plt.text(x=k_means.cluster_centers_[label, 0], y=k_means.cluster_centers_[label, 1], s=label, fontsize=32, 
             horizontalalignment='center', verticalalignment='center', color='black',
             bbox=dict(facecolor='white', edgecolor='black', boxstyle='round, pad=0.1', alpha=0.2));


# ### Internal Validation Metrics

# In[480]:


# WCSS == Inertia
k_means.inertia_

# Silhouette_score
silhouette_score(df, k_means.labels_)


# In[481]:


# Try K-Means with k in range(2, 10)
def do_kmeans(df, k):
    k_means = KMeans(init='k-means++', n_clusters=k, n_init=10, max_iter=1000, random_state=40)
    k_means.fit(df)
    wcss = k_means.inertia_
    sil = silhouette_score(df, k_means.labels_)
    
    plt.style.use('default');

    sample_silhouette_values = silhouette_samples(df, k_means.labels_)
    sizes = 200*sample_silhouette_values

    plt.figure(figsize=(16, 10));
    plt.grid(True);

    plt.scatter(df.iloc[:, 0], df.iloc[:, 1], s=sizes, c=k_means.labels_)
    plt.scatter(k_means.cluster_centers_[:, 0], k_means.cluster_centers_[:, 1], marker='x', s=300, c="black")

    plt.title("K-Means (K={}, WCSS={:.2f}, Sil={:.2f})".format(k, wcss, sil), fontsize=20);
    plt.xlabel('Age', fontsize=22);
    plt.ylabel('Income', fontsize=22);
    plt.xticks(fontsize=18);
    plt.yticks(fontsize=18);
    plt.show()
    
    visualizer = SilhouetteVisualizer(k_means)
    visualizer.fit(df)
    visualizer.poof()
    fig = visualizer.ax.get_figure();
    
    print("K={}, WCSS={:.2f}, Sil={:.2f}".format(k, wcss, sil))

for k in range(2, 10):
    do_kmeans(df, k)


# In[482]:


#Add the labels to the df dataset
df["KMeans Labels"] = k_means.labels_
df.shape
df.head()

#Number of people in each cluster
df["KMeans Labels"].value_counts()


# ## DBSCAN

# ### Elbow method

# In[484]:


# Create copy of the orginial dataset
df = df_org.copy()


# In[485]:


silhouettes = {}
for eps in np.arange(0.1, 1, 0.1):
    db = DBSCAN(eps=eps, min_samples=3).fit(df)
    silhouettes[eps] = silhouette_score(df, db.labels_, metric='euclidean')
    

plt.figure();
plt.plot(list(silhouettes.keys()), list(silhouettes.values()), '-bx');
plt.title('DBSCAN, Elbow Method')
plt.xlabel("Eps");
plt.ylabel("Silhouette");
plt.grid(True);

# Select eps = 0.5


# ### Clustering: DBSCAN

# In[497]:


db = DBSCAN(eps=0.5, min_samples=10)
db.fit(df)


# In[487]:


db.labels_


# In[488]:


silhouette_score(df, db.labels_)


# In[489]:


plt.figure(figsize=(16, 10));
plt.grid(True);

unique_labels = set(db.labels_)
colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))];

for k in unique_labels:
    if k == -1:        # Black used for noise.
        col = [0, 0, 0, 1]
    else:
        col = colors[k]

    xy = df[db.labels_ == k]
    plt.plot(xy.iloc[:, 0], xy.iloc[:, 1], 'o', markerfacecolor=tuple(col), markeredgecolor='k', markersize=14);

    
plt.title('');
plt.title("DBSCAN (n_clusters = {:d}, black = outliers)".format(len(unique_labels)), fontsize=20);
plt.xlabel('Age (K)', fontsize=22);
plt.ylabel('Income', fontsize=22);
plt.xticks(fontsize=18);
plt.yticks(fontsize=18);


# In[491]:


# Add the labels to the df dataset
df["DBSCAN Labels"] = db.labels_
df.shape
df.head()

# Number of people in each cluster
df["DBSCAN Labels"].value_counts()


# ## Hierarchical (Agglomerative)

# In[500]:


# Create copy of the orginial dataset
df = df_org.copy()


# In[502]:


#Build 4 clusters
agg = AgglomerativeClustering(n_clusters=4, affinity='euclidean', linkage='ward')
agg.fit(df)


# In[503]:


agg.labels_


# In[504]:


silhouette_score(df, agg.labels_)


# ### Dendograms, Linkages, and Distance Metrics

# In[506]:


# Plot the dendogram and scatterplot for 4 clusters
def plot_agg(df, linkage, metric):
    aggl = scipy.cluster.hierarchy.linkage(df, method=linkage, metric=metric)
    
    labels = scipy.cluster.hierarchy.fcluster(aggl, 4, criterion="maxclust")
    
    sil = 0
    n = len(set(labels))
    if n > 1:
        sil = silhouette_score(df , labels, metric=metric)
    print("Linkage={}, Metric={}, Clusters={}, Silhouette={:.3}".format(linkage, metric, n, sil))
    
    # Plot the dendogram
    plt.figure(figsize=(12, 5))  
    plt.title("Customer Dendogram (Linkage={}, Distance={}, N={}, Sil={:.3f})".format(linkage, metric, n, sil))  
    dend = scipy.cluster.hierarchy.dendrogram(aggl);
    
    # Plot the points
    plt.style.use('default');
    plt.figure(figsize=(16, 10));
    plt.grid(True);

    plt.scatter(df.iloc[:, 0], df.iloc[:, 1], s=100, c=labels);
    plt.title("Customer Agglomerative (Linkage={}, Distance={}, N={}, Sil={:.3f})".format(linkage, metric, n, sil), fontsize=20);
    plt.xlabel('Age', fontsize=22);
    plt.ylabel('Income', fontsize=22);
    plt.xticks(fontsize=18);
    plt.yticks(fontsize=18);

    
linkages = ['ward']
metrics = ['euclidean']

for prod in list(itertools.product(linkages, metrics)):
    
    # Some combos are not allowed
    if (prod[0] in ['ward', 'centroid']) and prod[1] != 'euclidean':
        continue
        
    plot_agg(df, prod[0], prod[1])


# In[507]:


#Add the labels to the df dataset
df["Agg Labels"] = agg.labels_
df.shape
df.head()

#Number of people in each cluster
df["Agg Labels"].value_counts()


# ## Gaussian Mixture Models (GMM)

# In[643]:


# Create copy of the orginial dataset
df = df_org.copy()


# ### Elbow graph

# In[644]:


# How many components (clusters) to have?
n_components = np.arange(1, 10)

models = [GaussianMixture(n, covariance_type='full', random_state=0).fit(df)
          for n in n_components]

plt.plot(n_components, [m.bic(df) for m in models], label='BIC')
plt.plot(n_components, [m.aic(df) for m in models], label='AIC')
plt.legend(loc='best')
plt.xlabel('Information Criterion')
plt.xlabel('n_components');

# The optimal number of clusters is the value that minimizes the AIC or BIC, depending on which approximation we wish to use.
# Here we can see that 4,5,6,7 seem reasonable. we will try with all 4 and select the best option


# ### Clustering: GMM

# In[645]:


gmm = GaussianMixture(n_components=5, n_init = 10, covariance_type='full', random_state=40).fit(df)
labels = gmm.predict(df)
labels


# In[646]:


# Check to see if the algorithm coverged and how many interations it took
gmm.converged_
gmm.n_iter_


# In[647]:


silhouette_score(df, labels)


# In[648]:


# Plot scatterplot and make ellipses representing Gaussian distribution for 4 clusters
def make_ellipses(gmm, ax):
    for n, label in enumerate(set(labels)):
        if gmm.covariance_type == 'full':
            covariances = gmm.covariances_[n][:2, :2]
        elif gmm.covariance_type == 'tied':
            covariances = gmm.covariances_[:2, :2]
        elif gmm.covariance_type == 'diag':
            covariances = np.diag(gmm.covariances_[n][:2])
        elif gmm.covariance_type == 'spherical':
            covariances = np.eye(gmm.means_.shape[1]) * gmm.covariances_[n]
        v, w = np.linalg.eigh(covariances)
        u = w[0] / np.linalg.norm(w[0])
        angle = np.arctan2(u[1], u[0])
        angle = 180 * angle / np.pi  # convert to degrees
        v = 2. * np.sqrt(2.) * np.sqrt(v)
        ell = mpl.patches.Ellipse(gmm.means_[n, :2], v[0], v[1],
                                  180 + angle, color='red')
        ell.set_clip_box(ax.bbox)
        ell.set_alpha(0.2)
        ax.add_artist(ell)
        ax.set_aspect('equal', 'datalim')


plt.style.use('default');

plt.figure(figsize=(12, 10));
plt.grid(True);

plt.scatter(df.iloc[:, 0], df.iloc[:, 1], s=100, c=labels);
make_ellipses(gmm, plt.gca())
plt.title("GMM", fontsize=20);
plt.xlabel('Age', fontsize=22);
plt.ylabel('Income', fontsize=22);
plt.xticks(fontsize=18);
plt.yticks(fontsize=18);


# In[649]:


def describe_clusters(df, labels):
    X2 = df.copy()
    X2['ClusterID'] = labels
    print('\nCluster sizes:')
    print(X2.groupby('ClusterID').size())
    print('\nCluster stats:')
    display(X2.groupby('ClusterID').describe(include='all').transpose())

describe_clusters(df, labels)


# ## Cluster interpretation: GMM 5 clusters

# In[651]:


# Reverse the scaler to reveal true data points
col_names = ['Age', 'Income', 'SpendingScore', 'Savings', 'Spendings', 'Save_Spend_ratio', 'Save_Income_ratio', 'Spend_Income_ratio']

df_fin = pd.DataFrame(scaler.inverse_transform(df[col_names]), columns=col_names)
df_fin['GMM Labels'] = labels

df_fin.head()


# In[652]:


# Build statistics tables for all 5 clusters using orignal unscaled data

# CLUSTER 0 
print('\nCluster 0:')
Cluster_0 = df_fin[df_fin["GMM Labels"]==0]
Cluster_0.describe().transpose()

# CLUSTER 1
print('\nCluster 1:')
Cluster_1 = df_fin[df_fin["GMM Labels"]==1]
Cluster_1.describe().transpose()

# CLUSTER 2
print('\nCluster 2:')
Cluster_2 = df_fin[df_fin["GMM Labels"]==2]
Cluster_2.describe().transpose()

# CLUSTER 3
print('\nCluster 3:')
Cluster_3 = df_fin[df_fin["GMM Labels"]==3]
Cluster_3.describe().transpose()

# CLUSTER 4
print('\nCluster 4:')
Cluster_4 = df_fin[df_fin["GMM Labels"]==4]
Cluster_4.describe().transpose()


# In[653]:


# Print 5 members of each clusters for observations using original unscaled data
for label in set(labels):
    print('\nCluster {}:'.format(label))
    df_tmp = df_fin[labels==label].copy()
    df_tmp.loc['mean'] = df_tmp.mean()
    df_tmp.tail(5)


# ### Relative Importance Plot

# In[659]:


# Add GMM Lables column to the scaled dataset
df['GMM Labels'] = labels

# Calculate cluster mean and poplucation mean
cluster_avg = df.groupby(['GMM Labels']).mean()
population_avg = df.drop(['GMM Labels'], axis=1).mean()

# Find relative importance by taking the differences between cluster and population average
relative_imp = cluster_avg - population_avg

# Relative importance plot
plt.figure(figsize=(8, 4));
plt.title('Relative importance of features');
sns.heatmap(data=relative_imp, annot=True, fmt='.2f', cmap='RdYlGn');


# ### Find Exemplars

# In[655]:


# Find feature means of each clusters
df_fin1 = df_fin.loc[:, df_fin.columns != 'GMM Labels']
means = np.zeros((5, df_fin1.shape[1]))

for i, label in enumerate(set(labels)):
    means[i,:] = df_fin1[labels==label].mean(axis=0)
    
means


# In[656]:


# Find Exemplars in each of the 5 clusters
for i, label in enumerate(set(labels)):
    X_tmp= df_fin1
    exemplar_idx = distance.cdist([means[i]], df_fin1).argmin()
   
    print('\nCluster {}:'.format(label))
    print("  Examplar ID: {}".format(exemplar_idx))

    display(df_fin1.iloc[[exemplar_idx]])


# In[666]:


# Project the 5 clusters onto 3d plane using the original unscaled data
plt.style.use('default');
fig = plt.figure(figsize=(16, 10));
plt.grid(True);

ax = fig.add_subplot(111, projection='3d')
ax.scatter(df_fin['Age'], df_fin['Income'], df_fin['SpendingScore'], s=150, c=labels);

plt.title("GMM clusters", fontsize=20);
plt.xlabel('Age', fontsize=22);
plt.ylabel('Income', fontsize=22);
plt.xticks(fontsize=14);
plt.yticks(fontsize=12);


# In[ ]:




