import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

mall_df=pd.read_csv(r"C:\sudhanshu_projects\12-june-ml1-project\Mall_Customers.csv")
x=mall_df.iloc[:,[3,4]].values


#works on kmean cluster

from sklearn.cluster import KMeans

wcss = []

for i in range(1,11):
    kmeans=KMeans(n_clusters=i,init="k-means++",random_state=0)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)
    
plt.plot(range(1,11),wcss)
plt.title("The elbow method")
plt.xlabel("no of cluster")
plt.ylabel("wcss")
plt.show()



#now we make 5 cluster.
kmeans=KMeans(n_clusters=5,init="k-means++",random_state=0)
y_means=kmeans.fit_predict(x)

#visualize the cluster.
plt.scatter(x[y_means==0,0],x[y_means==0,1],s=100,c="red",label="cluster1")
plt.scatter(x[y_means == 1, 0], x[y_means == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')
plt.scatter(x[y_means == 2, 0], x[y_means == 2, 1], s = 100, c = 'green', label = 'Cluster 3')
plt.scatter(x[y_means == 3, 0], x[y_means == 3, 1], s = 100, c = 'cyan', label = 'Cluster 4')
plt.scatter(x[y_means == 4, 0], x[y_means == 4, 1], s = 100, c = 'magenta', label = 'Cluster 5')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = 'yellow', label = 'Centroids')
plt.title('Clusters of customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()

#Number of customer of cluster3.

c=0
for i in y_means:
    if (i==3):
        c=c+1
        
print(c)

#category3 has total 22 customer.
