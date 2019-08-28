#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# =============================================================================
# Importing packages
# =============================================================================
from sklearn.cluster import KMeans
from sklearn.datasets import load_digits
import numpy as n
import matplotlib.pyplot as w
import math
import seaborn as sn
from scipy.stats import mode 
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

# =============================================================================
# Part I : Please load your dataset, and 
# use K-means to build your model and groups your whole dataset into 10 clusters.
# =============================================================================
digits = load_digits()
data = digits.data
label = digits.target
images = digits.images

#KMeans instance with 10 clusters
model = KMeans(n_clusters=10,random_state=0)
model.fit(data)
clusters = model.predict(data)
model = KMeans(n_clusters=16)

# =============================================================================
# Part II: Visualize your dataset. Please plot the following figure in your source code.
# =============================================================================
w.figure(figsize = (6,6))
nrows, ncols = 4, 4
w.gray()
w.suptitle("Hand-Written Digits Images",size=20,color="blue")
for i in range(ncols * nrows):
    ax = w.subplot(nrows, ncols, i + 1)
    ax.matshow(digits.images[i,...])
    w.xticks([]); 
    w.yticks([])
    w.title(digits.target[i])
labels = n.zeros_like(clusters)
for i in range(10):
    mask = (clusters==i)
    labels[mask] = mode(label[mask])[0]
w.show()

# =============================================================================
# Part III: Please evaluate your K-means model performance by using confusion matrix. 
# Please plot your confusion matrix figure as the following.
# =============================================================================
score = accuracy_score(label, labels)
w.figure(figsize = (7,6))
sn.set()
mat = confusion_matrix(digits.target, labels)
sn.heatmap(mat.T, square = True, annot = True, 
           fmt = 'd', cbar = False,
           xticklabels = digits.target_names,
           yticklabels = digits.target_names)
w.xlabel('True Label',size=15)
w.ylabel('Predicted Label',size=15);
w.title("Confusion Matrix \n Accuracy = %.2f %% " %score,color='blue',size=20)
w.show()

# =============================================================================
# Part IV: Please use Elbow finding method to pick your best K. 
# Please plot the following figure.
# =============================================================================
bow = []
ks = range(1,15)
for k in ks:
    km = KMeans(n_clusters=k).fit(data)
    bow.append(math.log(km.inertia_,10)) 
w.figure()
w.plot(ks,bow,'-o')
w.xticks([i for i in range(1,15)])
w.text(10,bow[10] , r'K=10',color='red',size=20)
w.xlabel("Number of Clusters, K",size=15)
w.ylabel("Log 10 Inertia",size=15)
w.title("Elbow Finding K",size=20,color="blue")
w.show()

# =============================================================================
# Part V: Please use t-SNE to map your 64*64 Dimension data to 2D space. 
# Please plot the following figure in your source code. Here, the dimension is 64*64 instead of 10.
# =============================================================================
w.figure(figsize=(6,5))
model = TSNE(learning_rate = 100)
transformed = model.fit_transform(data)
xx = transformed[:,0]
yy = transformed[:,1]
w.xlim([-90,84])
w.ylim([-72,84])
w.scatter(xx,yy,c=km.labels_)
w.title('Map 2D Samples to 2D Space',size=20,color="blue")
w.show()

##############################################################################