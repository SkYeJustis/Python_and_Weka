# -*- coding: utf-8 -*-
"""
Clustering - Weka
@author: SkYe


Dataset Description: https://archive.ics.uci.edu/ml/datasets/Adult

Main reference for python-weka-wrapper:
http://pythonhosted.org/python-weka-wrapper/
"""
import weka.core.jvm as jvm
jvm.start(max_heap_size="1000m")

# Load data: Must be a weka-derived object
# Dataset has nominal and numeric variables
import weka.core.converters as converters
data_dir = "data/"
data = converters.load_any_file(data_dir + "adult.csv")
# data.class_is_last() #Should not be present
data.delete_last_attribute()

# Create train and test sets
from weka.core.classes import Random
test, train =  data.train_test_split(0.90, Random(1))

# Check data in datasets
print(train.num_instances)
print(test.num_instances)

# Check data in datasets
print(train.num_attributes)
print(test.num_attributes)

# Create classifier
from weka.clusterers import Clusterer
# Creating a clusterer with 4 clusters
clusterer = Clusterer(classname="weka.clusterers.SimpleKMeans", options=["-N", "4"])
clusterer.build_clusterer(train)

# View cluster model
print(clusterer)

# cluster the test data
for inst in test:
    cl = clusterer.cluster_instance(inst)  # 0-based cluster index
    dist = clusterer.distribution_for_instance(inst)   # cluster membership distribution
    print("cluster=" + str(cl) + ", distribution=" + str(dist))

