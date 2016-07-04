
"""
Naive Bayes in Weka
Created on Sun Jul 03 15:49:46 2016

@author: SkYe
"""

import weka.core.jvm as jvm
jvm.start(max_heap_size="2500m")


# Load data: Must be a weka-derived object
# Dataset has nominal and numeric variables
import weka.core.converters as converters
data_dir = "data/"
data = converters.load_any_file(data_dir + "adult.csv")
data.class_is_last()


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
from weka.classifiers import Classifier
cls = Classifier(classname= "weka.classifiers.bayes.NaiveBayes" )


# No options of interest to adjust
# Build classifier on training data
cls.build_classifier(train)
#       print(cls)

#import weka.plot.graph as graph  
#graph.plot_dot_graph(cls.graph)

from weka.classifiers import Evaluation
from weka.core.classes import Random
evl = Evaluation(train)
evl.crossvalidate_model(cls, train, 10, Random(1))

print ("Kappa Score")
print (evl.kappa) # 0.50 - Not bad
print ("Evaluation Summary")
print (evl.summary()) # Accuracy: 83%

##  Test model on new data ##

evl = Evaluation(test)

from weka.classifiers import PredictionOutput
pred_output = PredictionOutput(
classname="weka.classifiers.evaluation.output.prediction.PlainText", options=["-distribution"])

evl.crossvalidate_model(cls, test, 10, Random(1), pred_output)

# View complete summary of the selected model on test data
print(evl.summary())
# The kappa statistic is 45% in this case. Not surprising given the low number of instances.
# The accuracy is 84.3%, which is fair. 
# The model does not do significantly better than random chance, 
# but the model does a fair job in classification accuracy
