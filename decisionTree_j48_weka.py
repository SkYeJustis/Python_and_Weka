"""
Decision Trees Demo - Weka
A dataset with numeric and nominal variables was used.

@author: SkYe
Dataset Description: https://archive.ics.uci.edu/ml/datasets/Adult

Main reference for python-weka-wrapper:
http://pythonhosted.org/python-weka-wrapper/
"""
import weka.core.jvm as jvm
jvm.start()


# Load data: Must be a Weka object
import weka.core.converters as converters
data_dir = "data/"
data = converters.load_any_file(data_dir + "adult.csv")
data.class_is_last()
#print(data)

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

cls = Classifier(classname="weka.classifiers.trees.J48", )


opt = { 0: ["-C", "0.2", "-M", "2"] ,
        1: ["-C", "0.1"] ,
        2: ["-C", "0.15", "-M", "3"],
        3: ["-C", "0.15", "-M", "2"]}
        
results = {}

for x in range(0, 4):
    #cls.options = ["-C", "0.2"]
    cls.options =  opt.get(x)
    
    #        print(cls.to_help())
    #        print(cls.options)
    
    
    cls.build_classifier(train)
    #       print(cls)
    
    #import weka.plot.graph as graph  
    #graph.plot_dot_graph(cls.graph)
    
    from weka.classifiers import Evaluation
    evl = Evaluation(train)
    evl.crossvalidate_model(cls, train, 10, Random(1))
    results.update({ x :["option:", opt.get(x), "kappa: ", evl.kappa] })

# View all kappa values and the corresponding options
print(results)

# ['option:', ['-C', '0.20', '-M', '2'], 'kappa: ', 0.6013608421053758] was the best
from weka.classifiers import PredictionOutput

pred_output = PredictionOutput(
classname="weka.classifiers.evaluation.output.prediction.PlainText", options=["-distribution"])

cls.options =  ["-C", "0.20", "-M", "2"]

from weka.classifiers import Evaluation
from weka.core.classes import Random
evl = Evaluation(train)
evl.crossvalidate_model(cls, train, 10, Random(1), pred_output)

# View complete summary of the selected model on train data
print(evl.summary())


# All predictions and actual values
print(pred_output.buffer_content())

#View text version of j48 model
print(cls)

# View png version of j48 model
#import weka.plot.graph as wgraph
#wgraph.plot_dot_graph(cls.graph, "j48_adult.png")

# If there is a problem with the graphViz dot plotting option
# View image of tree

# 1. Obtain .dot file text
print(cls.graph)

# 2. Download GraphViz to paste .dot file text and create .dot file as a image 
# (e.g., .png  .jpeg)
# http://www.graphviz.org/


##  Test model on new data ##
evl = Evaluation(test)
evl.crossvalidate_model(cls, test, 10, Random(1), pred_output)

# View complete summary of the selected model on test data
print(evl.summary())
# The kappa statistic is 34% in this case,
# even though the accuracy is 81%.
# The model does not do significantly better than random chance, 
# but the model does a fair job in classification accuracy

# View prediction output for model on test data
print(pred_output.buffer_content())






