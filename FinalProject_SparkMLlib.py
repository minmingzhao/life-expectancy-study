from pyspark import SparkConf, SparkContext 
from pyspark.mllib.regression import LabeledPoint
import numpy as np
import string 
from pyspark import HiveContext
conf = SparkConf().setMaster('local').setAppName('SparkMLib_FinalProject') 
sc = SparkContext(conf = conf) 

RDD = HiveContext(sc).sql('select * from finalproject_merged2')
RDD.count()

RDD.cache()
def get_mapping(rdd, idx):
	return rdd.map(lambda fields: fields[idx]).distinct().zipWithIndex().collectAsMap()
print "Mapping of the categorical feature column: %s" % get_mapping(RDD, 12) # count from 0
print "Mapping of the categorical feature column: %s" % get_mapping(RDD, 13) # count from 0
print "Mapping of the categorical feature column: %s" % get_mapping(RDD, 14) # count from 0

mappings = [get_mapping(RDD, i) for i in [12,13,14]]
def extract_features_dt(record):
	record_num_vec = [record[1],record[2],record[3],record[4],record[5],record[6],record[7],record[8],record[9],record[10],record[11],record[15],record[16],record[18],record[19]]
	record_cat_vec = [record[12],record[13],record[14]] # because we cannot directly use record[12,13,14]
	numvalues = np.array([float(field) for field in record_num_vec])
	cat_vec = np.zeros(3)
	i=0
	for field in record_cat_vec: 
		m_countrycat = mappings[i] # instead of directly call get_mapping(record,3), we create dict
		idx = m_countrycat[field]
		cat_vec[i]=idx
		i=i+1
	return np.concatenate((numvalues,cat_vec))

temp = RDD.map(lambda r: extract_features_dt(r))
print "############### Define dependent variable and features for dt model ############### " 
def extract_label(record):
	return float(record[17])

data_dt = RDD.map(lambda r: LabeledPoint(extract_label(r),extract_features_dt(r)))
first_point_dt = data_dt.first()
print "Decision Tree feature vector: " + str(first_point_dt.features)
print "Decision Tree feature vector length: " + str(len(first_point_dt.features))
	

training_dt, test_dt = data_dt.randomSplit([0.9, 0.1])
print "trainging_dt count = ", training_dt.count()
print "test_dt count = ", test_dt.count()

print "###########Start decision tree using Spark MLLib ################"
from pyspark.mllib.tree import DecisionTree
dt_model = DecisionTree.trainRegressor(training_dt,{})
preds = dt_model.predict(test_dt.map(lambda p: p.features))
actual = test_dt.map(lambda p: p.label)
true_vs_predicted_dt = actual.zip(preds)
print "Decision Tree predictions: " + str(true_vs_predicted_dt.collect())
print "Decision Tree depth: " + str(dt_model.depth())
print "Decision Tree number of nodes: " + str(dt_model.numNodes())
# data_dt.saveAsTextFile("file:///home/cloudera/MMZ/FinalProject/temp/temp_training_data_dt")

def squared_error(actual, pred): # Mean Squared Error (MSE)
	return (pred - actual)**2
def abs_error(actual, pred): # Mean Absolute Error (MAE)
	return np.abs(pred - actual)	
def squared_log_error(pred, actual): # Root Mean Squared Log Error (RMSE)
	return (np.log(pred + 1) - np.log(actual + 1))**2
mse_dt = true_vs_predicted_dt.map(lambda (t, p): squared_error(t, p)).mean()
mae_dt = true_vs_predicted_dt.map(lambda (t, p): abs_error(t, p)).mean()
rmsle_dt = np.sqrt(true_vs_predicted_dt.map(lambda (t, p): squared_log_error(t, p)).mean())
print "Decision Tree for hp - Mean Squared Error: %2.4f" % mse_dt
print "Decision Tree for hp - Mean Absolute Error: %2.4f" % mae_dt
print "Decision Tree for hp - Root Mean Squared Log Error: %2.4f" % rmsle_dt

print "########### show the decision tree model ################"
print(dt_model.toDebugString()) # display the full decision tree model
#################use linear regression############################
"""mappings = [get_mapping(RDD, i) for i in [12,13,14]]
cat_len = sum(map(len, mappings)) # 27
def extract_features(record):
	cat_vec = np.zeros(cat_len)
	i=0
	step=0
	for field in record[12:15]:
		m = mappings[i] # e.g. m = mappings[i] = {u'82': 0, u'76': 1, u'70': 2}
		idx = m[field] # convert whole column to idx based on the mappings
		cat_vec[idx+step] = 1
		i=i+1
		step = step + len(m)
	record_num_vec = [record[1],record[2],record[3],record[4],record[5],record[6],record[7],record[8],record[9],record[10],record[11],record[15],record[16],record[18],record[19]]
	
	numvalues = np.array([float(field) for field in record_num_vec])
	return np.concatenate((numvalues,cat_vec))

data = RDD.map(lambda r: LabeledPoint(extract_label(r),extract_features(r)))
training, test = data.randomSplit([0.9, 0.1])
from pyspark.mllib.regression import LinearRegressionWithSGD
linear_model = LinearRegressionWithSGD.train(training, iterations=10000,step=0.000001, intercept=False)

true_vs_predicted = test.map(lambda p: (p.label, linear_model.predict(p.features)))
print "Linear Model predictions: " + str(true_vs_predicted.collect())
print(linear_model.weights)
data.saveAsTextFile("file:///home/cloudera/MMZ/FinalProject/temp/temp_training_data")
"""
