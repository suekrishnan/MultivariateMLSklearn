#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  9 23:51:50 2017

@author: suvarnakrishnan
"""

from pyspark.ml import Pipeline
from pyspark.ml.regression import DecisionTreeRegressor, LinearRegression
from pyspark.ml.feature import VectorIndexer, VectorAssembler, StringIndexer
from pyspark.ml.evaluation import RegressionEvaluator, MulticlassClassificationEvaluator
from pyspark.sql import SparkSession
import numpy as np
import pandas as pd
from pyspark.sql.functions import UserDefinedFunction
from pyspark.sql.types import StructType, FloatType, IntegerType, StringType
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.classification import RandomForestClassifier


from pyspark.ml.regression import GBTRegressor
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.regression import DecisionTreeRegressor
from pyspark.ml.regression import LinearRegression
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import StandardScaler
from pyspark.ml.classification import GBTClassifier

from datetime import date



#StoreId,WeekStartDate,DemandHrs,SchedHrs,SchedEffectiveness,FTCount,PTCount,FixedHrs
def getData(path):
    spark = SparkSession.builder.appName("PTMLTest").master("local[4]").getOrCreate()
    #path = "/Users/suvarnakrishnan/Harvard/trudata.csv"
    myschema = StructType().add("StoreId", StringType()).add("weekStartdate",IntegerType()).add("demand",FloatType())\
    .add("schedule",FloatType()).add("eff",FloatType()).add("ft",IntegerType())\
    .add("pt",IntegerType()).add("fixedhrs",FloatType())
    data = spark.read.csv(path,myschema,header=True)
    data.show(10)
    return data

def GetRegressionModels(m):
    if m in ["GBT"]:
        rm = GBTRegressor(labelCol="indexedpt", featuresCol="features")
    elif m in["RF"]:
        rm = RandomForestRegressor(labelCol="indexedpt", featuresCol="features")
    elif m in["DT"]:
        rm = DecisionTreeRegressor(labelCol="indexedpt", featuresCol="features")
    elif m in ["LR"]:
        rm = LinearRegression(labelCol="indexedpt", featuresCol="features")
    return rm
def GetClassificationModels(c):
    if c in ["GBT"]:
        cm = GBTClassifier(labelCol="indexedpt", featuresCol="features")
    elif c in["RF"]:
        cm = RandomForestClassifier(labelCol="indexedpt", featuresCol="features")
    elif c in["DT"]:
        cm = DecisionTreeClassifier(labelCol="indexedpt", featuresCol="features")
    return cm

def TrainModel(modelname,labelIndexer,featureIndexer,trainingData,testData):
    # Chain indexer and tree in a Pipeline
    pipeline = Pipeline(stages=[labelIndexer, featureIndexer, GetRegressionModels(modelname)])


    print(pipeline.getStages)
    # Train model.  This also runs the indexer.
    regressionmodel = pipeline.fit(trainingData)
    #classificationmodel = pipeline1.fit(trainingData)
    

    predictions = regressionmodel.transform(testData)
    predictions.show(10)
    return predictions
def TrainClassificationModel(modelname,labelIndexer,featureIndexer,trainingData,testData):
    # Chain indexer and tree in a Pipeline
    p = Pipeline(stages=[labelIndexer, featureIndexer, GetClassificationModels(modelname)])


    print(p.getStages)
    # Train model.  This also runs the indexer.
    classificationmodel = p.fit(trainingData)
    #classificationmodel = pipeline1.fit(trainingData)
    

    predictions = classificationmodel.transform(testData)
    predictions.show(10)
    return predictions


def Predictionmetrics(testpredictions):
    # Select (prediction, true label) and compute test error
    evaluator = MulticlassClassificationEvaluator(labelCol="pt",predictionCol="prediction")
    #precision = evaluator.evaluate(predictions)
    #print("Precision : %s " % precision)

    # Select (prediction, true label) and compute test error
    evaluator = RegressionEvaluator(
                                    labelCol="indexedpt", predictionCol="prediction", metricName="rmse")
    rmse = evaluator.evaluate(testpredictions)
    print("Root Mean Squared Error (RMSE) on test data = %g" % rmse)

    # Select (prediction, true label) and compute test error
    evaluator2 = RegressionEvaluator(
                                     labelCol="indexedpt", predictionCol="prediction", metricName="mse")
    mse = evaluator2.evaluate(testpredictions)
    print(" Mean Squared Error (MSE) on test data = %g" % mse)
    # Select (prediction, true label) and compute test error
    evaluator3 = RegressionEvaluator(
                                     labelCol="indexedpt", predictionCol="prediction", metricName="mae")
    mae = evaluator3.evaluate(testpredictions)
    print(" Mean Absolute Error (MAE) on test data = %g" % mae)
    predictionMetrics=[mse,rmse,mae]
    return predictionMetrics



def Visualization(predictions,modelname,modeltype):
    #cpredictions = classificationmodel.transform(testData)
    #cpredictions.show(10)
    # Select example rows to display.
    predictions.select("prediction", "indexedpt", "features").show(20)
    #cpredictions.select("prediction", "indexedpt", "features").show(20)


    changedTypedf = predictions.withColumn("prediction", predictions["prediction"].cast("int"))
    #changedTypedf1 = cpredictions.withColumn("prediction", predictions["prediction"].cast("int"))


    changedTypedf.show(5)
    #changedTypedf1.show(5)


    
    import matplotlib.pyplot as plt
    pdf = changedTypedf.select("prediction")
    #pdfc = changedTypedf1.select("prediction")


    predictedpt= pdf.rdd.map(list).collect()
    #predictedptc= pdfc.rdd.map(list).collect()


    pdf1=changedTypedf.select("indexedpt")
    #pdfc=changedTypedf1.select("indexedpt")


    xaxis=changedTypedf.select("weekStartdate")
    #xaxis1=changedTypedf1.select("weekStartdate")


    originalpt= pdf1.rdd.map(list).collect()
    #originalpt1= pdfc.rdd.map(list).collect()


    dates=xaxis.rdd.map(list).collect()
    #dates1=xaxis1.rdd.map(list).collect()
    plt.title('Headcount Predictions by Day : ' + modelname + " - "+modeltype)
    plt.xlabel('Dates')
    plt.ylabel('Headcount')
    #print(sorted(dates))
    plt.grid(True)
    predicted=plt.plot(sorted(dates),predictedpt, color='darkorange', linewidth=1)
    #plt.plot(sorted(dates),originalpt, color='lightblue', linewidth=1)
    original=plt.scatter(sorted(dates),originalpt, color='lightblue', marker='*')
    plt.legend([original,predicted], ['Original','Predicted'] )

    #plt.scatter(dates,originalpt, color='darkgreen', marker='*')


    #plt.scatter(dates,predictedpt, color='lightblue', marker='^')
    plt.xlim(20170101,20171101 )
    plt.show()

def bestFit(path,modelname):
    #data=getData("/Users/suvarnakrishnan/Harvard/trudata.csv")
    data=getData(path)
    featuresCols = data.columns
    data.printSchema
    print(featuresCols)
    labelIndexer = StringIndexer(inputCol="pt", outputCol="indexedpt").fit(data)
    labelindexed = labelIndexer.transform(data)
    labelindexed.show(5)
    # Automatically identify categorical features, and index them.
    # We specify maxCategories so features with > 4 distinct values are treated as continuous.
    

    assembler = VectorAssembler(
                                inputCols=['weekStartdate','demand','schedule','fixedhrs','ft'], outputCol="features"
                                )

    assembled = assembler.transform(data)

    assembled.show(5)

    featureIndexer =\
        VectorIndexer(inputCol="features", outputCol="indexedfeatures", maxCategories=4).fit(assembled)
    (trainingData, testData) = assembled.randomSplit([0.8, 0.2])


    for x in modelname:
        #modelname="GBT"
        testRegression = TrainModel(x,labelIndexer,featureIndexer,trainingData,testData)
        accuracyMetrics = Predictionmetrics(testRegression)
        Visualization(testRegression,x,"Regression")
        testClassification=TrainClassificationModel(x,labelIndexer,featureIndexer,trainingData,testData)
        Visualization(testClassification,x,"Classification")

import os 
dir_path = os.path.dirname(os.path.realpath('__file__'))
bestFit(dir_path + "/data.csv",["DT","RF"])