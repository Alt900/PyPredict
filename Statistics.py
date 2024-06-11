from . import DataHandler, pd
import statsmodels.api as sm
import statsmodels
from scipy import stats
import matplotlib.pyplot as plt

#try to implement most in R


#add descriptive statistics calculation for JSON and add it to each interval in bars


#Skewness- the asymmetry of the data, how out of line some data is in comparison to the others


#mean - average of a distribution


#median - a value at the middle of the distribution that separates it at the 50th percentile  
#so that half the data is below the median and half the data is above the median
#the median is not as influenced by outliers and skewed data unlike the mean and mode,
#so it it a more robust statistical tool


#mode - the most commonly occurring value


#measures of spread - how much the data tends to diverge from the typical value


#range - the distance between the minimum and maximum values


#quartiles, adjusts different xth percentiles the median is calculating for.
#for 25th percentile quantile(0.25) would be used, for the 2nd percentile quantile(0.02) would be used...ect


#under pandas you can use df.describe() to return the five number quantiles, being medians of
#0, 0.25, 0.50, 0.75, and 1
#a 0 quartile describes the minimum median while a quartile of 100 describes the maximum value


#for IQR or interquartile range is the calculated distance between the 3rd and 1st quartile
#quantile 0.75 - quartile 0.25


#variance - the average of the squared differences that values have from the mean
#take the mean for every value, subtract that value from the mean, square it, and average all the squared differences
#df.var() on pandas


#standard deviation - the square root of the variance, it is expressed in terms of the same unit of the measured variable,
#unlike the variance that is expressed in terms of squared units
#df.std() on pandas


#naturally since both of these are based off the mean, they are susceptible to extreme outliers and skewed data


#median absolute deviation, a possible solution to variance and standard deviation. It starts with the same process as variance,
#taking each data point, subtracting it from the median this time, and applying absolute to those values,
#and find the median of those differences
#abs(df[datapoint].median()-df[datapoint]).median() for pandas


#kurtosis - measures how much data is in the tails of a distribution vs the center of the distribution
#df[datapoint].skew() # check skewness
#df[datapoint].kurt() # check kurtosis

class Statistics():
    def __init__(self):
        self.DataHandler = DataHandler.JSON()

    def seasonal_decompose(data):
        return statsmodels.tsa.seasonal.seasonal_decompose(data,model="additive")
