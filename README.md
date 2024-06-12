# PyPredict
PyPredict is a statistical and machine learning framework built off of Tensorflow, Numpy, Pandas, and Statsmodelsto deliver streamlined data analysis

## Requirements 
If you do not have any of the following installs then run the following commands for the respective libraries

`pip3 install numpy`

`pip3 install tensorflow`

`pip3 install pandas`

`pip3 install matplotlib`

`pip3 install polygon`

`pip3 install toml`

`pip3 install datetime`

## Getting started
There are only a handful of objects 
```
import PyPredict

Stats=pypredict.Statistics.Statistics()
args = PyPredict.TOMLHandler.load_args("Variables.toml") #load arguments from the TOML file

#split the scraped JSON into training, testing, and validation
#for time series prediction given a ratio tuple accepting
#raw percentage values
Stats.DataHandler.train_val_test_split((0.7,0.2,0.1))
#for a 70%, 20%, and 10% split
```
