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
There are only a handful of objects and methods currently available in the module, the code below will instantize a statistics object that downloads the stock data, load variables from a TOML file, and split the stock data to use in machine learning. 
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
As for the machine learning aspect of the library, if you want to train multiple models based off how many variables (on single-variable LSTM model) and/or how many tickers are in the TOML file loaded through `TOML_handler` then the `train_val_test_split` function must be treated as a generator function. `train_val_test_split` operates dynamically, either the function statclly returns a single train, test, and validation pair or it produces a series of train, test, and validation pairs to be captured. There are four valid approaches to this function:
#### Constant
```
data = Stats.DataHandler.train_val_test_split((0.7,0.2,0.1))
tickers, variables = args["ML"][0]["ticker"], args["ML"][0]["variable"]
```
### Static
```
#no list type on variable or ticker in TOML

ML=PyPredict.ML.Model(data, tickers, variable)
```

### Generator, multi-variables only
```
#no list type for ticker, list type for variable

for variable_set, variable in zip(data,variables):
    ML=PyPredict.ML.Model(variable_set, tickers, variable)
```

### Generator, multi-ticker only
```
#list type for ticker, no list type for variable

for ticker_set, ticker in zip(data, tickers):
  ML=PyPredict.ML.Model(ticker_set, ticker, variable)
```

### Generator, multi-ticker and multi-variable
```
#list type for ticker and variable

for ticker_set, ticker in zip(data, tickers):
    for variable_set, variable in zip(ticker_set,variables):
        ML=PyPredict.ML.Model(variable_set, ticker, variable)
```
