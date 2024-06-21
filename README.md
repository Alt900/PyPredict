# PyPredict

## Operating system restrictions
Currently the module is only supported for Windows however Linux and Mac OS file system support will be added soon

## Requirements
In order to run this module you will need to run the InstallRequirements batch file in the PyPredict directory

## Front-end setup
To import sub-modules use `from PyPredict import module` from a list of sub-modules `from PyPredict import API_Interface, DataHandler, Normalization, ML, Graphics`

### TOML file and args

### Statistics

### DataHandler

### API_Interface

### Normalization

### Graphics

### ML


Full code example:
```
from PyPredict import API_Interface, DataHandler, Normalization, ML, Graphics
from PyPredict import args
from PyPredict.API_Interface import data

###general setup###
API_Interface.download()#initiate the download of listed tickers

###graphing data###
Graphics.graph_df(data)#passing all the downloaded data 
###Machine Learning###
DH=DataHandler.ML()
Normalizer=Normalization.Normalize()
Normalizer.log_normalization()
#(test_ratio,train_ratio,validation_ratio)
data=DH.train_val_test_split((.6,.2,.2))
variables=args["ML"][0]["variables"]
#if data is a generator function
for ticker in args["ML"][0]["tickers"]:
    for (variable,y) in zip(variables,data):
        windowed = DH.Window_Data(y)
        LSTM=ML.LSTM(windowed,variable,ticker)
        LSTM.train()
        LSTM.predict()

#if data is not a generator function
LSTM=ML.LSTM(data,"open","NVDA")
LSTM.train()
LSTM.predict()
```