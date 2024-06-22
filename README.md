# PyPredict
![](logo.png)
## Operating system restrictions
Currently the module is only supported for Windows however Linux and Mac OS file system support will be added soon


## Requirements
In order to run this module you will need to run the `InstallRequirements.bat` batch file in the PyPredict directory


## Front-end setup
To import sub-modules use `from PyPredict import module` from a list of sub-modules `from PyPredict import API_Interface, DataHandler, Normalization, ML, Graphics`. The following is a full code example of the front end and different use cases:
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
        Graphics.graph_TTV(
            [LSTM.test,
            LSTM.train,
            LSTM.validation],
            ticker, variable
        )
        LSTM.train()
        prediction=LSTM.predict()
        Graphics.graph_prediction(prediction, LSTM.test, ticker, variable)

#if data is not a generator function
LSTM=ML.LSTM(data,"open","NVDA")
LSTM.train()
LSTM.predict()
```

### TOML file and args


The `Variables.toml` file contains a wide array of tunable parameters that are used throughout the code base. The file's variables are split into three different sections `[[ENV]]` for basic environmental variables, `[[ML]]` for machine learning parameters, and `[[STATS]]` for statistical parameters. The TOML file should be located in the current working directory of your local project, if one is not available a file will be generated in the same current working directory with what type of variable should be provided for each. Here is an example of what a generated TOML file will look like:
```
[[ENV]]
Omit_Cache = bool
tickers=list
normalization=str
timespan=str
from_=str YYYY-MM-DD
to=str YYYY-MM-DD
polygon_token = str


[[ML]]
tickers=str/list
variables=str/list
batch_size = int
windowsize = int
learning_rate = float
CNNAuxillary = bool
loadmodel = bool
savemodel = bool
epochs = int


[[Statistics]]
tickers=str/list
variables=str/list
```

And an example of what a fully filled out TOML file:

```

[[ENV]]
Omit_Cache = false
tickers=[
    "NVDA",
    "AAPL",
    "TTOO",
    "LMT",
    "BA"
]
##OR
tickers="LMT"

normalization="MinMax"
timespan="minute"
from_="2024-05-01"
to="2024-06-01"
polygon_token = "token string"

[[ML]]
tickers=[
    "NVDA",
    "AAPL"
]
##OR
tickers="LMT"

variables=["open","volume"]
##OR
variables="open"

batch_size = 32 
windowsize = 5
learning_rate = 0.0001
CNNAuxillary = false
loadmodel = false
savemodel = true 
epochs = 10

[[Statistics]]
tickers=["LMT","BA"]
##OR
tickers="LMT"

variables=["open","low"]
##OR
variables="open"
```

### Automated environment initialization


When the module is called for the first time from a local project directory the `__init__.py` file will generate any necessary directories that are not found in cwd such as `JSON_Data`, `Graphs`, and/or `Models` if the `savemodel` variable is set to `true`. Alongside directories `__init__.py` will generate a TOML file if one does not exist in cwd.


### API_Interface
In order to get started with a fresh environment, you will need to import `from PyPredict import API_Interface` and call the `download` function as `API_Interface.download()`. From there the backend will handle the downloads and store the resulting Polygon data in `cwd\\JSON_data\\ticker_data.json`.


#### A warning about Polygon
Polygon is a free to use API that retrieves consistent market data for any given company ticker however on the free version you are only limited to 5 API requests per minute. This can get tedious with large numbers of company tickers in the TOML file and if surpassed the module will throw `ResponseError('too many 429 error responses')`. This can be maneuvered around if you wait a minute, re-initialize the environment, and use the cache to skip over already downloaded stock data. This method however is pretty annoying to deal with [if you don't have the paid version](https://polygon.io/pricing), support for yahoo finance will be coming in later releases of the module but as of now polygon provided comprehensive market data (with limited API calls) for free with their own stable API framework unlike other platforms like Alpaca.


### DataHandler
The data handler library only has one class right now for machine learning with a statistics class coming in later versions. To import the sub-module use `from PyPredict import DataHandler` and instantized with `DH=DataHandler.ML()`.With only two machine learning based functions `train_val_test_split` and `Window_Data` the setup is just as simple as the API_Interface module. Importing the sub-module with `from pyPredict import DataHandler` the two methods can be called as `data=DH.train_val_test_split((.6,.2,.2))` and `windowed = DH.Window_Data(data)`. The `train_val_test_split` method has three different ways it is handled. Since the [[ML]] ticker and variable args can accept either a string or a list of strings the function should either be instantized once outside of a loop, iteratively instantized over a single loop if the ticker or variable argument is a list, or iteratively instantized if the ticker and variable arguments are lists. Here are a few examples for each:


Single instance:


```
data=DH.train_val_test_split((.6,.2,.2),ticker,variable)
```


Iteratively instantized, single loop:
```
for ticker in args["ML"][0]["tickers"]: #or for variable in args["ML"][0]["variables"]
    data=DH.train_val_test_split((.6,.2,.2),ticker,variable)
```


Iteratively instantized, double loop:
```
for ticker in args["ML"][0]["tickers"]:
    for variable in args["ML"][0]["variables"]:
        data=DH.train_val_test_split((.6,.2,.2),ticker,variable)
```


Whichever version is used, the `Window_Data` method will accept the `data` variable with `windowed = DH.Window_Data(data)`.

### Normalization
This sub-module has two classes, `Normalize` and `DeNormalize`, if you want meaningful data to be extracted you will have to instantize both classes as this sub-module imports the pandas dictionary, makes the dataframe a global, and exports its changes to the rest of the module such as ML to make it easier for a LSTM to analyze a time series. As with the rest of the sub-modules this sub-module will be imported as `from PyPredict import Normalization` and the two classes instantized as `normalizer=Normalization.Normalize()` and `denormalizer=Normalization.DeNormalize()`. With `normalizer` and `denormalizer` you can convert the dataset back and forth between a normalized and regular dataset using any of the following methods with their respective classes:


(Note that all methods are spelled the exact same for `Normalize` and `DeNormalize`)


``MinMax``


``log_normalization``


### Graphics
Rather than classes for this sub-module you can simply call methods directly as each method is specifically tailored around a specific graphical rendering for everything from the pandas dataframe, testing, training, and validation data, and LSTM predictions. As of now the Graphics only support those three methods through `graph_df`, `graph_TTV`, and `graph_prediction` respectively. The `graph_df` method only accepts a pandas dataframe as its only argument. `graph_TTV` accepts split testing, training, and validation data generated by `PyPredict.DataHandler.ML.train_val_test_split`, the ticker, and variable the LSTM is being trained off of. Similarly, `graph_prediction` accepts a windowed prediction from a LSTM output, the actual data (training data from `train_val_test_split`), the ticker, and variable. 

As of now the graphical library only supports uni-variate models and will only accept a single ticker and variable 

### ML
As of now the ML sub-module only has one class and one univariate LSTM available. The univariate model can be initialized by instantizing the LSTM class by importing `from PyPredict import ML` then instantized `LSTM=ML.LSTM(windowed,variable,ticker)`. From there you can either call `LSTM.train()` then `LSTM.predict()` to train the model then compare prediction results to the testing data or call both then graph the resulting generated data through `graph_TTV` and `graph_prediction` respectively. 