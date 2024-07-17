# PyPredict




## Requirements
In order to run this module you will need to run the `InstallRequirements.py` python file in the PyPredict directory, this will install all necessary modules through a `requirements.txt` file.




## Front-end setup
To import sub-modules use `from PyPredict import module` from a list of sub-modules `from PyPredict import API_Interface, DataHandler, Normalization, ML, Graphics, Statistics`. The following is a full code example of the front end and different use cases:
```
from PyPredict import API_Interface, DataHandler, Normalization, ML, Graphics, Statistics
from PyPredict.API_Interface import data

###general setup###

API_Interface.download()#initiate the download of listed tickers
Graphics.graph_df(data)#graph the entire dataset passing all the downloaded data 

#normalization
Normalizer=Normalization.Normalize("NVDA","open")
Normalizer.Logarithmic()#either pass a pandas column or a list of values
DeNormalizer=Normalization.DeNormalize("NVDA","open")


#perform some feature engineering
FE=DataHandler.Feature_Engineering(data["NVDA"]["open"],min=0.41,max=0.77)
decomposed=FE.seasonal_decompose(period=30)
MAD_result=FE.MAD(threshold=1.9)

#prepare the data for the LSTM, construct the model, and begin training
DH=DataHandler.LSTM_Prep(data["NVDA"]["open"],ratio=(.7,.3))
LSTM=ML.Univariate_LSTM(DH.TTV_x, DH.TTV_y,cell_count=70,output_size=1,filename="NVDA_open")
LSTM.train()

#extract a prediction and de-normalize the predicted set and the testing set
prediction=LSTM.predict(DH.TTV_x[1])#testing data
de_normalized_test_y = DeNormalizer.External_Sets["Logarithmic"](DH.TTV_y[1].flatten())
de_normalized_test_x = DeNormalizer.External_Sets["Logarithmic"](DH.TTV_x[1].flatten())
de_normalized_prediction = DeNormalizer.External_Sets["Logarithmic"](prediction)
Graphics.graph_prediction_overlapped(de_normalized_test_y,de_normalized_prediction)
Graphics.graph_prediction_merged(de_normalized_test_x,de_normalized_prediction)

#perform linear regression with a coefficient of volume and a dependent variable of opening prices
linear_regression=Statistics.Ordinary_least_squares(data["NVDA"]["volume"],data["NVDA"]["open"],pval_threshold=0.05)
print(linear_regression.summary)
linear_regression.null_hypothesis()
linear_regression.diagnose_heteroskedasticity("whites")
```


### TOML file and args




The `Variables.toml` file contains a wide array of tunable parameters that are used throughout the code base. The file's variables are split into two different sections `[[ENV]]` for basic environmental variables and `[[ML]]` for machine learning parameters. The TOML file should be located in the current working directory of your local project, if one is not available a file will be generated in the same current working directory with what type of variable should be provided for each. Here is an example of what a generated TOML file will look like:
```
[[ENV]]
Omit_Cache = bool
tickers=str || list
from_=list, format [YYYY,MM,DD]
to=list, format [YYYY,MM,DD]
alpaca_key = str
alpaca_secret = str
[[ML]]
batch_size = int
windowsize = int
learning_rate = float
load_model = bool
save_model = bool
epochs = int
plot_loss=bool
```


And an example of what a fully filled out TOML file:


```
[[ENV]]
Omit_Cache = false
tickers=[
    "NVDA,
    "AAPL",
    "MSFT"
]
from_=[2024,5,1]
to=[2024,7,1]
alpaca_key = "key"
alpaca_secret = "secret"
[[ML]]
batch_size = 32
windowsize = 5
learning_rate = 0.0001
load_model = false
save_model = true
epochs = 5
plot_loss=true
```


### Automated environment initialization




When the module is called for the first time from a local project directory the `__init__.py` file will generate any necessary directories that are not found in cwd such as `JSON_Data`, `Graphs`, and/or `Models` if the `savemodel` variable is set to `true`. Alongside directories `__init__.py` will generate a TOML file if one does not exist in cwd.




### API_Interface
In order to get started with a fresh environment, you will need to import `from PyPredict import API_Interface` and call the `download` function as `API_Interface.download()`. From there the backend will handle the downloads and store the resulting Alpaca data in `cwd\\JSON_data\\ticker_data.json`. If any data exists in the `JSON_data` directory and `OmitCache` in the TOML is set to `false` then the download for that stock will be skipped to save load time and internet. Even if all stocks are downloaded, `download` must be called in order to load all the cached JSON into a pandas dataframe for future use.




### DataHandler
To import the sub-module use `from PyPredict import DataHandler` and instantized with `DH=DataHandler.LSTM_Prep()`. The test/train split and dataset windowing is handled automatically on instantization so there is no need to call any other functions to split or window the dataset. The input-label data pair is stored as `TTV_x` and `TTV_y` respectively in the instantized `DataHandler.LSTM_Prep()` variable, `TTV_x` and `TTV_y` are lists of each dataset and each input-label pair is indexed based on which dataset they belong to. For example if you wanted the training dataset's input data and label data it would be accessed as such: `DH.TTV_x[0]` and `DH.TTV_y[0]`


### Normalization
This sub-module has two classes, `Normalize` and `DeNormalize`, if you want meaningful data to be extracted you will have to instantized both classes, one to normalize the dataset initially given the targeted ticker and variable and one to de-normalize the dataset. Both classes accept a `ticker` and `variable` argument to normalize or denormalize a specific dataset. The DeNormalizer has the exact same function names as the Normalizer and operates the same way Normalizer does, however it also accepts external datasets through the `External_Sets` function dict. External arrays can be denormalized as:
`de_normalized_array = DeNormalizer.External_Sets["Logarithmic"](array)`.




### Statistics
Alongside machine learning, PyPredict offers some standard statistical measurements like standard deviation, mean, median, and mode. As of now there is only one linear regression model under the `Ordinary_Least_Squares` class supporting two test methods: `null_hypothesis` and `diagnose_heteroskedasticity`. `diagnose_heteroskedasticity` supports either a `whites`, `breusch-pagan`, or `goldfeld-quandt` heteroskedasticity test.




### Graphics
Graphics currently only contains three functions, `graph_df`, `graph_prediction_overlapped` and `graph_prediction_merged`. `graph_df` plots the entire downloaded/loaded dataset while `graph_prediction_overlapped` and `graph_prediction_merged` both plot a prediction from the LSTM. The two serve different purposes for prediction visualization, `graph_prediction_overlapped` places the real-value dataset or `y_test` over the prediction result of the LSTM to see how the two compare and differ. `graph_prediction_merged` on the other hand combines the training dataset or `x_train` and attaches the result to the end of the training dataset to visualize how the prediction results fit with the training dataset.


`graph_prediction_merged`:
![](train_predict_merged.png)


`graph_prediction_overlapped`:
![](Test_predict_overlap.png)




### ML
Currently, the only machine learning model available is a univariate LSTM consisting of 2 LSTM layers, 2 dropout layers, and a final linear layer. The model can be initialized by instantizing the class `LSTM=ML.Univariate_LSTM(DH.TTV_x, DH.TTV_y,cell_count=70,output_size=1,filename="company_variable")` and trained with `LSTM.train()`. A prediction can be extracted by calling `LSTM.predict()` and a flattened 1D prediction will be returned.