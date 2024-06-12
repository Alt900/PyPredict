from . import np, plt, os
from . import TOMLHandler as TOML
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam 
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator

def TSG(data,windowsize):
    TSG_Train=TimeseriesGenerator(data[0].values,data[0].values,length=windowsize, batch_size=1)
    TSG_Test=TimeseriesGenerator(data[1].values,data[1].values,length=windowsize, batch_size=1)
    TSG_Validation=TimeseriesGenerator(data[2].values,data[2].values,length=windowsize, batch_size=1)
    train_x, train_y = [],[]
    test_x, test_y = [],[]
    validation_x, validation_y = [], []
    for i in range(len(TSG_Train)):
        train_x.append(TSG_Train[i][0])
        train_y.append(TSG_Train[i][0])

    for i in range(len(TSG_Test)):
        test_x.append(TSG_Test[i][0])
        test_y.append(TSG_Test[i][0])

    for i in range(len(TSG_Validation)):
        validation_x.append(TSG_Validation[i][0])
        validation_y.append(TSG_Validation[i][1])

    train_x = np.array(train_x).reshape(len(train_x), windowsize)
    train_y = np.array(train_y)

    test_x = np.array(test_x).reshape(len(test_x), windowsize)
    test_y = np.array(test_y)

    validation_x = np.array(validation_x).reshape(len(validation_x), windowsize)
    validation_y = np.array(validation_y)

    return[
        train_x, train_y,
        test_x, test_y,
        validation_x, validation_y
    ]

class Model():
    def __init__(self, data, ticker, variable):#default train on volume
        self.ticker=ticker
        self.variable=variable
        self.args=TOML.load_args("Variables.toml")
        self.model=models.Sequential()

        if not(self.args["ENV"][0]["plot"]):
            _, ax = plt.subplots(4,figsize=(14,14))
            plt.tight_layout()

            G1=[i for i in range(len(data[0]))]
            G2=[i for i in range(len(data[1]))]
            G3=[i for i in range(len(data[2]))]
            self.TestGenerator=[i for i in range(len(data[0]),len(data[0])+len(data[1]))]
            self.TestLength=len(data[1])+(self.args["ML"][0]["windowsize"]-1)

            TotalSize=sum([len(data[i]) for i in range(3)])
            ReCombined=np.append(np.append(data[0].values,data[1].values),data[2].values)
            ax[0].plot(G1, data[0].values, color='r')
            ax[3].plot([i for i in range(TotalSize)], ReCombined, color='b')
            ax[3].plot(G1, data[0].values, color='r')
            ax[0].title.set_text("Training Data")
            ax[1].plot(G2, data[1].values, color='g')
            ax[3].plot(self.TestGenerator, data[1].values, color='g')
            ax[1].title.set_text("Testing Data")
            ax[2].plot(G3, data[2].values, color='b')
            ax[2].title.set_text("Validation Data")
            ax[3].title.set_text("Full Dataset")
            plt.savefig(f"Graphs\\{ticker}_{variable}_LSTM_TTV.png")
            plt.close()
            _, self.ax = plt.subplots(2,figsize=(10,10))
            #prepare for prediction plot
            self.ax[0].title.set_text(f"Actual {self.variable}")
            self.ax[1].title.set_text(f"Predicted {self.variable}")
            self.ax[0].plot(self.TestGenerator, data[1], color='g')
            print(data[1].shape)

        data=TSG(data,self.args["ML"][0]["windowsize"])
        self.train_x, self.train_y = data[0], data[1]
        self.test_x, self.test_y = data[2], data[3]
        self.validation_x, self.validation_y = data[4], data[5]

        if self.args["ML"][0]["savemodel"] and not(os.path.isdir("Models")):
            os.mkdir("Models")


        if self.args["ML"][0]["loadmodel"]:
            try:
                self.model=tf.keras.models.load_model(f"Models\\{ticker}_{variable}.keras")
            except ValueError:
                print(f"No model found for {ticker} variable {variable}")
                exit()

        else:
            if self.args["ML"][0]["CNNAuxillary"]:
                self.model.add(layers.InputLayer((1,self.args["ML"][0]["windowsize"])))
                #CNN
                self.model.add(layers.LSTM(64))
                self.model.add(layers.Dense(8, 'relu'))
                self.model.add(layers.Dense(1, 'linear'))
            else:
                self.model.add(layers.InputLayer((self.args["ML"][0]["windowsize"],1)))
                self.model.add(layers.LSTM(64, return_sequences=True))
                self.model.add(layers.Dense(20, 'relu'))
                self.model.add(layers.Dense(1, 'linear'))


        print(self.model.summary())

    def train(self):

        self.model.compile(
            loss='mse', 
            optimizer = Adam(learning_rate = self.args["ML"][0]['learning_rate']), 
            metrics = [RootMeanSquaredError()]
        )
        cp = ModelCheckpoint(f"Models\\{self.ticker}_{self.variable}.keras",save_best_only=True)
        self.model.fit(
            self.train_x,
            self.train_y, 
            validation_data=(self.validation_x,self.validation_y),
            epochs = self.args["ML"][0]["epochs"],
            callbacks=[cp]
        )
        if self.args["ML"][0]["savemodel"]:
            self.model.save(f"Models\\{self.ticker}_{self.variable}.keras")

    def predict(self):
        predicted = self.model.predict(self.test_x).flatten()
        if not(self.args["ENV"][0]["plot"]):
            self.ax[1].plot([x for x in range(len(predicted))], predicted, color='g')
            plt.savefig(f"Graphs\\{self.ticker}_{self.variable}_LSTM_Prediction.png")
            plt.close()
        return predicted