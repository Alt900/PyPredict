from . import np, plt, os
from . import TOMLHandler as TOML
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam 
from tensorflow.keras.metrics import RootMeanSquaredError

class Model():
    def __init__(self, data, ticker, variable, origdata):#default train on volume
        self.ticker=ticker
        self.variable=variable
        self.args=TOML.load_args("Variables.toml")
        self.model=models.Sequential()
        
        self.train_x, self.train_y = data[0][0], data[0][1]
        self.test_x, self.test_y = data[1][0], data[1][1]
        self.validation_x, self.validation_y = data[2][0], data[2][1]


        if self.args["ML"][0]["savemodel"] and not(os.path.isdir("Models")):
            os.mkdir("Models")

        if self.args["ENV"][0]["plot"]:

            self.plot_train_x = self.train_x
            self.plot_test_x = self.test_x
            self.plot_validation_x = self.validation_x

            _,ax = plt.subplots(4,figsize=(10,10))
            for i in range(len(self.train_x)):
                ax[0].plot(range(i,i+self.args["ML"][0]["windowsize"]), self.train_x[i], color='r')
                ax[0].title.set_text("Training Data")
            for i in range(len(self.test_x)):
                ax[1].plot(range(i,i+self.args["ML"][0]["windowsize"]), self.test_x[i], color='g')
                ax[1].title.set_text("Testing Data")
            for i in range(len(self.validation_x)):
                ax[2].plot(range(i,i+self.args["ML"][0]["windowsize"]), self.validation_x[i], color='b')
                ax[2].title.set_text("Validation Data")
            for i in range(len(origdata)):
                ax[3].plot(range(len(origdata)), origdata)
                ax[3].title.set_text("Full Dataset")
            ax[0].plot(range(self.args["ML"][0]["windowsize"], self.args["ML"][0]["windowsize"]+len(self.train_y)), self.train_y, color='r')
            ax[1].plot(range(self.args["ML"][0]["windowsize"], self.args["ML"][0]["windowsize"]+len(self.test_y)), self.test_y, color='g')
            ax[2].plot(range(self.args["ML"][0]["windowsize"], self.args["ML"][0]["windowsize"]+len(self.validation_x)), self.validation_y, color='b')
            plt.savefig(f"Graphs\\{ticker}_{variable}_LSTM_TTV.png")

        self.train_x = self.train_x.reshape(self.train_x.shape[0],self.args["ML"][0]["windowsize"],1)
        self.test_x = self.test_x.reshape(self.test_x.shape[0],self.args["ML"][0]["windowsize"],1)
        self.validation_x = self.validation_x.reshape(self.validation_x.shape[0],self.args["ML"][0]["windowsize"],1)

        if self.args["ML"][0]["loadmodel"]:
            self.model=tf.load_model(f"model_{self.args['ML'][0]['ticker']}_{self.args['ML'][0]['variable']}.keras")

        else:
            if self.args["ML"][0]["CNNAuxillary"]:
                self.model.add(layers.InputLayer((1,self.args["ML"][0]["windowsize"])))
                #CNN
                self.model.add(layers.LSTM(64))
                self.model.add(layers.Dense(8, 'relu'))
                self.model.add(layers.Dense(1, 'linear'))
            else:
                self.model.add(layers.InputLayer((self.args["ML"][0]["batch_size"],self.args["ML"][0]["windowsize"])))
                self.model.add(layers.LSTM(100, return_sequences=True))
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
        print(predicted)
        if self.args["ENV"][0]["plot"]:
            _, ax = plt.subplots(2,figsize=(10,10))

            ax[0].plot(range(self.train_y.shape[0]), self.train_y, color='b')
            ax[0].title.set_text(f"Actual {self.variable}")
            ax[1].plot(range(self.train_y.shape[0]), self.train_y, color='b')
            ax[0].title.set_text(f"Predicted {self.variable}")

            ax[0].plot(range(self.train_y.shape[0],(self.test_y.shape[0]+self.train_y.shape[0])), self.test_y, color='g')

            ax[1].plot(range(self.train_y.shape[0],(len(predicted)+self.train_y.shape[0])), predicted, color='g')

            plt.savefig(f"Graphs\\{self.ticker}_{self.variable}_LSTM_Prediction.png")
        exit()
        return predicted