from . import tf
from .API_Interface import args
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam 
from tensorflow.keras.metrics import RootMeanSquaredError

class LSTM():
    def __init__(self, data, ticker, variable):#default train on volume
        self.ticker=ticker
        self.variable=variable
        self.model=models.Sequential()

        self.train_x, self.train_y = data[0], data[1]
        self.test_x, self.test_y = data[2], data[3]
        self.validation_x, self.validation_y = data[4], data[5]

        if args["ML"][0]["loadmodel"]:
            try:
                self.model=tf.keras.models.load_model(f"Models\\{ticker}_{variable}.keras")
            except ValueError:
                print(f"No model found for {ticker} variable {variable}")
                exit()

        else:
            if args["ML"][0]["CNNAuxillary"]:
                self.model.add(layers.InputLayer((1,args["ML"][0]["windowsize"])))
                self.model.add(layers.LSTM(64))
                self.model.add(layers.Dense(8, 'relu'))
                self.model.add(layers.Dense(1, 'linear'))
            else:
                self.model.add(layers.InputLayer((args["ML"][0]["windowsize"],1)))
                self.model.add(layers.LSTM(32, return_sequences=True))#reconstruct into a VAE-LSTM
                self.model.add(layers.Dense(20, 'relu'))
                self.model.add(layers.Dense(1, 'linear'))


        print(self.model.summary())

    def train(self):

        self.model.compile(
            loss='mse', 
            optimizer = Adam(learning_rate = args["ML"][0]['learning_rate']), 
            metrics = [RootMeanSquaredError()]
        )
        checkpoint = ModelCheckpoint(f"Models\\{self.ticker}_{self.variable}.keras",save_best_only=True)
        self.model.fit(
            self.train_x,
            self.train_y,
            validation_data=(self.validation_x,self.validation_y),
            epochs = args["ML"][0]["epochs"],
            callbacks=[checkpoint]
        )
        if args["ML"][0]["savemodel"]:
            self.model.save(f"Models\\{self.ticker}_{self.variable}.keras")

    def predict(self):
        return self.model.predict(self.test_y).flatten()

    def __add__(self,UNVM1,UNVM2):
        #Uni_Variate_1 + Uni_Variate_2 = Multi-Model LSTM
        pass