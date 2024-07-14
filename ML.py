from .API_Interface import args
from . import torch

window_size = args["ML"][0]["windowsize"]
plot_loss = args["ML"][0]["plot_loss"]
epochs = args["ML"][0]["epochs"]

if plot_loss:
    from . import plt

device = (
    "cuda"#NVIDIA GPU
    if torch.cuda.is_available()
    else "mps"#AMD GPU
    if torch.backends.mps.is_available()
    else "cpu"
)

##add a class of model diagnostic tools that includes:
#f1 score for comparing different models with different configurations on the same dataset
#

class UNI_Backend(torch.nn.Module):
    def __init__(self, cell_count=32, output_size=1, layers=2):
        super().__init__()
        
        self.layers=layers
        self.cell_count=cell_count
        self.LSTM_1=torch.nn.LSTM(1,cell_count,layers,batch_first=True)
        self.dropout_1=torch.nn.Dropout(p=0.2)
        self.LSTM_2=torch.nn.LSTM(cell_count,cell_count,layers,batch_first=True)
        self.dropout_2=torch.nn.Dropout(p=0.05)
        self.linear=torch.nn.Linear(cell_count,output_size)

    def forward(self, x, future=0):
        predicted = []
        batch_size = x.shape[0]

        hidden_state_1 = torch.zeros(self.layers, batch_size, self.cell_count, dtype=torch.float32)
        cell_state_1 = torch.zeros(self.layers, batch_size, self.cell_count, dtype=torch.float32)
        hidden_state_2 = torch.zeros(self.layers, batch_size, self.cell_count, dtype=torch.float32)
        cell_state_2 = torch.zeros(self.layers, batch_size, self.cell_count, dtype=torch.float32)

        hidden_state_1, cell_state_1 = self.LSTM_1(x,(hidden_state_1,cell_state_1))
        hidden_state_1 = self.dropout_1(hidden_state_1)
        hidden_state_2, cell_state_2 = self.LSTM_2(hidden_state_1,(hidden_state_2,cell_state_2))
        hidden_state_2 = self.dropout_2(hidden_state_2)


        output=self.linear(hidden_state_2)
        predicted.append(output)

        for _ in range(future):
            hidden_state_1, cell_state_1 = self.LSTM_1(output,(hidden_state_1,cell_state_1))
            output=self.linear(hidden_state_1)
            predicted.append(output)

        predicted=torch.cat(predicted, dim=1)
        return predicted

class Univariate_LSTM():
    def __init__(self, X, Y, cell_count=20, output_size=1,filename=None):
        
        self.filename=filename

        self.model = UNI_Backend(cell_count=cell_count, output_size=output_size)
        if args["ML"][0]["load_model"]:
            self.model=self.model.load_state_dict(torch.load(f"Models\\{filename}.pt"))
            self.model.eval()

        self.optimizer = torch.optim.Adam(self.model.parameters())
        self.LossFunction = torch.nn.MSELoss()

        self.train_x, self.train_y = torch.from_numpy(X[0]), torch.from_numpy(Y[0])
        self.test_x, self.test_y = torch.from_numpy(X[1]), torch.from_numpy(Y[1])

        self.TrainingLoader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(self.train_x,self.train_y),
            batch_size=args["ML"][0]["batch_size"],
            shuffle=True
        )

    def train(self):
        for epoch in range(epochs):
            for X_train, Y_train in self.TrainingLoader:
                self.model.train()#sets the model training mode for dropout and batch norm layers
                y_predicted = self.model(X_train)#get predicted time step -1 from last linear single-prediction cell
                loss=self.LossFunction(y_predicted, Y_train)#calculate the MSE loss between predicted and real values
                self.optimizer.zero_grad()#initialize the gradient's at zero
                loss.backward()#backpropagate the loss through the network
                self.optimizer.step()#begin optimization

            print(f"epoch - {epoch}\nRMSE loss: {loss}")
            if plot_loss:
                if epoch==0:
                    compiled_loss=[]
                compiled_loss.append(loss)

            if epoch%100!=0:#if the epoch is 0, the first training iteration, then the gradient will be zerod/initialized for backprop
                continue

            self.model.eval()#sets the model training mode for dropout and batch norm layers

            with torch.no_grad():
                y_predicted = self.model(torch.tensor(self.train_x))#perform a single forward pass on training data
                train_rmse=(torch.sqrt(self.LossFunction(y_predicted,self.train_y)))#root mean squared error for training data
                y_predicted=self.model(self.test_x)#perform a single forward pass on testing data
                test_rmse=(torch.sqrt(self.LossFunction(y_predicted,self.test_y)))#root mean squared error for testing data

            print(f"epoch - {epoch}\nTrain RMSE: {train_rmse}\nTest RMSE: {test_rmse}")

        if plot_loss:
            for i in range(epochs):
                c='g' if compiled_loss[i-1]>compiled_loss[i] else 'r'
                if i!=0:
                    plt.plot(
                        [i-1,i],
                        [float(compiled_loss[i-1]),float(compiled_loss[i])],
                        color=c
                    )
                else:
                    plt.scatter(
                        i,
                        float(compiled_loss[i]),
                        color=c
                    )
            plt.title("RMSE loss")
            plt.savefig(f"Graphs\\RMSE_Loss.png")
            plt.close()

        if args["ML"][0]["save_model"]:
            torch.save(self.model.state_dict(),f"Models\\{self.filename}.pt")

    def predict(self,x):
        return self.model(torch.tensor(x)).flatten().tolist()