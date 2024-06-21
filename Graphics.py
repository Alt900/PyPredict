from . import plt,np
from .API_Interface import args

def graph_TTV(TTV_data,ticker,variable):
    _, ax = plt.subplots(4,figsize=(14,14))
    plt.tight_layout()

    G1=[i for i in range(len(TTV_data[0]))]
    G2=[i for i in range(len(TTV_data[1]))]
    G3=[i for i in range(len(TTV_data[2]))]

    TotalSize=sum([len(TTV_data[i]) for i in range(3)])
    ReCombined=np.append(np.append(TTV_data[0].values,TTV_data[1].values),TTV_data[2].values)

    ax[0].plot(G1, TTV_data[0].values, color='r')
    ax[3].plot([i for i in range(TotalSize)], ReCombined, color='b')
    ax[3].plot(G1, TTV_data[0].values, color='r')
    ax[0].title.set_text("Training TTV_data")
    ax[1].plot(G2, TTV_data[1].values, color='g')
    ax[3].plot([i for i in range(len(TTV_data[0]),len(TTV_data[0])+len(TTV_data[1]))], TTV_data[1].values, color='g')
    ax[1].title.set_text("Testing TTV_data")
    ax[2].plot(G3, TTV_data[2].values, color='b')
    ax[2].title.set_text("Validation TTV_data")
    ax[3].title.set_text("Full TTV_dataset")
    plt.savefig(f"Graphs\\{ticker}_{variable}_LSTM_TTV.png")
    plt.close()

def graph_prediction(predicted,actual,variable,ticker):
    _, ax = plt.subplots(2,figsize=(10,10))
    ax[0].title.set_text(f"Actual {variable}")
    ax[1].title.set_text(f"Predicted {variable}")
    ax[0].plot([x for x in range(len(actual))], actual, color='g')
    ax[1].plot([x for x in range(len(predicted))], predicted, color='g')
    plt.savefig(f"Graphs\\LSTM_univariate_{ticker}_{variable}_prediction.png")
    plt.close()

def graph_df(df): 
    print("Graphing Open, High, Low, and Closing prices...")
    for ticker in df:
        currentplot=0
        XAxis = [x for x in range(len(df[ticker]["volume"]))]
        plt.tight_layout()
        _, axes = plt.subplots(7,figsize=(16,16))
        for variable in df[ticker]:
            axes[currentplot].plot(
                XAxis,
                df[ticker][variable],
                color='g'
            )
            axes[currentplot].title.set_text(f"{variable}_{ticker}")
            currentplot+=1
        plt.savefig(f"Graphs\\{ticker}_data.png")
        plt.close()