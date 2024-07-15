from . import plt
from .API_Interface import args

def graph_df(df): 
    print("Graphing Open, High, Low, and Closing prices...")
    for ticker in df:
        currentplot=0
        try:
            XAxis = [x for x in range(len(df[ticker]["volume"]))]
        except TypeError:
            print(ticker)
            print(df[ticker])
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

def graph_prediction(time_series,predicted_vector):
    XAxis = [x for x in range(len(time_series))]
    plt.tight_layout()
    _, axes = plt.subplots(3,figsize=(16,16))
    axes[0].plot(
        XAxis,
        time_series,
        color='r'
    )
    axes[0].title.set_text(f"Original dataset")
    PredictedXAxis=[x for x in range(len(XAxis),len(predicted_vector)+len(XAxis))]

    axes[1].plot(
        PredictedXAxis,
        predicted_vector,
        color='g'
    )

    axes[1].title.set_text(f"Predicted dataset")

    FullXAxis = XAxis+PredictedXAxis
    FullYAxis = time_series + predicted_vector
    axes[2].plot(
        FullXAxis,
        FullYAxis,
        color='b'
    )
    axes[2].title.set_text(f"Original and predicted dataset")
    plt.savefig(f"Graphs\\orig_predicted_data.png")
    plt.close()

def graph_column(column,name):
    plot=column.plot()
    plot.savefig(name)
    plot.close()