from statsmodels.tsa.seasonal import seasonal_decompose as SD

global data
from .API_Interface import data
from . import args

tickers=args["Statistics"][0]["tickers"]
variables=args["Statistics"][0]["variables"]


def seasonal_decompose():
    for x in data:
        return SD(data[x],model="additive",period=data[x].shape[0])

def Central_Tendacy():
    for ticker in tickers:
        for variable in variables:
            yield [
                data[ticker][variable].mean(),
                data[ticker][variable].median(),
                data[ticker][variable].mode()[0],]
            
def standard_deviation(ReturnMean=False):
    pass