from . import np
global data
from .API_Interface import data, tickers

DeNormalMask={}


class Normalize():
    def __init__(self):
        self.vars=[x for x in data[tickers[0]]]

    def MinMax(self):
        for ticker in tickers:
            DeNormalMask[ticker]={}
            for variable in self.vars:
                df=data[ticker][variable]
                DeNormalMask[ticker][variable]=[df.min(),df.max()]
                data[ticker][variable]=(df-DeNormalMask[ticker][variable][0])/(DeNormalMask[ticker][variable][1]-DeNormalMask[ticker][variable][0])

    def log_normalization(self):
        for ticker in tickers:
            for variable in self.vars:
                data[ticker][variable]=np.log(data[ticker][variable])
        
class DeNormalize():
    def __init__(self):
        self.vars=[x for x in data[tickers[0]]]

    def MinMax(self):
        for ticker in tickers:
            for variable in self.vars:
                Min,Max=DeNormalMask[ticker][variable]
                denormalized=np.zeros((data[ticker][variable].shape[0],))
                for i in range(data[ticker][variable].shape[0]):
                    denormalized[i]=data[ticker][variable][i]*(Max-Min)+Min
                data[ticker][variable]=denormalized

    def log_normalization(self):
        for ticker in tickers:
            for variable in self.vars:
                data[ticker][variable]=round(np.e**data[ticker][variable],2)