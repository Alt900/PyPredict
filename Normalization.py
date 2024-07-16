from . import np
global data
from .API_Interface import data

__DeNormalMask={}


class Normalize():
    def __init__(self,ticker,variable):
        self.ticker = ticker
        self.variable = variable

    def MinMax(self):
        df=data[self.ticker][self.variable]
        __DeNormalMask[self.ticker][self.variable]=[df.min(),df.max()]
        data[self.ticker][self.variable]=(df-__DeNormalMask[self.ticker][self.variable][0])/(__DeNormalMask[self.ticker][self.variable][1]-__DeNormalMask[self.ticker][self.variable][0])

    def Logarithmic(self):
        data[self.ticker][self.variable]=np.log(data[self.ticker][self.variable])

    def Z_score(self):
        std=data[self.ticker][self.variable].std()
        mean=data[self.ticker][self.variable].mean()
        __DeNormalMask[self.ticker][self.variable]["std"]=std
        __DeNormalMask[self.ticker][self.variable]["mean"]=mean
        data[self.ticker][self.variable]=(data[self.ticker][self.variable]-mean)/std
        
class DeNormalize():
    def __init__(self,ticker,variable):
        self.ticker = ticker
        self.variable = variable

        self.External_Sets={
            "MinMax":lambda external_set, exmin, exmax: [(x*(exmax-exmin)+exmin) for x in external_set],
            "Logarithmic":lambda external_set:[round(np.e**x,2) for x in external_set],
            "z_score":lambda external_set, exstd, exmean: [(x*exstd+exmean) for x in external_set]
        }

    def MinMax(self):
        Min,Max=__DeNormalMask[self.ticker][self.variable]
        denormalized=np.zeros((data[self.ticker][self.variable].shape[0],))
        for i in range(data[self.ticker][self.variable].shape[0]):
            denormalized[i]=data[self.ticker][self.variable][i]*(Max-Min)+Min
        data[self.ticker][self.variable]=denormalized

    def Logarithmic(self):
        data[self.ticker][self.variable]=round(np.e**data[self.ticker][self.variable],2)

    def Z_score(self):
        data[self.ticker][self.variable]=(data[self.ticker][self.variable]*__DeNormalMask[self.ticker][self.variable]["std"])+__DeNormalMask[self.ticker][self.variable]["mean"]