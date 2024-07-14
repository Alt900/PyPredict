from . import np, pd, args
from . import torch
import statsmodels
#export to ML

class LSTM_Prep():
    def __init__(self,
        time_series:pd.DataFrame,
        ratio=(.7,.3),
        time_shift=1,
        label_size=1
        ):
        
        self.time_shift = time_shift
        self.label_size = label_size

        self.training, self.testinging = self.__trailing_split(time_series,ratio)

        self.__Window_Data()

    def __trailing_split(self,df:pd.DataFrame,ratio:tuple):
        n=len(df)
        s_i=0
        for x in ratio:
            RtoI=int(n*x) #ratio to index location
            yield df[s_i:s_i+RtoI]
            s_i+=RtoI #adds the previously calculated index location as a starting point for the next
    
    def __Window_Data(self):
        TFTTV=[]
        TTV=[self.training,self.testinging]
        for x in range(2):
            size=len(TTV[x])
            indexes=[]
            labels=[]
            initial_rows=int((size-args["ML"][0]["windowsize"])/self.time_shift)
            indexes = [y for y in range(0,size,self.time_shift)]

            indexes=indexes[0:initial_rows]#filters out non-valid windows
            matrix=np.zeros((len(indexes)-1,args["ML"][0]["windowsize"]))

            for y in range(len(indexes)):
                try:
                    labels.append([TTV[x][indexes[y]+args["ML"][0]["windowsize"]+self.label_size]])#a label needs to be present in order to complete the set
                    matrix[y]=TTV[x][indexes[y]:indexes[y]+args["ML"][0]["windowsize"]]
                except IndexError:#if one cannot be grabbed due to the index being out of bounds the windowed set will be dropped
                    pass

            TFTTV.append([matrix[:,:,np.newaxis],np.array(labels)[:,np.newaxis]])

        self.TTV_x=[TFTTV[0][0].astype(np.float32),TFTTV[1][0].astype(np.float32)]
        self.TTV_y=[TFTTV[0][1].astype(np.float32),TFTTV[1][1].astype(np.float32)]

class Feature_Engineering():#everything works, howww? 0_o
    def __init__(self,
        time_series:pd.DataFrame,
        min:float,
        max:float
        ):
        self.torched_series=torch.Tensor(time_series)
        self.time_series=time_series
        self.min = min
        self.max = max

    def seasonal_decompose(self,period=5):
        return statsmodels.tsa.seasonal.seasonal_decompose(self.time_series,model="additive",period=period)

    def standard_clip(self):
        return torch.clamp(self.torched_series,self.min,self.max)
    
    def MAD(self,threshold):
        median = np.median(self.time_series)
        mad = np.median(abs(self.time_series-median))
        lower_cut = median-threshold*mad
        upper_cut = median+threshold*mad
        return torch.clamp(torch.Tensor(self.time_series),min=lower_cut,max=upper_cut)
    
    def difference(self):
        return self.time_series.diff().fillna(0)