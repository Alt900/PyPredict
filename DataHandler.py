from . import tf, args
from .API_Interface import data

class ML():
    def __split(self,df,ratio):
            n=len(df)
            r1=int(n*ratio[0])
            r2=int(n*ratio[1])+r1
            r3=int(n*ratio[2])+r2
            return[
                tf.data.Dataset.from_tensor_slices(list(df[:r1])),
                tf.data.Dataset.from_tensor_slices(list(df[r1:r2])),
                tf.data.Dataset.from_tensor_slices(list(df[r2:r3]))
            ]
        
    def train_val_test_split(self,ratio,ticker,variable):
        df=data[ticker][variable]
        return self.__split(df,ratio)
        
    def Window_Data(self,TTV,shift=1):#TTV needs to be a tf.data.dataset.from_tensor_slice array
        T1Y, T1X = [],[]
        T2Y, T2X = [],[]
        VY, VX = [],[]
        
        T1_set = TTV[0].window(args["ML"][0]["windowsize"],shift=shift,drop_remainder=True)
        T2_set = TTV[1].window(args["ML"][0]["windowsize"],shift=shift,drop_remainder=True)
        V_set = TTV[2].window(args["ML"][0]["windowsize"],shift=shift,drop_remainder=True)

        T1_set = T1_set.flat_map(lambda window: window.batch(args["ML"][0]["windowsize"]))
        T2_set = T2_set.flat_map(lambda window: window.batch(args["ML"][0]["windowsize"]))
        V_set = V_set.flat_map(lambda window: window.batch(args["ML"][0]["windowsize"]))

        T1_set = T1_set.map(lambda window: (window[:-1],window[-1:]))
        T2_set = T2_set.map(lambda window: (window[:-1],window[-1:]))
        V_set = V_set.map(lambda window: (window[:-1],window[-1:]))

        for x,y in T1_set:
            T1X.append(x.numpy()), T1Y.append(y.numpy()) 

        for x,y in T2_set:
            T2X.append(x.numpy()), T2Y.append(y.numpy()) 

        for x,y in V_set:
            VX.append(x.numpy()), VY.append(y.numpy()) 

        return[
            T1X, T1Y,
            T2X, T2Y,
            VX, VY
        ]