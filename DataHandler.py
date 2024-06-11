from . import pd, np, plt, os
from . import TOMLHandler as TOML
from polygon import RESTClient
import datetime
import json

client = RESTClient(api_key="JfOImVgvEriAx0nSXw6pOKMah8mTVhX4")
cwd=os.getcwd()
def downloader(ticker):
        open_=[]
        high=[]
        low=[]
        close=[]
        volume=[]
        vwap = []
        transactions = []
        timestamp = []

        for x in client.list_aggs(
            ticker=ticker,
            multiplier=1,
            timespan="minute",
            from_="2024-05-01",
            to="2024-06-01",
            limit=50000
        ):
            open_.append(x.open)
            high.append(x.high)
            low.append(x.low)
            close.append(x.close)
            volume.append(x.volume)
            vwap.append(x.vwap)
            transactions.append(x.transactions)
            date=datetime.datetime.fromtimestamp(x.timestamp/1000.)
            timestamp.append(date)

        df = pd.DataFrame({
            "timestamp":timestamp,
            "open":open_,
            "high":high,
            "low":low,
            "close":close,
            "volume":volume,
            "vwap":vwap,
            "transactions":transactions
        }).set_index("timestamp")
        df.to_json(f"{ticker}_data.json", orient = 'split', compression = 'infer', index = 'true')


class JSON():
    def __init__(self):
        self.args = TOML.load_args("Variables.toml")
        self.data={}
        self.tickers=self.args['ENV'][0]["tickers"]

        if not(os.path.isdir("Graphs")):
            os.mkdir("Graphs")
            print("Created Graphs directory.")
        
        if not(os.path.isdir("JSON_Data")):
            os.mkdir("JSON_Data")
            print("Created JSON data directory.")

        DataDirectory=cwd+"\\JSON_Data"
        os.chdir(DataDirectory)
        Cached=[x.split("_")[0] for x in os.listdir(DataDirectory)]
        if not(self.args['ENV'][0]["Omit_Cache"]):
            if len(Cached)>0:
                args=[x for x in self.tickers if x not in Cached]
                print(f"Found pre-existing data for the following tickers, skipping them:\n{Cached}.")
            else:
                print(f"No pre-existing data found for the following tickers:\n{self.tickers}.")
                args=self.tickers
        else:
            args=self.tickers
            print(f"Found pre-existing data for the following tickers, cache omitted, overwriting existing data for:\n{Cached}.")

        if len(args)>0:
            print("Downloading...")
            for x in args:
                downloader(x)
            print("Download complete.")
        else:
            print("No tickers found to scrape data for, passing download.")

        files=[x for x in os.listdir(DataDirectory) if not(any([x.endswith(".py"),x.endswith(".ini")]))]
        for file,ticker in zip(files,self.tickers):
            df = pd.read_json(file, orient ='split', compression = 'infer')
            self.data[ticker]=df

        if self.args['ENV'][0]["plot"]:    
            os.chdir(cwd+"\\Graphs")
            print("Graphing Open, High, Low, and Closing prices...")
            for ticker in self.tickers:
                currentplot=0
                XAxis = [x for x in range(len(self.data[ticker]["volume"]))]
                plt.tight_layout()
                _, axes = plt.subplots(7,figsize=(16,16))
                for variable in self.data[ticker]:
                    axes[currentplot].plot(
                        XAxis,
                        self.data[ticker][variable],
                        color='g'
                    )
                    axes[currentplot].title.set_text(f"{variable}_{ticker}")
                    currentplot+=1
                plt.savefig(f"{ticker}_data.png")
                plt.close()

        
        os.chdir(cwd)
        if str(self.args['ENV'][0]["normalization"])=="MinMax":
            for ticker in self.tickers:
                df=self.data[ticker]
                self.data[ticker]=(df-df.min())/(df.max()-df.min())

    def __split(self,df,ratio):

        mask_1=np.array([x for x in range(len(df))])<int(len(df)*ratio[0])
        mask_2=np.array([x for x in range(len(df))])<int(len(df)*ratio[1])
        mask_3=np.array([x for x in range(len(df))])<int(len(df)*ratio[2])

        return[
            df[mask_1],
            df[mask_1:mask_2],
            df[mask_2:mask_3]
        ]
    
    def train_val_test_split(self,ratio):
        if type(self.args['ML'][0]["ticker"])==list:
            if type(self.args['ML'][0]["variable"])==list:
                for x in self.args['ML'][0]["ticker"]:
                    for y in self.args['ML'][0]["variable"]:
                        df=self.data[x][y]
                        yield self.__split(df,ratio)
            else:
                for x in self.args['ML'][0]["ticker"]:
                    df=self.data[x][self.args['ML'][0]["variable"]]
                    yield self.__split(df,ratio)

        else:
            df=self.data[self.args['ML'][0]["ticker"]][self.args['ML'][0]["variable"]]
            return self.__split(df,ratio)

    
    def check_prices(self,strikeprices):
        
        headers = {"accept": "application/json",
            "APCA-API-KEY-ID": "PK633KOPQ2D108RTHDPE",
            "APCA-API-SECRET-KEY": "ByGftglsHBXvt3ZLMrDN1GY8eJjkpE8Z2zGfTjE8"
        }

        import requests

        for x,y in zip(self.tickers,strikeprices):
            url=f"https://data.alpaca.markets/v2/stocks/bars/latest?symbols={x}&feed=iex"
            resp=requests.get(url, headers=headers).text
            resp=json.loads(resp)["bars"][x]["c"]
            if resp<y:
                print(f"{x} is below the target price of ${y}, currently at {resp}")
            elif resp==y:
                print(f"{x} is currently at the exact target price of ${y}")
            else:
                print(f"{x} is not at the target price of ${y}, currently at {resp} with ${round(resp-y,2)} to go")
