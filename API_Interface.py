import alpaca.data
from . import args
from . import pd, os
from alpaca.data import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.live import StockDataStream,CryptoDataStream
from datetime import datetime
import alpaca

client = StockHistoricalDataClient(args["ENV"][0]["alpaca_key"],args["ENV"][0]["alpaca_secret"])
cwd=os.getcwd()
data={}
tickers=args['ENV'][0]["tickers"]

try:
    timeframe=eval(f"alpaca.data.timeframe.Timeframe.{args['ENV'][0]['time_span']}")
except Exception as E:
    print("Could not resolve TOML timeframe, defaulting to minute scale...")
    timeframe=alpaca.data.timeframe.TimeFrame.Minute

def _downloader(ticker):
    try:
        from_=datetime(*args["ENV"][0]["from_"])
        to=datetime(*args["ENV"][0]["to"])
        df=client.get_stock_bars(
            StockBarsRequest(
                symbol_or_symbols=ticker,
                timeframe=timeframe,#'Day', 'Hour', 'Minute', 'Month', 'Week'
                start=from_,
                end=to
            )
        ).df.reset_index(level=[0])
        df=df.drop(columns=["symbol"])
        df.to_json(f"{ticker}_data.json", orient = 'split', compression = 'infer', index = 'true')
        return df
    
    except AttributeError:
        print(f"Could not download data for {ticker}, skipping")

def download():
    DataDirectory=cwd+"\\JSON_Data"
    os.chdir(DataDirectory)

    #check for cache arg and cached files
    Cached=[x.split("_")[0] for x in os.listdir(DataDirectory)]
    if not(args['ENV'][0]["Omit_Cache"]):
        if len(Cached)>0:
            todownload=[x for x in tickers if x not in Cached]
            print(f"Found pre-existing data for the following tickers, skipping them:\n{Cached}.")
        else:
            print(f"No pre-existing data found for the following tickers:\n{tickers}.")
            todownload=tickers
    else:
        todownload=tickers
        print(f"Found pre-existing data for the following tickers, cache omitted, overwriting existing data for:\n{Cached}.")


    if len(todownload)>0:
        print("Downloading...")
        for x in todownload:
            data[x]=_downloader(x)
        print("Download complete.")
    else:
        print("No tickers found to scrape data for, passing download.")

    files=[x for x in os.listdir(DataDirectory) if not(any([x.endswith(".py"),x.endswith(".ini")]))]
    on_hand=[x for x in data]
    for file,ticker in zip(files,tickers):
        if ticker in on_hand:
            continue
        df = pd.read_json(file, orient ='split', compression = 'infer')
        data[ticker]=df

    os.chdir(cwd)
class LiveStream():
    def __init__(self):
        self.stock_client=StockDataStream(args["ENV"][0]["alpaca_key"],args["ENV"][0]["alpaca_secret"])
        self.crypto_client=CryptoDataStream(args["ENV"][0]["alpaca_key"],args["ENV"][0]["alpaca_secret"])

    async def stream_handler(self,data):
        print(data)
    
    def open_stream(self,ticker):
        self.stock_client.subscribe_quotes(self.stream_handler,ticker)
        self.stock_client.run()

    def open_crypto_stream(self,ticker):
        self.crypto_client.subscribe_quotes(self.stream_handler,ticker)
        self.crypto_client.run()