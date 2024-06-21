from . import args
from . import pd, plt, os
from polygon import RESTClient
import datetime

client = RESTClient(api_key=args["ENV"][0]["polygon_token"])
cwd=os.getcwd()
data={}
tickers=args['ENV'][0]["tickers"]

def _downloader(ticker):
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
        timespan=args["ENV"][0]["timespan"],
        from_=args["ENV"][0]["from_"],
        to=args["ENV"][0]["to"],
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
            _downloader(x)
        print("Download complete.")
    else:
        print("No tickers found to scrape data for, passing download.")

    files=[x for x in os.listdir(DataDirectory) if not(any([x.endswith(".py"),x.endswith(".ini")]))]
    for file,ticker in zip(files,tickers):
        df = pd.read_json(file, orient ='split', compression = 'infer')
        data[ticker]=df

    os.chdir(cwd)