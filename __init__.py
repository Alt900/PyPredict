import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import toml
import tensorflow as tf

import matplotlib
matplotlib.use('Agg')

print("Initializing environment...")

if not(os.path.isfile("Variables.toml")):
    TomlString="""
[[ENV]]
Omit_Cache = bool
tickers=list
normalization=str
timespan=str
from_=str YYYY-MM-DD
to=str YYYY-MM-DD
polygon_token = str

[[ML]]
tickers=str/list
variables=str/list
batch_size = int
windowsize = int
learning_rate = float
CNNAuxillary = bool
loadmodel = bool
savemodel = bool 
epochs = int

[[Statistics]]
tickers=str/list
variables=str/list"""

args = toml.load("Variables.toml")

if not(os.path.isdir("Graphs")):
    os.mkdir("Graphs")
    print("Created Graphs directory.")

if not(os.path.isdir("JSON_Data")):
    os.mkdir("JSON_Data")
    print("Created JSON data directory.")

if args["ML"][0]["savemodel"] and not(os.path.isdir("Models")):
    os.mkdir("Models")

    with open("Variables.toml","w+") as F:
        F.write(TomlString)
    print(f"no TOML file was found, one was crated in the current directory of {os.getcwd()}")
    exit()