import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import toml
import torch

import matplotlib

print("Initializing environment...")

if not(os.path.isfile("Variables.toml")):
    TomlString="""
[[ENV]]
Omit_Cache = bool
tickers=str || list
timespan=str
from_=list, format [YYYY,MM,DD]
to=list, format [YYYY,MM,DD]
alpaca_key = str
alpaca_secret = str
[[ML]]
batch_size = int
windowsize = int
learning_rate = float
load_model = bool
save_model = bool 
epochs = int
plot_loss=bool"""

args = toml.load("Variables.toml")

if not(os.path.isdir("Graphs")):
    os.mkdir("Graphs")
    print("Created Graphs directory.")

if not(os.path.isdir("JSON_Data")):
    os.mkdir("JSON_Data")
    print("Created JSON data directory.")

if args["ML"][0]["save_model"] and not(os.path.isdir("Models")):
    os.mkdir("Models")

    with open("Variables.toml","w+") as F:
        F.write(TomlString)
    print(f"no TOML file was found, one was crated in the current directory of {os.getcwd()}")
    exit()