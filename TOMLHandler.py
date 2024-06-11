import toml

def load_args(filename):
    return toml.load(filename)

def update_args(filename,argstruct):
    toml.dump(argstruct,filename)

