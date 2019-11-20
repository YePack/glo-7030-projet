import pickle


def readfile(name):
    with open('{}.pkl'.format(name), 'rb') as fich:
        seq_bin = fich.read()
        return pickle.loads(seq_bin)


def savefile(classe, name):
    with open('{}.pkl'.format(name), 'wb') as fich:
        fich.write(pickle.dumps(classe, pickle.HIGHEST_PROTOCOL))
