import pandas as pd



class TableLoader:

    def __init__(self, fpath, loader, **kwargs):

        self.fpath = fpath
        self.data = loader(self.fpath, **kwargs)

    def __repr__(self):

        return "A TableLoader instance with file path at {}".format(self.fpath)

    def cfilter(self, col, target):
        
        if isinstance(target, list):
            self.data = self.data[self.data[col].isin(target)]
        else:
            self.data = self.data[self.data[col] == target]

    def rfilter(self, row):

        pass

    @property
    def len(self):

        return len(self.data)