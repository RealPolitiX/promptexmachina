import numpy as np
import pandas as pd
import qstab.data as qdata


class AnswerCollector:
    
    def __init__(self, rowvals=[], colnames=["answer"], **kwargs):
        self.answers = pd.DataFrame(rowvals, columns=colnames, **kwargs)
    
    @property
    def num_entry(self):
        return self.answers.shape[0]
    
    @property
    def num_collection(self):
        return len(self.answers.columns)
        
    @property
    def colnames(self):
        return self.answers.columns.values
    
    @classmethod
    def from_array(self, rowvals, **kwargs):
        return self(rowvals=rowvals, **kwargs)
    
    def add_entry(self, **kwargs):
        ''' Add a row to the answers dataframe (default values are nan).
        '''
        last_index = self.num_entry
        # self.answers.loc[last_index] = np.nan
        for k, v in kwargs.items():
            self.answers.at[last_index, k] = v
            
    def remove_entry(self, index):
        ''' Remove a row from the answer collection.
        '''
        self.answers = self.answers.drop(index, axis=0)
        
    def add_collection(self, collection, name=None):
        ''' Add a collection of values for a new column.
        '''
        self.answers[name] = collection
    
    def remove_collection(self, name):
        ''' Remove a column from the answer collection.
        '''
        self.answers = self.answers.drop(name, axis=1)
    
    def clean(self):
        pass
    
    def to_excel(self, filepath, sheet_name='Results', **kwargs):
        self.answers.to_excel(filepath, sheet_name=sheet_name, **kwargs)
        
    def to_csv(self, filepath, **kwargs):
        self.answers.to_csv(filepath, **kwargs)