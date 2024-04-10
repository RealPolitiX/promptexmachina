import numpy as np
import operator
from functools import reduce


class TextAggregator:

    def __init__(self, init_text=''):

        self.text = init_text

    def aggregate(self, text, spacer='\n'):

        self.text += text
        if spacer is not None:
            self.text += spacer

    def aggregate_n(self, textlist, spacer='\n', skip_last_spacer=True, shuffle=False):

        if shuffle:
            textlist_ag = np.random.choice(textlist, size=len(textlist), replace=False).tolist()
        else:
            textlist_ag = textlist
        
        if spacer is not None:
            if skip_last_spacer:
                modtextlist = [t+spacer for t in textlist_ag[:-1]]
                modtextlist += textlist_ag[-1]
            else:
                modtextlist = [t+spacer for t in textlist_ag]
            self.text += reduce(operator.add, modtextlist)
            
        else:
            self.text += reduce(operator.add, textlist_ag)    