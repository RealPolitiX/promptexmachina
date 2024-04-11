import numpy as np
import operator
from functools import reduce
from nltk.tokenize import sent_tokenize


def sent_tile(text, per=1):

    sentences = sent_tokenize(text)
    sent_reformatted = sentences[:1] + [' ' + s for s in sentences[1:]]
    # sent_reformatted = sent_reformatted[::2]
    nsent_stages = range(len(sent_reformatted))[::per]
    tiled_sentences = [reduce(operator.add, sent_reformatted[:n+1]) for n in nsent_stages]

    return tiled_sentences


class TextAggregator:

    def __init__(self, init_text=''):

        self.text = init_text

    def aggregate(self, text, loc='after', spacer='\n', end_spacer='\n'):
        """ Aggregate a single text piece.
        """

        if loc == 'before':
            self.text = text + spacer + self.text
        elif loc == 'after':
            self.text = self.text + spacer + text

        if end_spacer is not None:
            self.text += end_spacer

    def aggregate_n(self, textlist, spacer='\n', skip_last_spacer=True, shuffle=False):
        """ Aggregate n text pieces.
        """

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

    def disaggregate(self):

        sentences = sent_tokenize(self.text)

        return sentences
    
    def cumsum(self, per):

        text_cumsum = sent_tile(self.text, per)

        return text_cumsum

