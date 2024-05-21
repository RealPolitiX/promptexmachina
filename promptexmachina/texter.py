import numpy as np
import operator
from functools import reduce
import random, string
from nltk.tokenize import sent_tokenize


def sent_tile(text, per=1, first=None):

    sentences = sent_tokenize(text)
    sent_reformatted = sentences[:1] + [' ' + s for s in sentences[1:]]
    # sent_reformatted = sent_reformatted[::2]
    nsent_stages = range(len(sent_reformatted))[::per]
    tiled_sentences = [reduce(operator.add, sent_reformatted[:n+1]) for n in nsent_stages]
    if first is None:
        return tiled_sentences
    else:
        return [first] + tiled_sentences


class TextAggregator:
    ''' Text operations (aggregation, disaggregation) at the sentence level.
    '''

    def __init__(self, init_text=''):

        self.text = init_text

    @property
    def n_sent(self):

        if self.text:
            sents = self.disaggregate()
            return len(sents)
        else:
            return 0

    def aggregate(self, text, loc='after', spacer='\n', end_spacer='\n'):
        ''' Attach a single text piece to the existing one.
        '''

        if loc == 'before':
            self.text = text + spacer + self.text
        elif loc == 'after':
            self.text = self.text + spacer + text

        if end_spacer is not None:
            self.text += end_spacer

    def aggregate_n(self, textlist, spacer='\n', skip_last_spacer=True, shuffle=False):
        ''' Aggregate n text pieces.
        '''

        if shuffle:
            textlist_ag = np.random.choice(textlist, size=len(textlist), replace=False).tolist()
        else:
            textlist_ag = textlist
        
        if spacer is not None:
            # Add spacer between sentences (skip the last sentence or not)
            if skip_last_spacer:
                modtextlist = [t+spacer for t in textlist_ag[:-1]]
                modtextlist += textlist_ag[-1]
            else:
                modtextlist = [t+spacer for t in textlist_ag]
            self.text += reduce(operator.add, modtextlist)
            
        else:
            self.text += reduce(operator.add, textlist_ag)

    def disaggregate(self):
        ''' Decompose a paragraph into sentences.
        '''

        sentences = sent_tokenize(self.text)

        return sentences
    
    def keep_n(self, n, inplace=False):
        ''' Keep the first n sentences.
        '''

        n = int(n)
        sent_cumsum = self.cumsum(per=1, first=None)

        if n < 1:
            raise ValueError('n should be at least 1.')
        elif n > self.n_sent:
            n = self.n_sent

        if inplace:
            self.text = sent_cumsum[n-1]

        return sent_cumsum[n-1]
    
    def cumsum(self, per, first=None):
        ''' Cumulative summation for text at the sentence level.
        '''

        text_cumsum = sent_tile(self.text, per, first=first)

        return text_cumsum


def randstr(n):
    res = ''.join(random.choices(string.ascii_letters +
                                 string.digits, k=n))
    return res

def rand_replace(sentence, extra=None):
    words = sentence.split(' ')
    nw = len(words)
    
    replaced_text = ''
    for iw, word in enumerate(words):
        nletters = len(list(word))
        word_rep = randstr(nletters)
        if iw < nw-1:
            replaced_text += word_rep + ' '
        else:
            replaced_text += word_rep + '.'

    if extra is None:
        return replaced_text
    else:
        return replaced_text + extra