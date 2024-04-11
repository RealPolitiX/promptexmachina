import numpy as np
import pandas as pd



class Accuracy:
    """
    Accuracy calculator.
    """

    def __init__(self, results, baseline=0):

        self.results = results
        self.baseline = baseline

    def overall(self, score='flag', percent=False, ret=True):
        """ Calculate overall accuracy.
        """

        nq = len(self.results)
        acc = self.results[score].sum() / nq
        if percent:
            acc *= 100
        
        self.overall = self.baseline + acc
        
        if ret:
            return acc

    def topical(self, topic, score='flag', percent=False, ret=True):
        """ Calculate topic-resolved accuracy.
        """

        self.topics = self.results[topic].unique()
        acc = {}
        
        for top in self.topics:
            partial_results = self.results[self.results[topic] == top]
            nq_partial = len(partial_results)
            acc_partial = partial_results[score].sum() / nq_partial
            if percent:
                acc_partial *= 100
            acc[top] = acc_partial

        self.topical = acc

        if ret:
            return acc        


def split_answer(text):
    
    try:
        text = text.replace('\n', '')
        text = text.split('[Answer]: ')[1]
    except:
        pass
    if text.startswith(' '):
        text = text[1:]
        
    return text


def soft_validate(text, gt, start=0, bound=4):

    if gt in text[start:bound]:
        return True
    else:
        return False