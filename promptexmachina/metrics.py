import numpy as np
import pandas as pd



class Accuracy:
    

    def __init__(self, results, baseline=0):

        self.results = results
        self.baseline = baseline

    def overall(self, score='flag', percent=False, ret=True):

        nq = len(self.results)
        acc = self.results[score].sum() / nq
        if percent:
            acc *= 100
        
        self.overall = self.baseline + acc
        
        if ret:
            return acc

    def topical(self, topic, score='flag', percent=False, ret=True):

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