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
    


class AnswerParser:

    def __init__(self, answers, nsent_name=None, class_name=None):

        self.answers = answers
        self.class_name = class_name
        if class_name is not None:
            self.topics = answers[class_name].unique().tolist()
        else:
            self.topics = []

        self.n_sents = answers[nsent_name].unique()
        self.maxsent = np.max(self.n_sents)

    @property
    def n_topics(self):
        return len(self.topics)

    @property
    def n_answers(self):
        return len(self.answers)        
            
    def measure_overall(self, metric, score='flag', n_sent=None,
                        calculate_division_size=True, division_size=None):
        ''' Calculate overall performance metric.
        '''

        if calculate_division_size:
            divsize = int(self.n_answers / self.maxsent)
        else:
            divsize = division_size
            
        if n_sent is None:
            nsts = list(range(self.maxsent))
            score_matrix = np.zeros((1, self.maxsent))
        # elif isinstance(n_sent, int):
        #     if n_sent > 0:
        #         nsts = np.zeros
        
        for n in nsts:
            scorer = metric(results=self.answers[divsize*n:divsize*(n+1)])
            overall_score = scorer.overall(score=score)
            score_matrix[0, n] = overall_score

        return score_matrix
        
    def measure_topical(self, metric, topic_name=None, topic_list=None, score='flag',
                        calculate_division_size=True, division_size=None):
        ''' Calculate topical performance metric.
        '''

        if calculate_division_size:
            divsize = int(self.n_answers / self.maxsent)
        else:
            divsize = division_size

        if topic_name is None:
            topic_name = self.class_name

        if topic_list is None:
            topic_list = self.topics
        n_selected_topics = len(topic_list)
        score_matrix = np.zeros((n_selected_topics, self.maxsent))
        
        for n in range(self.maxsent):
            scorer = metric(results=self.answers[divsize*n:divsize*(n+1)])
            topical_score = scorer.topical(score=score, topic=topic_name)
            score_matrix[:, n] = np.array([topical_score[top] for top in topic_list])

        return score_matrix