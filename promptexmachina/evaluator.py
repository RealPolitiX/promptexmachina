from . import metrics as m
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


class Evaluator:

    def __init__(self, dataloader, colnames=['mod_ans', 'flag', 'persona',
                                             'true_ans', 'true_ans_text']):

        self.data = dataloader.data
        self.datalen = dataloader.len
        self.outcome = AnswerCollector(colnames=colnames)

    @classmethod
    def prompt_format(cls, entry, addon):

        entry["reference"] = None
        # entry["prompt_prefix"] = "[Instruction]: Assume you {} Answer the question without explanation.\n[Context]: ".format(psn)
        entry["prompt_prefix"] = "[Instruction]: {}Answer the question without explanation.\n[Context]: ".format(addon)
        entry["prompt_suffix"] = "[Answer]: "
        qobj = qdata.question.Question.from_dict(entry)
        qsent = qobj.question.split('. ')[-1]
        qobj.question = qobj.question[:-len(qsent)]
        qobj.question = qobj.question + qsent
        
        return qobj
    
    def enable_accelerator(self, name='deepspeed'):
        
        if name == 'deepspeed':
            from mii import pipeline
        else:
            raise NotImplemented
    
    def evaluate(self, i, addon, model, tokenizer, model_kwargs=None):

        if model_kwargs is None:
            model_kwargs = {"max_new_tokens":128, "do_sample":False,
                            "pad_token_id":tokenizer.eos_token_id}
        entry_dict = self.data.iloc[i,:].to_dict()
        qobj = self.prompt_format(entry_dict, addon)
        answ = qobj.query(model, tokenizer, model_kwargs=model_kwargs)
        answ = m.split_answer(answ)
        flag = m.soft_validate(answ, qobj.answer_idx)

        return entry_dict, answ, flag
    
    def evalpipe(self, i, addon, model, tokenizer, pipeline, model_kwargs=None):

        if model_kwargs is None:
            model_kwargs = {"max_new_tokens":128, "do_sample":False, "pad_token_id":tokenizer.eos_token_id}
        entry_dict = self.data.iloc[i,:].to_dict()
        qobj = self.prompt_format(entry_dict, addon)
        # print(qobj.full_question)
        answ = pipeline(qobj.full_question, return_full_text=True, **model_kwargs)
        answ = m.split_answer(answ[0].generated_text)
        flag = m.soft_validate(answ, qobj.answer_idx)
        
        return entry_dict, answ, flag
    

class StudyAnalyzer:
    ''' Analyzing optuna study instances.
    '''
    
    def __init__(self, study, multiobj=False):

        self.trials_df = study.trials_dataframe()
        self.trials = study.trials
        self.n_trials = len(self.trials)
        if multiobj:
            self.best_params = [t.params for t in study.best_trials]
            self.best_value = [t.values for t in study.best_trials]
        else:
            self.best_params = study.best_params
            self.best_value = study.best_value

    def trial_values(self):
        
        return self.trials_df['value'].values
        
    def cummax(self, colname='value'):

        return self.trials_df[colname].cummax().values

    def cummin(self, colname='value'):

        return self.trials_df[colname].cummin().values
    
    def cummean(self, colname='value'):
        
        objective_values = self.trials_df[colname].cumsum().values
        indices = np.array(range(1, self.n_trials+1))
        
        return objective_values / indices