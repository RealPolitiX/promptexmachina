import numpy as np


def parse_arc(item):
    
    outdict = {}
    outdict['question'] = item['question']
    choices = item['choices']
    outdict['options'] = dict(zip(choices['label'], choices['text']))
    
    outdict['answer_idx'] = item['answerKey']
    outdict['answer'] = outdict['options'][item['answerKey']]
    
    return outdict


def parse_mmlu(item):
    # Parser for the MMLU dataset

    outdict = {}
    lettlist = list('ABCDEFGHIJKLMN')
    nlett = len(lettlist)
    outdict['question'] = item['question']
    outdict['concept'] = item['subject']
    outdict['label'] = item['answer']
    nopts = len(item['choices'])
    
    if nlett >= nopts:
        outdict['options'] = dict(zip(lettlist[:nopts], item['choices']))
        outdict['answer_idx'] = lettlist[item['answer']]
    else:
        print('Too many options!')
    outdict['answer'] = item['choices'][item['answer']]

    return outdict


def parse_sciq(item):
    # Parser for the SciQ dataset

    outdict = {}
    lettlist = list('ABCD')
    nlett = len(lettlist)
    outdict['question'] = item['question']
    outdict['support'] = item['support']
    np.random.shuffle(lettlist)
    outdict['label'] = lettlist[0]
    distractors = [val for key, val in item.items() if key.startswith('distractor')]
    answ = item['correct_answer']
    outdict['answer'] = answ

    # Get options sorted by the letter index
    choices = distractors + [answ]
    sort_order = np.argsort(lettlist)
    outdict['options'] = dict([(lettlist[i], choices[i]) for i in sort_order])
    
    outdict['answer_idx'] = lettlist[-1]
    
    return outdict


def sort_by_indexes(lst, indexes, reverse=False):
    return [val for (_, val) in sorted(zip(indexes, lst), key=lambda x: \
          x[0], reverse=reverse)]


def parse_truthful(item, reorder=True):
    # Parser for the TruthfulQA dataset

    outdict = {}
    lettlist = list('ABCDEFGHIJKLMN')
    nlett = len(lettlist)
    outdict['question'] = item['question']
    options = item['mc1_targets']['choices']
    labels = item['mc1_targets']['labels']
    nopts = len(options)
    
    if reorder:
        range_vec = list(range(nopts))
        np.random.shuffle(range_vec)
        options = sort_by_indexes(options, range_vec)
        labels = sort_by_indexes(labels, range_vec)
        
    if nlett >= nopts:
        outdict['options'] = dict(zip(lettlist[:nopts], options))
        if sum(labels) == 1:
            outdict['label'] = labels.index(1)
            outdict['answer_idx'] = lettlist[outdict['label']]
            outdict['answer'] = options[outdict['label']]
        else:
            print('More than one correct answer!')
    else:
        print('Too many options!')
    
    return outdict


def parse_bbq(item):
    # Parser for the BBQ dataset
    
    outdict = {}
    lettlist = list('ABCDEFGHIJKLMN')
    nlett = len(lettlist)
    outdict['context'] = item['context']
    outdict['question'] = item['question']
    outdict['concept'] = item['category']
    item_keys = list(item.keys())
    answ_keys = [ik for ik in item_keys if ik.startswith('ans') and '_' not in ik]
    ans_vals = [item[ak] for ak in answ_keys]
    nopts = len(answ_keys)
    
    if nlett >= nopts:
        outdict['options'] = dict(zip(lettlist[:nopts], ans_vals))
        outdict['answer_idx'] = lettlist[item['label']]
    else:
        print('Too many options!')
        
    outdict['answer'] = ans_vals[item['label']]
    
    return outdict
