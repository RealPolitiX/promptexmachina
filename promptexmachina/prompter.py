from . import texter
import numpy as np
import operator
from functools import reduce
from nltk.tokenize import sent_tokenize



openai_models = ['gpt-3.5-turbo-instruct',
                 'gpt-3.5-turbo-0125',
                 'gpt-3.5-turbo-1106',
                 'gpt-3.5-turbo']


def prompt_openai(client, prompt, model='gpt-3.5-turbo-instruct', temp=0, maxtok=1024):

    # Call the OpenAI API to generate a response
    response = client.completions.create(
        model=model,  # GPT-3.5-turbo engine, text-davinci-003
        prompt=prompt,
        temperature=temp,
        max_tokens=maxtok,  # You can adjust this to limit the response length
        n=1,
        stop=None,
    )

    # Extract and print the model's reply
    reply = response.choices[0].text.strip()
    
    return reply


class Prompter:

    def __init__(self, prompt):

        self.prompt = prompt

    def query(self, model, **kwargs):

        if model in openai_models:
            from openai import OpenAI

            api_key = kwargs.pop('api_key', None)
            client = OpenAI(
                  api_key=api_key,
                )
            output = prompt_openai(client=client, model=model, prompt=self.prompt, **kwargs)
        else:
            raise NotImplementedError

        return output
    

class PersonaGenerator(Prompter):

    def __init__(self, prompt, key_info):

        super().__init__(self, prompt)
        self.prompt = "Write a biography of up to {} sentences for a person who may know all the answers to the following questions. Write in the second-person voice and do not use personal names. Start with 'Assume you are ...'\n".format(key_info)

    def instance_fetcher(self, dataloader, n=10):
        """ Retrieve the instances from a dataset.
        """
        
        data = dataloader.data
        dlen = dataloader.len
        if n < dlen:
            inds = np.random.randint(0, dlen, size=n)
        else:
            inds = np.array(list(range(0, dlen)))

        textlist = data.iloc[inds]['question'].values
        text_aggr = texter.TextAggregator().aggregate_n(textlist).text

        return text_aggr

    def extend_prompt(self, loc, text):

        if loc == 'prepend':
            self.prompt = text + self.prompt
        elif loc == 'append':
            self.prompt += text


def sent_tile(text):

    sentences = sent_tokenize(text)
    sent_reformatted = sentences[:1] + [' ' + s for s in sentences[1:]]
    # sent_reformatted = sent_reformatted[::2]
    nsent_stages = range(len(sent_reformatted))[::2]
    tiled_sentences = [reduce(operator.add, sent_reformatted[:n+1]) for n in nsent_stages]

    return tiled_sentences

