import json
import torch
import pandas as pd
import string
import nltk

from   transformers     import PreTrainedTokenizerBase
from   torch.utils.data import Dataset

'''
Custom pytorch Dataset for loading the sentiment data
'''
class SentimentDataset(Dataset):
    filename = 'data/Sarcasm_Headlines_Dataset.json'
    def __init__(self,
                 dataframe: pd.DataFrame            | None = None,
                 tokenizer: PreTrainedTokenizerBase | None = None,
                 max_length=128):
        if dataframe is None:
            self.df     = self.__class__.getSarcasmDataset()
        else:
            self.df     = dataframe
        self.texts      = self.df['headline'].values
        self.labels     = self.df['is_sarcastic'].values
        self.weights    = self.df['weight'].values
        self.tokenizer  = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    '''
    Tokenize the input and return as a pytorch tensor
    '''
    def __getitem__(self,
                    idx: int) -> dict:

        if self.tokenizer is None:
            raise AttributeError('Tokenizer must be initialized to get items from dataset')
        
        text  = self.texts[idx]
        label = self.labels[idx]
        wgt   = self.weights[idx]
        
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            return_token_type_ids=True,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'token_type_ids': encoding['token_type_ids'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long),
            'weights': torch.tensor(wgt, dtype=torch.float)
        }

    ####
    # Implement some static functions unrelated to the pytorch
    # dataset concept.  This is for convenience in loading data
    ####

    '''
    Use nltk to remove stopwords
    '''
    def remove_stopwords(sentence: str,
                         stopwords: set[str]) -> str:
        final_text = []
        for i in sentence.split():
            if i.strip().lower() not in stopwords:
                final_text.append(i.strip())
        return " ".join(final_text)

    '''
    Load the sarcasm dataset and apply some cuts to it
    '''
    @classmethod
    def getSarcasmDataset(cls,
                          removeStopwords:      bool = True,
                          removeShortHeadlines: int = 3) -> pd.DataFrame:

        # get the stopwords from both nltk and string (for punctuation)
        nltk.download('stopwords')
        stopwords   = set(nltk.corpus.stopwords.words('english'))
        punctuation = list(string.punctuation)
        stopwords.update(punctuation)
        stopwords.update(['\'s'])

        # load the json into pandas dataframe
        df = pd.read_json(cls.filename, lines=True)
        df = df[df['headline'] != '']
        if 'article_link' in df.columns: df.drop('article_link', inplace=True, axis=1)
        if removeStopwords:
            df['headline']=df['headline'].apply(SentimentDataset.remove_stopwords, args=(stopwords,))
        if removeShortHeadlines >= 0:
            df = df[[len(row['headline'].split()) >= removeShortHeadlines for _, row in df.iterrows()]]

        # if weight and length are not keys in the dataframe, add them
        if not 'length' in df.columns:
            df['length'] = [len(row['headline'].split()) for _, row in df.iterrows()]
        if not 'weight' in df.columns:
            df['weight'] = [1.0 for _, row in df.iterrows()]
        
        return df

'''
Inherit from the sentiment dataset but provide a new dataset of
trump-centric sarcasm from clickhole
'''
class SarcasticTrumpDataset(SentimentDataset):
    filename = 'data/Clickhole_trump.json'
    
