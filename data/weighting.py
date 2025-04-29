import pandas as pd

from   sklearn.feature_extraction.text import TfidfVectorizer
from   collections import defaultdict

class WeightingTool:
    def __init__(self,
                 df: pd.DataFrame,
                 autoBalance: bool=True):
        # initialize some dictionaries to remember
        # weighted and unweighted distributions of words
        self.words_sarcastic     = defaultdict(int)
        self.words_not_sarcastic = defaultdict(int)
        self.weighted_sarcastic  = defaultdict(float)
        self.lens_sarcastic      = defaultdict(int)
        self.lens_not_sarcastic  = defaultdict(int)

        self.df = df

        # if weight and length are not keys in the dataframe, add them
        if not 'length' in self.df.columns:
            self.df['length'] = [len(row['headline'].split()) for _, row in self.df.iterrows()]
        if not 'weight' in self.df.columns:
            self.df['weight'] = [1.0 for _, row in self.df.iterrows()]

        if autoBalance:
            self.applyLengthBalance()
            self.applySemanticBalance()
        
    '''
    Re-compute dictionaries
    Is called whenever weights are updated
    '''
    def fill_dictionaries(self) -> None:
        self.words_sarcastic    .clear()
        self.words_not_sarcastic.clear()
        self.weighted_sarcastic .clear()
        self.lens_sarcastic     .clear()
        self.lens_not_sarcastic .clear()
        
        for index, row in self.df.iterrows():
            if row['is_sarcastic']:
                self.lens_sarcastic[len(row['headline'].split())] += 1
                for word in row['headline'].split():
                    self.words_sarcastic[word] += 1
                    if 'weight' in row.keys():
                        self.weighted_sarcastic[word] += row['weight']
            else:
                self.lens_not_sarcastic[len(row['headline'].split())] += 1
                for word in row['headline'].split():
                    self.words_not_sarcastic[word] += 1

    '''
    Define an element-by-element re-weighting
    to ensure equally weighted sarcastic and non-sarcastic
    by sentence length
    '''
    def lengthWeight(self, row, threshold=100):
        if not row['is_sarcastic']: return 1.0 # only weight the sarcastic relative to the non-sarcastic

        # re-weight by length
        return self.lens_not_sarcastic[row['length']]/self.lens_sarcastic[row['length']]


    '''
    Apply a reweighting to the headlines based on length of headline
    '''
    def applyLengthBalance(self) -> None:
        self.fill_dictionaries()
        self.df['temp'] = [self.lengthWeight(row, 100) for _, row in self.df.iterrows()] # apply a weight
        self.df['weight'] = self.df['weight'] * self.df['temp']
        self.df.drop('temp', axis=1)
        
        # finally, re-update the dictionaries of weighted norms
        self.fill_dictionaries()
        

    '''
    Apply a reweighting to headlines based on whether certain features appear

    Args:
        df:          dataframe on which to balance
        features:    list of features for which balance is applied
        ngram_range: tuple with 2 elements: min/max number of words per feature

    Returns:
        None
    '''
    def applySemanticBalance(self,
                             features: list[str] =['gop', 'donald', 'trump', 'man', 'area', 'report', 'says', 'women', 'nation'],
                             ngram_range: tuple = (1, 1)) -> None:

        # Use TF-IDF to extract features
        tfidf_vectorizer = TfidfVectorizer(
            max_features=5000,  # Limit to top 5000 words by frequency
            min_df=5,           # Ignore terms that appear in less than 5 documents
            max_df=0.7,         # Ignore terms that appear in more than 70% of documents
            ngram_range=ngram_range
        )

        # Fit the vectorizer on all headlines to learn vocabulary
        tfidf_vectorizer.fit(self.df['headline'])

        # Get feature names (words)
        feature_names = tfidf_vectorizer.get_feature_names_out()

        # get current normalizations
        nSarc    = self.df[self.df['is_sarcastic']==1]['weight'].sum()
        nNotSarc = self.df[self.df['is_sarcastic']==0]['weight'].sum()
        for feature in features:
            cond_in = (self.df['is_sarcastic']==1) & (self.df['headline'].str.contains(feature))
            cond_out= (self.df['is_sarcastic']==1) & (~ self.df['headline'].str.contains(feature))
            pFeatSarc    = self.df[cond_in]['weight'].sum()/nSarc
            pNotFeatSarc = 1-pFeatSarc
            pFeatNotSarc = self.df[(self.df['is_sarcastic']==0) & (self.df['headline'].str.contains(feature))]['weight'].sum()/nNotSarc
            pNotFeatNotSarc = 1-pFeatNotSarc
            self.df.loc[cond_in,  'weight'] = self.df.loc[cond_in, 'weight']*pFeatNotSarc/pFeatSarc
            self.df.loc[cond_out, 'weight'] = self.df.loc[cond_out, 'weight']*pNotFeatNotSarc/pNotFeatSarc

        # recompute dictionary values
        self.fill_dictionaries()


    '''
    Produce a table of frequencies of individual feature occurance

    Args:
        weighted: use weights in computation if True
        is_sarcastic: if True use sarcastic headlines to determine the ordering
        features: return the table over these specific features
                  useful in comparing tables between before and after weighting, when order changes
    Returns:
        two frequency tables in list[list] format e.g. [[100, "cheese"]]
        the first returned will be the sarcastic, second non-sarcastic
    '''
    def getFrequencyTables(self,
                           weighted: bool=False,
                           is_sarcastic: bool=False,
                           features: list[str] | None=None,
                           ) -> list[list]:

        # determine which dict is "main": controlling the order
        # if features is specified, this will overrid the main
        if is_sarcastic and weighted:
            dict_main   = self.weighted_sarcastic
            dict_second = self.words_not_sarcastic
        elif is_sarcastic:
            dict_main   = self.words_sarcastic
            dict_second = self.words_not_sarcastic
        elif weighted:
            dict_main   = self.words_not_sarcastic
            dict_second = self.weighted_sarcastic
        else:
            dict_main   = self.words_not_sarcastic
            dict_second = self.words_sarcastic
            
        # compute the frequencies, either ranked by main or by explicit feature list
        if features is None:
            word_freq_main = [[v, k] for k, v in dict_main.items()]
            word_freq_main.sort(reverse=True)
        else:
            word_freq_main = [[dict_main[f], f] for f in features]
                
        word_freq_second = [[dict_second[k], k] for _, k in word_freq_main]

        # swap and return
        if is_sarcastic:
            freq_sarc, freq_not_sarc = word_freq_main, word_freq_second
        else:
            freq_sarc, freq_not_sarc = word_freq_second, word_freq_main

        return freq_sarc, freq_not_sarc
