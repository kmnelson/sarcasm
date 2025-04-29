import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import string

from plotting.mpl_config import setup_minor_ticks
from plotting.plots      import horizontal_bar
from data.datasets       import SentimentDataset
from collections         import defaultdict

words_sarcastic = defaultdict(int)
words_not_sarcastic = defaultdict(int)
weighted_sarcastic = defaultdict(float)
lens_sarcastic = defaultdict(int)
lens_not_sarcastic = defaultdict(int)

def fill_dictionaries(df):
    for index, row in df.iterrows():
        if row['is_sarcastic']:
            lens_sarcastic[len(row['headline'].split())] += 1
            for word in row['headline'].split():
                words_sarcastic[word] += 1
        else:
            lens_not_sarcastic[len(row['headline'].split())] += 1
            for word in row['headline'].split():
                words_not_sarcastic[word] += 1

def fill_weighted_dict(d, df, weight='weight'):
    # assumes df only has sarcastic words
    for index, row in df.iterrows():
        for word in row['headline'].split():
            d[word] += row[weight]
                
def remove_stopwords(sentence):
    final_text = []
    for i in sentence.split():
        if i.strip().lower() not in stopwords:
            final_text.append(i.strip())
    return " ".join(final_text)

'''
Define an element-by-element re-weighting
'''
def weight(row, global_weight_sarcastic=1.0, threshold=100):
    if not row['is_sarcastic']: return 1.0 # only weight the sarcastic relative to the non-sarcastic

    # global weight
    weight = global_weight_sarcastic

    # re-weight by length
    weight *= lens_not_sarcastic[row['length']]/lens_sarcastic[row['length']]/global_weight_sarcastic

    return weight


df = SentimentDataset.getSarcasmDataset()
global_weight_sarcastic = (df.shape[0] - df['is_sarcastic'].sum())/df['is_sarcastic'].sum()
print(global_weight_sarcastic)
fill_dictionaries(df)
df['length'] = [len(row['headline'].split()) for _, row in df.iterrows()]
print(min(df['length']))
print(lens_sarcastic)
df['weight'] = [weight(row, global_weight_sarcastic, 100) for _, row in df.iterrows()]



print(min(df['length']))


#
# Plot the unweighted data for word length
#

plt.figure(figsize=(6, 6))
ax = plt.gca()
sns.histplot(data=df, x='length', hue='is_sarcastic', multiple='layer',
             bins=[i for i in range(25)],
             element='step',
             palette=['blue', 'red'], alpha=0.0)
setup_minor_ticks(ax)
plt.show()

#
# Plot the weighted data for word length
#

plt.figure(figsize=(6, 6))
ax = plt.gca()
sns.histplot(data=df, x='length', hue='is_sarcastic', multiple='layer',
             weights='weight',
             bins=[i for i in range(25)],
             element='step',
             palette=['blue', 'red'], alpha=0.0)
setup_minor_ticks(ax)
plt.show()

#
# Plot the values of the weights
#

plt.figure(figsize=(6, 6))
ax = plt.gca()
sns.histplot(data=df,
             x='weight',
             hue='is_sarcastic',
             multiple='layer',
             bins=[float(i)/10 for i in range(25)],
             alpha=0.0,
             palette=['blue', 'red'],
             element='step')
setup_minor_ticks(ax)
plt.show()

#
# Plot the frequency of top sarcastic words
#
word_freq_sarc = [[v, k] for k, v in words_sarcastic.items()]
word_freq_sarc.sort(reverse=True)
word_freq_not_sarc = [[words_not_sarcastic[k], k] for _, k in word_freq_sarc]

horizontal_bar(word_freq_sarc, word_freq_not_sarc)
plt.show()

#
# Plot the frequency of top non-sarcastic words
#
word_freq_not_sarc = [[v, k] for k, v in words_not_sarcastic.items()]
word_freq_not_sarc.sort(reverse=True)
word_freq_sarc = [[words_sarcastic[k], k] for _, k in word_freq_not_sarc]

horizontal_bar(word_freq_sarc, word_freq_not_sarc, "Top non-sarcastic words")
plt.show()



#
# Use TF-IDF to balance the classes
#

from sklearn.feature_extraction.text import TfidfVectorizer

# Initialize the TF-IDF vectorizer
# You can adjust parameters like max_features, min_df, max_df as needed
tfidf_vectorizer = TfidfVectorizer(
    max_features=5000,  # Limit to top 5000 words by frequency
    min_df=5,           # Ignore terms that appear in less than 5 documents
    max_df=0.7,         # Ignore terms that appear in more than 70% of documents
    ngram_range=(1,1) 
)

# Fit the vectorizer on all headlines to learn vocabulary
tfidf_vectorizer.fit(df['headline'])

# Get feature names (words)
feature_names = tfidf_vectorizer.get_feature_names_out()


nSarc    = df[df['is_sarcastic']==1]['weight'].sum()
nNotSarc = df[df['is_sarcastic']==0]['weight'].sum()
# feature_names
for feature in ['trump']:
    cond_in = df['is_sarcastic']==1 & (feature in df['headline'])
    cond_out= df['is_sarcastic']==1 & (feature not in df['headline'])
    pFeatSarc    = df[cond_in]['weight'].sum()/nSarc
    pNotFeatSarc = 1-pFeatSarc
    pFeatNotSarc = df[df['is_sarcastic']==0 & (feature in df['headline'])]['weight'].sum()/nNotSarc
    pNotFeatNotSarc = 1-pFeatNotSarc
    df.loc[cond_in,  'weight'] = df.loc[cond_in, 'weight']*pFeatNotSarc/pFeatSarc
    df.loc[cond_out, 'weight'] = df.loc[cond_out, 'weight']*pNotFeatNotSarc/pNotFeatSarc

#
# Plot the re-normalized histogram of word frequency
#
fill_weighted_dict(weighted_sarcastic, df, weight='weight')
word_freq_sarc = [[v, k] for k, v in weighted_sarcastic.items()]
word_freq_sarc.sort(reverse=True)
word_freq_not_sarc = [[words_not_sarcastic[k], k] for _, k in word_freq_sarc]

horizontal_bar(word_freq_sarc, word_freq_not_sarc)
plt.show()


#
# Plot the words which were the highest before normalization
#

word_freq_sarc = [[v, k] for k, v in words_sarcastic.items()]
word_freq_sarc.sort(reverse=True)
word_freq_not_sarc = [[words_not_sarcastic[k], k] for _, k in word_freq_sarc]
word_freq_sarc = [[weighted_sarcastic[k], k] for v, k in word_freq_sarc] # maintain order but transform to weighted words

horizontal_bar(word_freq_sarc, word_freq_not_sarc)
plt.show()

#
# Plot weighted data for word length again
#

plt.figure(figsize=(6, 6))
ax = plt.gca()
sns.histplot(data=df, x='length', hue='is_sarcastic', multiple='layer',
             weights='weight',
             bins=[i for i in range(25)],
             element='step',
             palette=['blue', 'red'], alpha=0.0)
setup_minor_ticks(ax)
plt.show()

#
# Compare word frequency to the trump clickhole dataset
#
dfTrump = ClickholeTrumpDataset.getSarcasmDataset()

