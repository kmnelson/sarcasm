import matplotlib.pyplot as plt
import seaborn as sns

from plotting.mpl_config import setup_minor_ticks
from plotting.plots      import horizontal_bar
from data.datasets       import SentimentDataset, SarcasticTrumpDataset
from data.weighting      import WeightingTool

ds = SentimentDataset()
wt = WeightingTool(ds.df) # automatically applies the weights to df 

#
# Plot the unweighted data for word length
#

plt.figure(figsize=(6, 6))
ax = plt.gca()
sns.histplot(data=ds.df, x='length', hue='is_sarcastic', multiple='layer',
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
sns.histplot(data=ds.df, x='length', hue='is_sarcastic', multiple='layer',
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
sns.histplot(data=ds.df,
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
word_freq_sarc, word_freq_not_sarc = wt.getFrequencyTables(is_sarcastic=True)
horizontal_bar([word_freq_sarc, word_freq_not_sarc],
               ['Sarcastic', 'Not Sarcastic'],
               fname='frequencySarcasticUnweighted.pdf')

#
# Plot the frequency of top non-sarcastic words
#
word_freq_sarc, word_freq_not_sarc = wt.getFrequencyTables(is_sarcastic=False)
horizontal_bar([word_freq_sarc, word_freq_not_sarc],
               ['Sarcastic', 'Not Sarcastic'],
               title="Top non-sarcastic words",
               fname='frequencyNonSarcasticUnweighted.pdf')


#
# Plot the re-normalized histogram of word frequency
#
word_freq_sarc, word_freq_not_sarc = wt.getFrequencyTables(is_sarcastic=False, weighted=True)
horizontal_bar([word_freq_sarc, word_freq_not_sarc],
               ['Sarcastic', 'Not Sarcastic'],
               title="Top Non-Sarcastic, After normalization",
               fname='frequencyNonSarcasticWeighted.pdf')


#
# Plot the words which were the highest before normalization
#

unwgt_word_freq_sarc, unwgt_word_freq_not_sarc = wt.getFrequencyTables(is_sarcastic=True)
word_freq_sarc, word_freq_not_sarc = wt.getFrequencyTables(weighted=True, features=[item[1] for item in unwgt_word_freq_sarc])
horizontal_bar([word_freq_sarc, word_freq_not_sarc],
               ['Sarcastic', 'Not Sarcastic'],
               title="After normalization",
               fname="frequencySarcasticWeighted.pdf")

#
# Plot weighted data for word length again
#

plt.figure(figsize=(6, 6))
ax = plt.gca()
sns.histplot(data=ds.df, x='length', hue='is_sarcastic', multiple='layer',
             weights='weight',
             bins=[i for i in range(25)],
             element='step',
             palette=['blue', 'red'], alpha=0.0)
setup_minor_ticks(ax)
plt.show()

#
# Compare word frequency to the trump clickhole dataset
#
dsTrump = SarcasticTrumpDataset()
wtTrump = WeightingTool(dsTrump.df, autoBalance=False)
wtTrump.fill_dictionaries()
trump_freq_sarc, _ = wtTrump.getFrequencyTables(features=[item[1] for item in unwgt_word_freq_sarc])
horizontal_bar([word_freq_sarc, word_freq_not_sarc, trump_freq_sarc],
               ['Sarcastic', 'Not Sarcastic', 'Sarcastic, Trump'],
               title='Comparison to Trump-centric sarcasm',
               fname='frequencyVsTrumpWeighted.pdf')

