import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

from plotting.mpl_config      import setup_minor_ticks

'''
Create a histogram from a dataframe
The plot will be saved to a file

Args: 
    dataframe: pandas dataframe 
    x:         column name to plot
    hue:       column name to separate multiple histograms
    fname:     output name of file
    bins:      list of bin edges
    weights:   column name for weights (Optional)
'''
def wireHistogram(dataframe: pd.DataFrame,
                  x:         str,
                  hue:       str,
                  fname:     str,
                  bins:      list,
                  weights:   str | None = None,
                  show:      bool = True,
                  ):
    os.makedirs('figures', exist_ok=True)
    
    plt.figure(figsize=(6, 6))
    ax = plt.gca()

    default_kwargs = {
        'multiple':'layer',
        'element':'step',
        'palette':['blue', 'red'],
        'alpha':0.0,        
        }

    if weights: default_kwargs['weights'] = weights
    
    sns.histplot(data=dataframe,
                 x=x,
                 hue=hue,
                 bins=bins,
                 **default_kwargs
                 )
    setup_minor_ticks(ax)

    plt.savefig(os.path.join('figures', fname + '.pdf'))
    plt.savefig(os.path.join('figures', fname + '.png'))
    if show: plt.show()

'''
Create a horizontal bar chart
The user can supply a list of frequency tables, each table will produce a new 
Set of bars on the chart (up to 5 labels are supported in the color palette).

Args:
    freq:     frequency tables.  Each entry in freq is a list of items, where each item has a 
              frequency and word: e.g. [[ [100, "cheese"] ]]
    category: names applied to legend.
    title:    title displayed at the top of the plot
    maxwords: maximum number of words on y axis 

Returns:
    None
'''
def horizontal_bar(freq: list[list[list]],
                   category: list,
                   title: str="Top Sarcastic Words",
                   fname: str="horizontal_bar.pdf",
                   maxwords: int=20,
                   show: bool = True) -> None:
    os.makedirs('figures', exist_ok=True)
    
    # normalize each entry to the maximum
    maxima = [max([word[0] for word in f]) for f in freq]
    for i, m in enumerate(maxima):
        maximum = 1.0 if m == 0 else m
        freq[i] = [[word[0]/maximum, word[1]] for word in freq[i]]

    # Convert to DataFrames
    dfs = [pd.DataFrame(f[:maxwords], columns=['count', 'word']) for f in freq]

    # Add category column to each DataFrame
    for df, cat in zip(dfs, category):
        df['category'] = cat
    
    # Combine DataFrames
    df_combined = pd.concat(dfs)
    
    # Create the plot
    plt.figure(figsize=(6, 6))
    ax = plt.gca()
    # Create horizontal bar plot with words on y-axis
    sns.barplot(
        data=df_combined,
        x='count',
        y='word',
        hue='category',
        palette=['#ff6b6b', '#4ecdc4', '#f4a261', '#ffd166', '#9a65fd'][:min(len(freq), 5)],
        orient='h'
    )
    all_words = df_combined['word'].unique()
    ax.set_yticks(range(len(all_words)))
    ax.set_yticklabels(all_words)
    ax.yaxis.set_minor_locator(plt.NullLocator())

    # Customize the plot
    plt.title(title, fontsize=16)
    plt.xlabel('Frequency Count', fontsize=12)
    plt.ylabel('Words', fontsize=12)
    plt.legend(title='Category')
    plt.tight_layout()
    plt.yticks(ticks=range(len(df_combined['word'].unique())))
    
    # Show plot
    plt.savefig(os.path.join('figures', fname + '.pdf'))
    plt.savefig(os.path.join('figures', fname + '.png'))
    if show: plt.show()
    
