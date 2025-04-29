import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def horizontal_bar(word_freq_sarc: list[list],
                   word_freq_not_sarc: list[list],
                   title: str="Top Sarcastic Words") -> None:

    # Convert to DataFrames
    df_sarc = pd.DataFrame(word_freq_sarc[:20], columns=['count', 'word'])
    df_not_sarc = pd.DataFrame(word_freq_not_sarc[:20], columns=['count', 'word'])

    # Add category column to each DataFrame
    df_sarc['category'] = 'Sarcastic'
    df_not_sarc['category'] = 'Non-sarcastic'
    
    # Combine DataFrames
    df_combined = pd.concat([df_sarc, df_not_sarc])
    
    # Create the plot
    plt.figure(figsize=(6, 6))
    ax = plt.gca()
    # Create horizontal bar plot with words on y-axis
    sns.barplot(
        data=df_combined,
        x='count',
        y='word',
        hue='category',
        palette=['#ff6b6b', '#4ecdc4'],
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
    plt.show()
    
