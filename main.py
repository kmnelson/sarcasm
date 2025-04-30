import argparse
import torch

from sklearn.model_selection  import train_test_split

from torchinfo                import summary
from torch.optim              import Adam
from torch.utils.data         import DataLoader

from transformers             import BertTokenizer, BertForSequenceClassification
from transformers             import get_linear_schedule_with_warmup

from plotting.plots           import horizontal_bar, wireHistogram
from data.datasets            import SentimentDataset, SarcasticTrumpDataset
from data.weighting           import WeightingTool
from training.utils           import trainingEpoch, evaluateModel, save_model, load_bert_model

'''
Main method for BERT sarcasm detection study

Author: Kevin Nelson

Usage: 
See command line arguments below.  If no arguments are supplied, all steps will automatically be executed.
'''


'''
Create plots showing properties of data:
word length, frequency of top words,
weighting distribution to match sarcast to non-sarcastic,
post-weighting distributions
'''
def visualizeData(show: bool):

    ds = SentimentDataset()
    wt = WeightingTool(ds.df) # automatically applies the weights to df 
    
    # Plot the unweighted data for word length
    wireHistogram(ds.df, x='length', hue='is_sarcastic',
                  bins=[i for i in range(25)],
                  fname='unweightedLength.pdf',
                  show=show)
    
    # Plot the weighted data for word length
    wireHistogram(ds.df, x='length', hue='is_sarcastic', weights='weight',
                  bins=[i for i in range(25)],
                  fname='weightedLength.pdf',
                  show=show)
    
    # Plot the values of the weights
    wireHistogram(ds.df, x='weight', hue='is_sarcastic',
                  bins=[float(i)/10 for i in range(25)],
                  fname='weights.pdf',
                  show=show)
    
    # Plot the frequency of top sarcastic words
    word_freq_sarc, word_freq_not_sarc = wt.getFrequencyTables(is_sarcastic=True)
    horizontal_bar([word_freq_sarc, word_freq_not_sarc],
                   ['Sarcastic', 'Not Sarcastic'],
                   fname='frequencySarcasticUnweighted.pdf',
                   show=show)
    
    # Plot the frequency of top non-sarcastic words
    word_freq_sarc, word_freq_not_sarc = wt.getFrequencyTables(is_sarcastic=False)
    horizontal_bar([word_freq_sarc, word_freq_not_sarc],
                   ['Sarcastic', 'Not Sarcastic'],
                   title="Top non-sarcastic words",
                   fname='frequencyNonSarcasticUnweighted.pdf',
                   show=show)


    # Plot the re-normalized histogram of word frequency
    word_freq_sarc, word_freq_not_sarc = wt.getFrequencyTables(is_sarcastic=False, weighted=True)
    horizontal_bar([word_freq_sarc, word_freq_not_sarc],
                   ['Sarcastic', 'Not Sarcastic'],
                   title="Top Non-Sarcastic, After normalization",
                   fname='frequencyNonSarcasticWeighted.pdf',
                   show=show)
    
    
    # Plot the words which were the highest before normalization
    unwgt_word_freq_sarc, unwgt_word_freq_not_sarc = wt.getFrequencyTables(is_sarcastic=True)
    word_freq_sarc, word_freq_not_sarc = wt.getFrequencyTables(weighted=True, features=[item[1] for item in unwgt_word_freq_sarc])
    horizontal_bar([word_freq_sarc, word_freq_not_sarc],
                   ['Sarcastic', 'Not Sarcastic'],
                   title="After normalization",
                   fname="frequencySarcasticWeighted.pdf",
                   show=show)

    # Compare word frequency to the trump clickhole dataset
    dsTrump = SarcasticTrumpDataset()
    wtTrump = WeightingTool(dsTrump.df, autoBalance=False)
    wtTrump.fill_dictionaries()
    trump_freq_sarc, _ = wtTrump.getFrequencyTables(features=[item[1] for item in unwgt_word_freq_sarc])
    horizontal_bar([word_freq_sarc, word_freq_not_sarc, trump_freq_sarc],
                   ['Sarcastic', 'Not Sarcastic', 'Sarcastic, Trump'],
                   title='Comparison to Trump-centric sarcasm',
                   fname='frequencyVsTrumpWeighted.pdf',
                   show=show)

'''
Train a model

Args:
    wgt: if true, apply the semantic weighting to improve extrapolation accuracy
'''
def train(wgt: bool = False):

    # Set random seeds for reproducibility
    seed_val = 42
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)
    
    # load the dataset
    df = SentimentDataset.getSarcasmDataset()
    if wgt: wt = WeightingTool(df) # automatically applies the weights to df
    
    # Split into train and validation sets
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=seed_val)
    
    # Initialize tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    # Create datasets
    train_dataset = SentimentDataset(dataframe=train_df,
                                     tokenizer=tokenizer)
    val_dataset   = SentimentDataset(dataframe=train_df,
                                     tokenizer=tokenizer)
    
    # Create data loaders
    batch_size = 128
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Load pre-trained model
    num_labels = 2  # Binary classification
    model = BertForSequenceClassification.from_pretrained(
        'bert-base-uncased',
        num_labels=num_labels,
        output_attentions=False,
        output_hidden_states=False,
    )
    
    # Set up device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    # Set up optimizer and learning rate scheduler
    optimizer = Adam(model.parameters(), lr=2e-5, eps=1e-8)
    
    # Number of training epochs
    epochs = 4
    
    # Total number of training steps
    total_steps = len(train_dataloader) * epochs
    
    # Learning rate scheduler
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )
    
    summary(model)

    for epoch in range(epochs):
        print(f'======== Epoch {epoch + 1} / {epochs} ========')
    
        # Train
        train_loss = trainingEpoch(model,
                                   train_dataloader,
                                   optimizer,
                                   scheduler)
        print(f'Training loss: {train_loss:.4f}')
        
        # Evaluate
        val_accuracy, val_loss = evaluateModel(model, val_dataloader)
        print(f'Validation loss: {val_loss:.4f}')
        print(f'Validation accuracy: {val_accuracy:.4f}')
        print('')

    fname = "fine_tuned_bert_model"
    if wgt: fname += "_wgt"
    save_model(model, tokenizer, fname)

'''
Evaluate the model with and without the semantic weighting applied
and compare the accuracy in extrapolating to clickhole headlines
with trump mentioned
'''
def evaluate():
    # load models and tokenizers
    model_unwgt, tokenizer_unwgt = load_bert_model("fine_tuned_bert_model")
    model_wgt,   tokenizer_wgt   = load_bert_model("fine_tuned_bert_model_wgt")

    # compute F1 score of the trained models
    df = SentimentDataset.getSarcasmDataset()

    full_dataset_unwgt = SentimentDataset(dataframe=df,
                                          tokenizer=tokenizer_unwgt)
    full_dataloader_unwgt = DataLoader(full_dataset_unwgt,
                                       batch_size=128)
    acc, _, f1_unwgt = evaluateModel(model_unwgt, full_dataloader_unwgt)

    full_dataset_wgt = SentimentDataset(dataframe=df,
                                        tokenizer=tokenizer_wgt)
    full_dataloader_wgt = DataLoader(full_dataset_wgt,
                                     batch_size=128)
    acc, _, f1_wgt = evaluateModel(model_wgt, full_dataloader_wgt)

    print('F1, unweighted: ', f1_unwgt)
    print('F1, weighted:   ', f1_wgt)
    
    # try generalization to a dataset of headlines from Clickhole
    # which are selected to have trump mentioned
    df     = SarcasticTrumpDataset.getSarcasmDataset()
    trump_dataset_unwgt    = SentimentDataset(dataframe=df,
                                              tokenizer=tokenizer_unwgt)
    trump_dataloader_unwgt = DataLoader(trump_dataset_unwgt, batch_size=128)

    trump_dataset_wgt      = SentimentDataset(dataframe=df,
                                              tokenizer=tokenizer_wgt)
    trump_dataloader_wgt   = DataLoader(trump_dataset_wgt, batch_size=128)

    val_accuracy_unwgt, _, __ = evaluateModel(model_unwgt, trump_dataloader_unwgt)
    val_accuracy_wgt,   _, __ = evaluateModel(model_wgt,   trump_dataloader_wgt)
    print("Accuracy in extrapolating to trump-centric sarcastic headlines:")
    print("No semantic weighting:  ", val_accuracy_unwgt)
    print("With semantic weighting:", val_accuracy_wgt)
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--visualize", "-v", action="store_true", help="Run the data visualization to create plots")
    parser.add_argument("--train",     "-t", action="store_true", help="Execute model training")
    parser.add_argument("--train-wgt", "-w", action="store_true", help="Execute model training with weights")
    parser.add_argument("--eval",      "-e", action="store_true", help="Evaluate the trained model(s) on the trump-centric sarcastic dataset")
    parser.add_argument("--show",      "-s", type=bool, default=False, help="Show plots interactively as they are produced")
    args = parser.parse_args()

    if not args.visualize and not args.train and not args.train_wgt and not args.eval:
        print('No arguments supplied.  Running *all* steps.')
        visualizeData(args.show)
        train()
        train(wgt=True)
        evaluate()
    else:
        if args.visualize: visualizeData(args.show)
        if args.train:     train()
        if args.train_wgt: train(wgt=True)
        if args.eval:      evaluate()
    

main()
