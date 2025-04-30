import torch
import torch.nn as nn
import os

from torch.optim              import Adam, Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data         import Dataset, DataLoader
from torchinfo                import summary
from transformers             import BertTokenizer, BertForSequenceClassification
from transformers             import get_linear_schedule_with_warmup
from sklearn.model_selection  import train_test_split
from tqdm.auto                import tqdm
from typing                   import Tuple

from data.datasets            import SentimentDataset


'''
Execute a single training epoch

Args:
    model:      model with tunable parameters
    dataloader: pytorch dataloader
    optimizer:  optimizer to control the learning rate
    scheduler:  scheduler to control the learning rate ramp up
Returns:
    loss: average loss on this epoch
'''
def trainingEpoch(model:      nn.Module,
                  dataloader: DataLoader,
                  optimizer:  Optimizer,
                  scheduler:  LRScheduler,
                  ) -> float:
    
    model.train() # Set model to training mode
    device = next(model.parameters()).device
    
    total_loss = 0
    progress_bar = tqdm(dataloader, desc="Training", leave=True)

    criterion = nn.BCEWithLogitsLoss(reduction='none')
    
    # loop over batches in progress bar
    for batch in progress_bar:
        optimizer.zero_grad()
        
        # Get inputs
        input_ids      = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        token_type_ids = batch['token_type_ids'].to(device)
        labels         = batch['labels'].to(device)
        weights        = batch['weights'].to(device)
        one_hot_labels = nn.functional.one_hot(labels, num_classes=2).float()
        
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            labels=labels
        )

        logits        = outputs.logits
        weighted_loss = torch.dot(weights, torch.mean(criterion(logits, one_hot_labels), dim=-1))
        
        total_loss += weighted_loss.item()
        
        weighted_loss.backward()

        progress_bar.set_description(f"Training - Loss: {weighted_loss.item():.4f}")
        
        # Clip gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()
        scheduler.step()
    
    avg_loss = total_loss / len(dataloader)
    return avg_loss

'''
Evaluate a trained model on some dataloader

Args:
    model: the trained model
    dataloader: the pytorch dataloader to run over
Returns: 
    Average accuracy and loss
'''
def evaluateModel(model:      nn.Module,
                  dataloader: DataLoader) -> Tuple[float, float]:
    model.eval() # Set model to evaluation mode
    device = next(model.parameters()).device
    
    total_eval_accuracy = 0
    total_eval_loss = 0

    progress_bar = tqdm(dataloader, desc="Evaluating", leave=True)
    
    for batch in progress_bar:

        # Get inputs
        input_ids      = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        token_type_ids = batch['token_type_ids'].to(device)
        labels         = batch['labels'].to(device)
        
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                labels=labels
            )
        
        loss = outputs.loss
        logits = outputs.logits
        
        total_eval_loss += loss.item()
        
        # Calculate accuracy
        predictions = torch.argmax(logits, dim=1)
        accuracy = (predictions == labels).float().mean().item()
        total_eval_accuracy += accuracy
        
        progress_bar.set_description(f"Evaluating - Loss: {loss.item():.4f}, Acc: {accuracy:.4f}")
    
    # Calculate average accuracy and loss
    avg_val_accuracy = total_eval_accuracy / len(dataloader)
    avg_val_loss = total_eval_loss / len(dataloader)
    
    return avg_val_accuracy, avg_val_loss


'''
Args:
    model: the model to save
    fname: the file name for the model
'''
def save_model(model: nn.Module,
               tokenizer: BertTokenizer,
               fname: str = "fine_tuned_bert_model"):

    model_save_path = os.path.join('saved_models', fname)
    model.save_pretrained(model_save_path)
    tokenizer.save_pretrained(model_save_path)
    print(f'Model saved to {model_save_path}')

'''
Load a pretrained model for evaluation
'''
def load_bert_model(fname: str = "fine_tuned_bert_model"):
    model_save_path = os.path.join('saved_models', fname)
    model = BertForSequenceClassification.from_pretrained(
        model_save_path
    )
    return model

'''
Args:
    text: the text on which to run the prediction
Returns:
    pred: the binary sarcasm prediction
'''
def predict_sarcasm(text: str):
    # Prepare input
    encoding = tokenizer(
        text,
        add_special_tokens=True,
        max_length=128,
        return_token_type_ids=True,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )
    
    # Move to device
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    token_type_ids = encoding['token_type_ids'].to(device)
    
    # Get prediction
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
    
    logits = outputs.logits
    prediction = torch.argmax(logits, dim=1).item()
    
    return prediction

