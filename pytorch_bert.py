import torch
import numpy as np

from torch.utils.data        import Dataset, DataLoader
from transformers            import BertTokenizer, BertForSequenceClassification, AdamW
from transformers            import get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split


# Set random seeds for reproducibility
seed_val = 42
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)

# load the data frame
df = SentimentDataset.getSarcasmDataset()

# Split into train and validation sets
train_texts, val_texts, train_labels, val_labels = train_test_split(
    df['headline'], df['is_sarcastic'], test_size=0.2, random_state=seed_val
)

# Initialize tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Create datasets
train_dataset = SentimentDataset(train_texts, train_labels, tokenizer)
val_dataset = SentimentDataset(val_texts, val_labels, tokenizer)

# Create data loaders
batch_size = 4
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size)

# Load pre-trained model
num_labels = 2  # Binary classification in this example
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
optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8)

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

# Training loop
def train():
    # Set model to training mode
    model.train()
    
    # Track loss
    total_loss = 0
    
    # Train the model
    for batch in train_dataloader:
        # Clear gradients
        optimizer.zero_grad()
        
        # Get inputs
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        token_type_ids = batch['token_type_ids'].to(device)
        labels = batch['labels'].to(device)
        
        # Forward pass
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            labels=labels
        )
        
        loss = outputs.loss
        total_loss += loss.item()
        
        # Backward pass
        loss.backward()
        
        # Clip gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        # Update parameters
        optimizer.step()
        
        # Update learning rate
        scheduler.step()
    
    # Calculate average loss
    avg_loss = total_loss / len(train_dataloader)
    return avg_loss

# Evaluation loop
def evaluate():
    # Set model to evaluation mode
    model.eval()
    
    # Track variables
    total_eval_accuracy = 0
    total_eval_loss = 0
    
    # Evaluate data
    for batch in val_dataloader:
        # Get inputs
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        token_type_ids = batch['token_type_ids'].to(device)
        labels = batch['labels'].to(device)
        
        # Forward pass (no gradients)
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
    
    # Calculate average accuracy and loss
    avg_val_accuracy = total_eval_accuracy / len(val_dataloader)
    avg_val_loss = total_eval_loss / len(val_dataloader)
    
    return avg_val_accuracy, avg_val_loss

# Training and evaluation
for epoch in range(epochs):
    print(f'======== Epoch {epoch + 1} / {epochs} ========')
    
    # Train
    train_loss = train()
    print(f'Training loss: {train_loss:.4f}')
    
    # Evaluate
    val_accuracy, val_loss = evaluate()
    print(f'Validation loss: {val_loss:.4f}')
    print(f'Validation accuracy: {val_accuracy:.4f}')
    print('')

# Save the model
model_save_path = 'fine_tuned_bert_model'
model.save_pretrained(model_save_path)
tokenizer.save_pretrained(model_save_path)
print(f'Model saved to {model_save_path}')

# Make predictions with the fine-tuned model
def predict_sentiment(text):
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
    
    return "Positive" if prediction == 1 else "Negative"

# Example usage of the fine-tuned model
test_text = "I really enjoyed using this product!"
sentiment = predict_sentiment(test_text)
print(f'Text: "{test_text}"')
print(f'Sentiment: {sentiment}')
