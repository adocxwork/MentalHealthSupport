"""
IMPROVED EMOTION CLASSIFICATION MODEL
Target: ~80% Hamming & Subset Accuracy

Key Improvements:
1. RoBERTa model (better than BERT)
2. Focal Loss for hard examples
3. Optimized thresholds per emotion
4. Class weighting for imbalanced data
5. Extended training with better scheduling
6. Label smoothing

Requirements:
pip install transformers torch pandas scikit-learn tqdm numpy
"""

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import (
    RobertaTokenizer, 
    RobertaForSequenceClassification,
    get_cosine_schedule_with_warmup
)
from torch.optim import AdamW
from sklearn.metrics import accuracy_score, f1_score, hamming_loss
from tqdm import tqdm
import json
import os

# ============================================================
# CONFIGURATION - OPTIMIZED FOR HIGH ACCURACY
# ============================================================
class Config:
    EMOTIONS = [
        'admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring',
        'confusion', 'curiosity', 'desire', 'disappointment', 'disapproval',
        'disgust', 'embarrassment', 'excitement', 'fear', 'gratitude', 'grief',
        'joy', 'love', 'nervousness', 'optimism', 'pride', 'realization',
        'relief', 'remorse', 'sadness', 'surprise', 'neutral'
    ]

    NUM_LABELS = 28
    
    # IMPROVED SETTINGS
    MAX_LENGTH = 256              # Increased from 128 for better context
    BATCH_SIZE = 8                # Smaller for more stable gradients
    EPOCHS = 8                    # Increased from 4
    LEARNING_RATE = 1e-5          # Lower for more stable fine-tuning
    WARMUP_RATIO = 0.1            # 10% warmup
    MODEL_NAME = 'roberta-base'   # RoBERTa outperforms BERT
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Files
    TRAIN_FILE = 'train.tsv'
    DEV_FILE = 'dev.tsv'
    TEST_FILE = 'test.tsv'
    MODEL_SAVE_PATH = 'emotion_roberta_model'

    # Training optimizations
    GRADIENT_ACCUMULATION_STEPS = 4  # Effective batch size = 8 * 4 = 32
    MAX_GRAD_NORM = 1.0
    WEIGHT_DECAY = 0.01
    LABEL_SMOOTHING = 0.1         # Prevent overconfidence
    
    # Focal loss parameters
    USE_FOCAL_LOSS = True
    FOCAL_ALPHA = 0.25
    FOCAL_GAMMA = 2.0
    
    # Threshold optimization
    OPTIMIZE_THRESHOLDS = True
    DEFAULT_THRESHOLD = 0.5

print("="*70)
print("🚀 IMPROVED EMOTION CLASSIFIER - TARGET: 80% ACCURACY")
print("="*70)
print(f"Device: {Config.DEVICE}")
print(f"Model: {Config.MODEL_NAME}")
print(f"Batch Size: {Config.BATCH_SIZE} (effective: {Config.BATCH_SIZE * Config.GRADIENT_ACCUMULATION_STEPS})")
print(f"Epochs: {Config.EPOCHS}")
print(f"Focal Loss: {Config.USE_FOCAL_LOSS}")
print(f"Threshold Optimization: {Config.OPTIMIZE_THRESHOLDS}")
print("="*70)

# ============================================================
# FOCAL LOSS FOR MULTI-LABEL CLASSIFICATION
# ============================================================
class FocalLoss(nn.Module):
    """
    Focal Loss for multi-label classification
    Helps the model focus on hard examples
    """
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        
    def forward(self, inputs, targets):
        BCE_loss = nn.functional.binary_cross_entropy_with_logits(
            inputs, targets, reduction='none'
        )
        pt = torch.exp(-BCE_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        return focal_loss.mean()

# ============================================================
# DATASET CLASS
# ============================================================
class EmotionDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.FloatTensor(label)
        }

# ============================================================
# DATA LOADING
# ============================================================
def load_goemotion_data(file_path):
    print(f"\nLoading {file_path}...")
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    df = pd.read_csv(file_path, sep='\t', header=None)

    if len(df.columns) == 3:
        df.columns = ['text', 'emotions', 'id']
    elif len(df.columns) == 2:
        df.columns = ['text', 'emotions']
    else:
        raise ValueError(f"Unexpected columns: {len(df.columns)}")

    texts = df['text'].tolist()
    labels = []
    
    for emotions_str in df['emotions']:
        label = [0] * Config.NUM_LABELS
        if pd.notna(emotions_str):
            emotion_indices = str(emotions_str).split(',')
            for idx_str in emotion_indices:
                try:
                    idx = int(idx_str.strip())
                    if 0 <= idx < Config.NUM_LABELS:
                        label[idx] = 1
                except ValueError:
                    continue
        labels.append(label)

    print(f"✓ Loaded {len(texts)} examples")
    return texts, labels

# ============================================================
# CALCULATE CLASS WEIGHTS
# ============================================================
def calculate_class_weights(labels):
    """Calculate weights for imbalanced classes"""
    labels_array = np.array(labels)
    pos_counts = labels_array.sum(axis=0)
    neg_counts = len(labels_array) - pos_counts
    
    # Calculate positive class weight
    weights = neg_counts / (pos_counts + 1e-5)
    weights = np.clip(weights, 0.5, 10.0)  # Clip extreme values
    
    print("\n📊 Class Weights (top 10 imbalanced):")
    weight_info = list(zip(Config.EMOTIONS, weights, pos_counts))
    weight_info.sort(key=lambda x: x[1], reverse=True)
    
    for emotion, weight, count in weight_info[:10]:
        print(f"   {emotion:15s}: weight={weight:.2f}, samples={int(count)}")
    
    return torch.FloatTensor(weights)

# ============================================================
# THRESHOLD OPTIMIZATION
# ============================================================
def optimize_thresholds(model, dataloader, device):
    """Find optimal threshold for each emotion using validation set"""
    print("\n🎯 Optimizing thresholds...")
    
    model.eval()
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Collecting predictions"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels']
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.sigmoid(outputs.logits).cpu().numpy()
            
            all_probs.extend(probs)
            all_labels.extend(labels.numpy())
    
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    
    # Find best threshold for each emotion
    thresholds = []
    for i in range(Config.NUM_LABELS):
        best_threshold = 0.5
        best_f1 = 0
        
        for threshold in np.arange(0.3, 0.7, 0.05):
            preds = (all_probs[:, i] > threshold).astype(int)
            f1 = f1_score(all_labels[:, i], preds, zero_division=0)
            
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
        
        thresholds.append(best_threshold)
    
    print("✓ Optimal thresholds found")
    return np.array(thresholds)

# ============================================================
# TRAINING
# ============================================================
def train_epoch(model, dataloader, optimizer, scheduler, device, epoch, criterion, class_weights):
    model.train()
    total_loss = 0
    optimizer.zero_grad()
    
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}")
    
    for step, batch in enumerate(progress_bar):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        # Apply label smoothing
        if Config.LABEL_SMOOTHING > 0:
            labels = labels * (1 - Config.LABEL_SMOOTHING) + 0.5 * Config.LABEL_SMOOTHING
        
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        
        # Custom loss with class weights
        if Config.USE_FOCAL_LOSS:
            loss = criterion(outputs.logits, labels)
        else:
            loss = nn.functional.binary_cross_entropy_with_logits(
                outputs.logits, labels, pos_weight=class_weights
            )
        
        loss = loss / Config.GRADIENT_ACCUMULATION_STEPS
        loss.backward()
        
        if (step + 1) % Config.GRADIENT_ACCUMULATION_STEPS == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), Config.MAX_GRAD_NORM)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        
        total_loss += loss.item() * Config.GRADIENT_ACCUMULATION_STEPS
        progress_bar.set_postfix({'loss': f'{loss.item() * Config.GRADIENT_ACCUMULATION_STEPS:.4f}'})
    
    return total_loss / len(dataloader)

# ============================================================
# EVALUATION
# ============================================================
def evaluate(model, dataloader, device, thresholds=None):
    model.eval()
    all_preds = []
    all_labels = []
    total_loss = 0
    
    if thresholds is None:
        thresholds = np.full(Config.NUM_LABELS, Config.DEFAULT_THRESHOLD)
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            
            # Calculate loss
            loss = nn.functional.binary_cross_entropy_with_logits(
                outputs.logits, labels
            )
            
            probs = torch.sigmoid(outputs.logits).cpu().numpy()
            
            # Apply per-emotion thresholds
            preds = np.zeros_like(probs)
            for i in range(Config.NUM_LABELS):
                preds[:, i] = (probs[:, i] > thresholds[i]).astype(float)
            
            total_loss += loss.item()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # Calculate metrics
    subset_accuracy = accuracy_score(all_labels, all_preds)
    hamming_accuracy = 1 - hamming_loss(all_labels, all_preds)  # Convert to accuracy
    hamming_loss_val = hamming_loss(all_labels, all_preds)
    
    f1_micro = f1_score(all_labels, all_preds, average='micro', zero_division=0)
    f1_macro = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    f1_per_emotion = f1_score(all_labels, all_preds, average=None, zero_division=0)
    
    return {
        'loss': total_loss / len(dataloader),
        'subset_accuracy': subset_accuracy,
        'hamming_accuracy': hamming_accuracy,
        'hamming_loss': hamming_loss_val,
        'f1_micro': f1_micro,
        'f1_macro': f1_macro,
        'f1_per_emotion': f1_per_emotion
    }

def print_metrics(metrics):
    print(f"\n{'='*70}")
    print("📊 EVALUATION METRICS")
    print(f"{'='*70}")
    print(f"Loss:              {metrics['loss']:.4f}")
    print(f"Subset Accuracy:   {metrics['subset_accuracy']:.4f} 🎯 (target: 0.80)")
    print(f"Hamming Accuracy:  {metrics['hamming_accuracy']:.4f} 🎯 (target: 0.80)")
    print(f"Hamming Loss:      {metrics['hamming_loss']:.4f} (lower is better)")
    print(f"F1 Micro:          {metrics['f1_micro']:.4f}")
    print(f"F1 Macro:          {metrics['f1_macro']:.4f}")
    print(f"{'='*70}")

# ============================================================
# MAIN
# ============================================================
def main():
    print("\n📚 PHASE 1: DATA LOADING")
    print("="*70)
    
    train_texts, train_labels = load_goemotion_data(Config.TRAIN_FILE)
    dev_texts, dev_labels = load_goemotion_data(Config.DEV_FILE)
    
    print(f"\nDataset: {len(train_texts)} train, {len(dev_texts)} dev")
    
    # Calculate class weights
    class_weights = calculate_class_weights(train_labels).to(Config.DEVICE)
    
    # Save emotion mapping
    emotion_mapping = {
        'emotions': Config.EMOTIONS,
        'emotion_to_idx': {e: i for i, e in enumerate(Config.EMOTIONS)},
        'idx_to_emotion': {i: e for i, e in enumerate(Config.EMOTIONS)}
    }
    with open('emotion_mapping.json', 'w') as f:
        json.dump(emotion_mapping, f, indent=2)
    
    print("\n🤖 PHASE 2: MODEL INITIALIZATION")
    print("="*70)
    
    tokenizer = RobertaTokenizer.from_pretrained(Config.MODEL_NAME)
    
    train_dataset = EmotionDataset(train_texts, train_labels, tokenizer, Config.MAX_LENGTH)
    dev_dataset = EmotionDataset(dev_texts, dev_labels, tokenizer, Config.MAX_LENGTH)
    
    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=0)
    dev_loader = DataLoader(dev_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=0)
    
    model = RobertaForSequenceClassification.from_pretrained(
        Config.MODEL_NAME,
        num_labels=Config.NUM_LABELS,
        problem_type="multi_label_classification"
    )
    model.to(Config.DEVICE)
    
    print(f"✓ Model loaded: {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Setup training
    optimizer = AdamW(model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY)
    
    total_steps = len(train_loader) * Config.EPOCHS // Config.GRADIENT_ACCUMULATION_STEPS
    warmup_steps = int(total_steps * Config.WARMUP_RATIO)
    
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )
    
    criterion = FocalLoss(Config.FOCAL_ALPHA, Config.FOCAL_GAMMA) if Config.USE_FOCAL_LOSS else None
    
    print(f"✓ Training steps: {total_steps}, Warmup: {warmup_steps}")
    
    print("\n🚀 PHASE 3: TRAINING")
    print("="*70)
    
    best_hamming_acc = 0
    best_subset_acc = 0
    best_combined = 0
    optimal_thresholds = None
    training_history = []
    
    for epoch in range(1, Config.EPOCHS + 1):
        print(f"\n{'='*70}")
        print(f"EPOCH {epoch}/{Config.EPOCHS}")
        print(f"{'='*70}")
        
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, Config.DEVICE, epoch, criterion, class_weights)
        print(f"✓ Training Loss: {train_loss:.4f}")
        
        # Optimize thresholds every 2 epochs
        if Config.OPTIMIZE_THRESHOLDS and epoch % 2 == 0:
            optimal_thresholds = optimize_thresholds(model, dev_loader, Config.DEVICE)
        
        val_metrics = evaluate(model, dev_loader, Config.DEVICE, optimal_thresholds)
        print_metrics(val_metrics)
        
        # Track metrics
        training_history.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'val_hamming_acc': val_metrics['hamming_accuracy'],
            'val_subset_acc': val_metrics['subset_accuracy'],
            'val_f1_macro': val_metrics['f1_macro']
        })
        
        # Save best model based on combined metric
        combined_acc = (val_metrics['hamming_accuracy'] + val_metrics['subset_accuracy']) / 2
        
        if combined_acc > best_combined:
            best_combined = combined_acc
            best_hamming_acc = val_metrics['hamming_accuracy']
            best_subset_acc = val_metrics['subset_accuracy']
            
            print(f"\n✨ NEW BEST MODEL!")
            print(f"   Combined: {combined_acc:.4f}")
            print(f"   Saving to: {Config.MODEL_SAVE_PATH}")
            
            model.save_pretrained(Config.MODEL_SAVE_PATH)
            tokenizer.save_pretrained(Config.MODEL_SAVE_PATH)
            
            # Save thresholds and metrics
            save_data = {
                'thresholds': optimal_thresholds.tolist() if optimal_thresholds is not None else None,
                'metrics': {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in val_metrics.items()}
            }
            with open(os.path.join(Config.MODEL_SAVE_PATH, 'best_config.json'), 'w') as f:
                json.dump(save_data, f, indent=2)
    
    with open('training_history.json', 'w') as f:
        json.dump(training_history, f, indent=2)
    
    print("\n" + "="*70)
    print("🎉 TRAINING COMPLETE!")
    print("="*70)
    print(f"Best Hamming Accuracy:  {best_hamming_acc:.4f}")
    print(f"Best Subset Accuracy:   {best_subset_acc:.4f}")
    print(f"Combined Score:         {best_combined:.4f}")
    print(f"\nModel saved to: {Config.MODEL_SAVE_PATH}")
    print("="*70)

if __name__ == "__main__":
    main()