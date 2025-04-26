import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from collections import Counter
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
from torch.nn.utils.rnn import pack_padded_sequence
import random
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import os
import json
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import precision_recall_fscore_support, classification_report
import pandas as pd


def get_device():
    # apple silicon mps
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Training on Apple Silicon GPU via MPS")
    else:
        device = torch.device("cpu")
        print("GPU not available, training using CPU")
    return device


def load_data(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


# under sampling majority class (no_relation)
def create_balanced_dataset(data, max_ratio=3.0):
    #  samples per relation
    relation_counts = Counter([sample["relation"] for sample in data])

    majority_class = relation_counts.most_common(1)[0][0]
    majority_count = relation_counts[majority_class]
    minority_count = sum(count for rel, count in relation_counts.items() if rel != majority_class)

    # current imbalance ratio
    current_ratio = majority_count / max(1, minority_count)

    print(f"\nCurrent imbalance: {majority_class} vs others = {current_ratio:.1f}:1")

    # majority class samples to keep
    target_majority_count = int(minority_count * max_ratio)

    majority_samples = [sample for sample in data if sample["relation"] == majority_class]
    minority_samples = [sample for sample in data if sample["relation"] != majority_class]
    random.shuffle(majority_samples)
    sampled_majority = majority_samples[:target_majority_count]
    balanced_data = sampled_majority + minority_samples
    random.shuffle(balanced_data)

    print(f"\nBalanced dataset created:")
    print(f"  Original: {len(data)} total samples, {majority_count} {majority_class}")
    print(f"  Balanced: {len(balanced_data)} total samples, {target_majority_count} {majority_class}")
    print(f"  Imbalance ratio reduced from {current_ratio:.1f}:1 to {max_ratio:.1f}:1")

    # new distribution
    new_counts = Counter([sample["relation"] for sample in balanced_data])
    print("\nNew distribution:")
    for rel, count in new_counts.most_common():
        print(f"  {rel}: {count} samples ({count / len(balanced_data) * 100:.1f}%)")

    return balanced_data


def build_vocab(data, min_freq=1):
    # tokens for train set
    word_counts = Counter(word.lower() for sample in data for word in sample["token"])
    vocab = {word: i + 4 for i, (word, count) in enumerate(word_counts.items()) if count >= min_freq}
    vocab["<PAD>"] = 0
    vocab["<UNK>"] = 1
    return vocab


def build_pos_ner_vocabs(data):
    # pos and ner vocab
    pos_tags = Counter()
    ner_tags = Counter()

    for sample in data:
        pos_tags.update(sample["stanford_pos"])
        ner_tags.update(sample["stanford_ner"])

    pos_vocab = {"<PAD>": 0, "<UNK>": 1}
    pos_vocab.update({tag: i + 2 for i, tag in enumerate(pos_tags)})
    ner_vocab = {"<PAD>": 0, "<UNK>": 1}
    ner_vocab.update({tag: i + 2 for i, tag in enumerate(ner_tags)})

    return pos_vocab, ner_vocab


def build_entity_type_vocab(data):
    # entity type vocab
    entity_types = set()
    for sample in data:
        entity_types.add(sample["subj_type"])
        entity_types.add(sample["obj_type"])

    entity_type_vocab = {"<UNK>": 0}
    entity_type_vocab.update({t: i + 1 for i, t in enumerate(entity_types)})
    return entity_type_vocab


def preprocess_data(data, word_vocab, pos_vocab, ner_vocab, entity_type_vocab, max_len=60):
    processed_data = []

    for idx, sample in enumerate(data):
        tokens = [t.lower() for t in sample["token"]]
        pos_tags = sample["stanford_pos"]
        ner_tags = sample["stanford_ner"]

        subj_start, subj_end = sample["subj_start"], sample["subj_end"]
        obj_start, obj_end = sample["obj_start"], sample["obj_end"]

        subj_positions = [i - ((subj_start + subj_end) // 2) for i in range(len(tokens))]
        obj_positions = [i - ((obj_start + obj_end) // 2) for i in range(len(tokens))]

        subj_type = entity_type_vocab.get(sample["subj_type"], 0)
        obj_type = entity_type_vocab.get(sample["obj_type"], 0)

        if len(tokens) > max_len:
            tokens = tokens[:max_len]
            pos_tags = pos_tags[:max_len]
            ner_tags = ner_tags[:max_len]
            subj_positions = subj_positions[:max_len]
            obj_positions = obj_positions[:max_len]

        token_ids = [word_vocab.get(t, word_vocab["<UNK>"]) for t in tokens]
        pos_ids = [pos_vocab.get(t, pos_vocab["<UNK>"]) for t in pos_tags]
        ner_ids = [ner_vocab.get(t, ner_vocab["<UNK>"]) for t in ner_tags]

        token_len = len(token_ids)
        pad_len = max_len - token_len

        token_ids += [word_vocab["<PAD>"]] * pad_len
        pos_ids += [pos_vocab["<PAD>"]] * pad_len
        ner_ids += [ner_vocab["<PAD>"]] * pad_len
        subj_positions += [0] * pad_len
        obj_positions += [0] * pad_len

        # get relation labels
        label = sample["relation"]

        processed_data.append({
            "token_ids": token_ids,
            "pos_ids": pos_ids,
            "ner_ids": ner_ids,
            "subj_positions": subj_positions,
            "obj_positions": obj_positions,
            "subj_type": subj_type,
            "obj_type": obj_type,
            "length": token_len,
            "label": label,
            "original_index": idx
        })

    return processed_data


class RelationDataset(Dataset):
    def __init__(self, data, label_encoder=None):
        self.data = data

        self.label_encoder = label_encoder
        if label_encoder:
            self.labels = label_encoder.transform([sample["label"] for sample in data])
        else:
            self.label_encoder = LabelEncoder()
            self.labels = self.label_encoder.fit_transform([sample["label"] for sample in data])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]

        # return as tensors
        return {
            "token_ids": torch.tensor(sample["token_ids"], dtype=torch.long),
            "pos_ids": torch.tensor(sample["pos_ids"], dtype=torch.long),
            "ner_ids": torch.tensor(sample["ner_ids"], dtype=torch.long),
            "subj_positions": torch.tensor(sample["subj_positions"], dtype=torch.long),
            "obj_positions": torch.tensor(sample["obj_positions"], dtype=torch.long),
            "subj_type": torch.tensor(sample["subj_type"], dtype=torch.long),
            "obj_type": torch.tensor(sample["obj_type"], dtype=torch.long),
            "length": sample["length"],
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
            "original_index": sample["original_index"]
        }


class LSTM(nn.Module):

    def __init__(self, word_vocab_size, pos_vocab_size, ner_vocab_size, entity_type_vocab_size,
                 embedding_dim=200, pos_dim=30, ner_dim=30, entity_type_dim=30, position_dim=30,
                 hidden_dim=256, num_layers=2, num_classes=10, dropout_rate=0.3, max_len=60):
        super(LSTM, self).__init__()

        self.max_len = max_len

        # Embeddings
        self.word_embedding = nn.Embedding(word_vocab_size, embedding_dim, padding_idx=0)
        self.pos_embedding = nn.Embedding(pos_vocab_size, pos_dim, padding_idx=0)
        self.ner_embedding = nn.Embedding(ner_vocab_size, ner_dim, padding_idx=0)
        self.entity_type_embedding = nn.Embedding(entity_type_vocab_size, entity_type_dim)

        self.position_embedding = nn.Embedding(2 * max_len + 1, position_dim, padding_idx=max_len)

        self.position_offset = max_len

        self.input_dim = embedding_dim + pos_dim + ner_dim + position_dim * 2

        # BiLSTM layer
        self.lstm = nn.LSTM(
            self.input_dim, hidden_dim // 2, num_layers,
            batch_first=True, bidirectional=True, dropout=dropout_rate if num_layers > 1 else 0
        )

        self.entity_rep_dim = entity_type_dim * 2

        # fully connected layers
        self.fc1 = nn.Linear(hidden_dim + self.entity_rep_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, num_classes)

        # Batch normalization
        self.batch_norm = nn.BatchNorm1d(hidden_dim)

    def forward(self, token_ids, pos_ids, ner_ids, subj_positions, obj_positions, subj_type, obj_type, lengths):
        batch_size = token_ids.size(0)
        seq_len = token_ids.size(1)

        # embedding lookup
        word_embeds = self.word_embedding(token_ids)
        pos_embeds = self.pos_embedding(pos_ids)
        ner_embeds = self.ner_embedding(ner_ids)

        adjusted_subj_positions = torch.clamp(subj_positions + self.position_offset, 0, 2 * self.max_len)
        adjusted_obj_positions = torch.clamp(obj_positions + self.position_offset, 0, 2 * self.max_len)

        # position embeddings
        subj_pos_embeds = self.position_embedding(adjusted_subj_positions)
        obj_pos_embeds = self.position_embedding(adjusted_obj_positions)

        # entity type embeddings
        subj_type_embeds = self.entity_type_embedding(subj_type)
        obj_type_embeds = self.entity_type_embedding(obj_type)

        # concatenate embeddings
        embeds = torch.cat([
            word_embeds, pos_embeds, ner_embeds,
            subj_pos_embeds, obj_pos_embeds
        ], dim=2)

        packed_input = pack_padded_sequence(embeds, lengths.cpu(), batch_first=True, enforce_sorted=False)

        # BiLSTM forward pass
        packed_output, (hidden, _) = self.lstm(packed_input)

        # use the final hidden state from both directions
        # hidden shape: [num_layers * num_directions, batch_size, hidden_dim//2]

        # get the last layer's hidden state (both directions)
        last_layer_hidden = hidden[-2:].transpose(0, 1).contiguous()
        # last_layer_hidden shape: [batch_size, 2, hidden_dim//2]

        # combine forward and backward hidden states
        final_hidden = last_layer_hidden.view(batch_size, -1)
        # final_hidden shape: [batch_size, hidden_dim]

        # concatenate with entity type representations
        entity_rep = torch.cat([subj_type_embeds, obj_type_embeds], dim=1)
        combined = torch.cat([final_hidden, entity_rep], dim=1)

        # classification
        out = self.fc1(combined)
        out = self.batch_norm(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)

        return out


# additional function to prepare data for batch processing in Pytorch's DataLoader, default collation of pytorch
# does not handle variable-length sequences well
def collate_fn(batch):
    batch = sorted(batch, key=lambda x: x["length"], reverse=True)

    # Collect items
    token_ids = torch.stack([item["token_ids"] for item in batch])
    pos_ids = torch.stack([item["pos_ids"] for item in batch])
    ner_ids = torch.stack([item["ner_ids"] for item in batch])
    subj_positions = torch.stack([item["subj_positions"] for item in batch])
    obj_positions = torch.stack([item["obj_positions"] for item in batch])
    subj_type = torch.stack([item["subj_type"] for item in batch])
    obj_type = torch.stack([item["obj_type"] for item in batch])
    lengths = torch.tensor([item["length"] for item in batch])
    labels = torch.stack([item["label"] for item in batch])
    original_indices = [item["original_index"] for item in batch]

    return {
        "token_ids": token_ids,
        "pos_ids": pos_ids,
        "ner_ids": ner_ids,
        "subj_positions": subj_positions,
        "obj_positions": obj_positions,
        "subj_type": subj_type,
        "obj_type": obj_type,
        "lengths": lengths,
        "labels": labels,
        "original_indices": original_indices
    }


# calculate weights for imbalance data, model pays more attention to examples from minority class
def calculate_class_weights(labels):
    class_counts = Counter(labels)
    total = sum(class_counts.values())
    class_weights = {cls: total / count for cls, count in class_counts.items()}

    total_weight = sum(class_weights.values())
    class_weights = {cls: weight / total_weight * len(class_weights) for cls, weight in class_weights.items()}

    # convert to tensor
    weights = torch.zeros(len(class_weights))
    for cls, weight in class_weights.items():
        weights[cls] = weight

    return weights


def train_and_evaluate(train_data, dev_data, test_data=None, batch_size=32, epochs=20,
                       lr=0.001, max_len=60, embedding_dim=200, hidden_dim=256, dropout_rate=0.3, balance_ratio=3.0):
    # vocabularies for training
    word_vocab = build_vocab(train_data)
    pos_vocab, ner_vocab = build_pos_ner_vocabs(train_data)
    entity_type_vocab = build_entity_type_vocab(train_data)

    print(f"Word vocabulary size: {len(word_vocab)}")
    print(f"POS vocabulary size: {len(pos_vocab)}")
    print(f"NER vocabulary size: {len(ner_vocab)}")
    print(f"Entity type vocabulary size: {len(entity_type_vocab)}")

    balanced_train_data = create_balanced_dataset(train_data, max_ratio=balance_ratio)
    processed_train = preprocess_data(balanced_train_data, word_vocab, pos_vocab, ner_vocab, entity_type_vocab, max_len)

    train_dataset = RelationDataset(processed_train)
    label_encoder = train_dataset.label_encoder
    processed_dev = preprocess_data(dev_data, word_vocab, pos_vocab, ner_vocab, entity_type_vocab, max_len)
    dev_dataset = RelationDataset(processed_dev, label_encoder)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    dev_loader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    # class weights for the loss function
    class_weights = calculate_class_weights(train_dataset.labels)

    # initialize model
    device = get_device()
    model = LSTM(
        word_vocab_size=len(word_vocab),
        pos_vocab_size=len(pos_vocab),
        ner_vocab_size=len(ner_vocab),
        entity_type_vocab_size=len(entity_type_vocab),
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        num_classes=len(label_encoder.classes_),
        dropout_rate=dropout_rate,
        max_len=max_len
    ).to(device)

    # loss and optimizer
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.5, verbose=False)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params:,} (trainable: {trainable_params:,})")

    # training loop
    train_losses = []
    dev_losses = []
    best_f1 = 0
    best_model_state = None
    patience = 5
    counter = 0

    print(f"Training on {device} with {len(train_dataset)} samples")
    print(f"Number of classes: {len(label_encoder.classes_)}")
    print(f"Class distribution: {dict(Counter(train_dataset.labels))}")

    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0
        for batch in train_loader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

            optimizer.zero_grad()

            # forward pass
            outputs = model(
                batch["token_ids"], batch["pos_ids"], batch["ner_ids"],
                batch["subj_positions"], batch["obj_positions"],
                batch["subj_type"], batch["obj_type"], batch["lengths"]
            )

            # calculate loss
            loss = criterion(outputs, batch["labels"])

            # backward pass and optimization
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()

        #  average training loss
        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # evaluation
        model.eval()
        dev_loss = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in dev_loader:
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

                # forward pass
                outputs = model(
                    batch["token_ids"], batch["pos_ids"], batch["ner_ids"],
                    batch["subj_positions"], batch["obj_positions"],
                    batch["subj_type"], batch["obj_type"], batch["lengths"]
                )

                # calculate loss
                loss = criterion(outputs, batch["labels"])
                dev_loss += loss.item()

                # get predictions
                _, preds = torch.max(outputs, 1)

                # collect predictions and labels
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(batch["labels"].cpu().numpy())

        # calculate average validation loss, update learning rate
        avg_dev_loss = dev_loss / len(dev_loader)
        dev_losses.append(avg_dev_loss)

        scheduler.step(avg_dev_loss)

        # metrics
        accuracy = accuracy_score(all_labels, all_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='macro')

        print(f"Epoch {epoch + 1}/{epochs}")
        print(f"Train Loss: {avg_train_loss:.4f}, Dev Loss: {avg_dev_loss:.4f}")
        print(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")

        # save model
        if f1 > best_f1:
            best_f1 = f1
            best_model_state = model.state_dict().copy()
            counter = 0

            # classification report
            print("\nClassification Report:")
            print(classification_report(
                all_labels,
                all_preds,
                labels=range(len(label_encoder.classes_)),
                target_names=label_encoder.classes_,
                zero_division=0
            ))
        else:
            counter += 1

        # early stopping trigger
        if counter >= patience:
            print(f"Early stopping triggered after {epoch + 1} epochs")
            break

    if best_model_state:
        model.load_state_dict(best_model_state)

    save_dict = {
        "model_state_dict": model.state_dict(),
        "word_vocab": word_vocab,
        "pos_vocab": pos_vocab,
        "ner_vocab": ner_vocab,
        "entity_type_vocab": entity_type_vocab,
        "label_encoder": label_encoder,
        "max_len": max_len
    }
    torch.save(save_dict, "../models_parameters/re_lstm_model.pt")

    # evaluate on test data
    if test_data:
        processed_test = preprocess_data(test_data, word_vocab, pos_vocab, ner_vocab, entity_type_vocab, max_len)
        test_dataset = RelationDataset(processed_test, label_encoder)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

        model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in test_loader:
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

                # forward pass
                outputs = model(
                    batch["token_ids"], batch["pos_ids"], batch["ner_ids"],
                    batch["subj_positions"], batch["obj_positions"],
                    batch["subj_type"], batch["obj_type"], batch["lengths"]
                )

                # get predictions
                _, preds = torch.max(outputs, 1)

                # Collect predictions and labels
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(batch["labels"].cpu().numpy())

        # metrics
        accuracy = accuracy_score(all_labels, all_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='macro')

        print("\nTest Results:")
        print(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")

        # classification report
        print("\nClassification Report:")
        print(classification_report(
            all_labels,
            all_preds,
            labels=range(len(label_encoder.classes_)),
            target_names=label_encoder.classes_,
            zero_division=0
        ))

    return model, word_vocab, pos_vocab, ner_vocab, entity_type_vocab, label_encoder


def load_model(model_path):
    # load model using CPU
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'), weights_only=False)

    word_vocab = checkpoint["word_vocab"]
    pos_vocab = checkpoint["pos_vocab"]
    ner_vocab = checkpoint["ner_vocab"]
    entity_type_vocab = checkpoint["entity_type_vocab"]
    label_encoder = checkpoint["label_encoder"]
    max_len = checkpoint.get("max_len", 60)

    # Initialize model
    model = LSTM(
        word_vocab_size=len(word_vocab),
        pos_vocab_size=len(pos_vocab),
        ner_vocab_size=len(ner_vocab),
        entity_type_vocab_size=len(entity_type_vocab),
        num_classes=len(label_encoder.classes_),
        max_len=max_len,
        dropout_rate=0.3
    )

    model.load_state_dict(checkpoint["model_state_dict"])

    model.eval()

    return model, word_vocab, pos_vocab, ner_vocab, entity_type_vocab, label_encoder, max_len


# extract relationship
def extract_relationship(sentence, entity1, entity2, model, word_vocab, pos_vocab, ner_vocab,
                         entity_type_vocab, label_encoder, max_len=60):

    model.eval()
    device = model.word_embedding.weight.device  # Get the current device of the model

    if isinstance(sentence, str):
        tokens = sentence.split()
    else:
        tokens = sentence

    tokens = [t.lower() for t in tokens]

    entity1_tokens = entity1.lower().split()
    entity2_tokens = entity2.lower().split()

    # subject position
    subj_start = -1
    for i in range(len(tokens) - len(entity1_tokens) + 1):
        if [tokens[i + j] for j in range(len(entity1_tokens))] == entity1_tokens:
            subj_start = i
            subj_end = i + len(entity1_tokens) - 1
            break

    # object position
    obj_start = -1
    for i in range(len(tokens) - len(entity2_tokens) + 1):
        if [tokens[i + j] for j in range(len(entity2_tokens))] == entity2_tokens:
            obj_start = i
            obj_end = i + len(entity2_tokens) - 1
            break

    # if entities not found, return error
    if subj_start == -1 or obj_start == -1:
        return "Entities not found in the sentence"

    # position features
    subj_center = (subj_start + subj_end) // 2
    obj_center = (obj_start + obj_end) // 2
    subj_positions = [i - subj_center for i in range(len(tokens))]
    obj_positions = [i - obj_center for i in range(len(tokens))]

    # POS and NER placeholders
    pos_ids = [pos_vocab["<UNK>"]] * len(tokens)
    ner_ids = [ner_vocab["<UNK>"]] * len(tokens)

    # guesses for entity types based on common patterns
    subj_type = entity_type_vocab.get("PERSON", 0)  # Default to PERSON
    obj_type = entity_type_vocab.get("ORGANIZATION", 0)  # Default to ORGANIZATION

    # convert tokens to IDs
    token_ids = [word_vocab.get(t, word_vocab["<UNK>"]) for t in tokens]

    # handle sequence length
    if len(tokens) > max_len:
        token_ids = token_ids[:max_len]
        pos_ids = pos_ids[:max_len]
        ner_ids = ner_ids[:max_len]
        subj_positions = subj_positions[:max_len]
        obj_positions = obj_positions[:max_len]

    # pad sequences
    token_len = len(token_ids)
    pad_len = max_len - token_len

    token_ids += [word_vocab["<PAD>"]] * pad_len
    pos_ids += [pos_vocab["<PAD>"]] * pad_len
    ner_ids += [ner_vocab["<PAD>"]] * pad_len
    subj_positions += [0] * pad_len
    obj_positions += [0] * pad_len

    # convert to tensors and add batch dimension
    token_ids = torch.tensor([token_ids], dtype=torch.long).to(device)
    pos_ids = torch.tensor([pos_ids], dtype=torch.long).to(device)
    ner_ids = torch.tensor([ner_ids], dtype=torch.long).to(device)
    subj_positions = torch.tensor([subj_positions], dtype=torch.long).to(device)
    obj_positions = torch.tensor([obj_positions], dtype=torch.long).to(device)
    subj_type = torch.tensor([subj_type], dtype=torch.long).to(device)
    obj_type = torch.tensor([obj_type], dtype=torch.long).to(device)
    lengths = torch.tensor([token_len], dtype=torch.long).to(device)

    # prediction
    with torch.no_grad():
        outputs = model(
            token_ids, pos_ids, ner_ids,
            subj_positions, obj_positions,
            subj_type, obj_type, lengths
        )

        # get index and convert to label
        _, predicted = torch.max(outputs, 1)
        relation = label_encoder.inverse_transform(predicted.cpu().numpy())[0]

    return relation


def prediction_result(model, word_vocab, pos_vocab, ner_vocab, entity_type_vocab, label_encoder, max_len=60):
    print("Enter 'exit' to quit\n")

    device = get_device()
    model.to(device)

    while True:
        print("Enter a sentence:")
        sentence = input().strip()
        if sentence.lower() == 'exit':
            break

        print("Enter the first entity (subject):")
        entity1 = input().strip()
        if entity1.lower() == 'exit':
            break

        print("Enter the second entity (object):")
        entity2 = input().strip()
        if entity2.lower() == 'exit':
            break

        # extract relationship
        relation = extract_relationship(
            sentence, entity1, entity2,
            model, word_vocab, pos_vocab, ner_vocab,
            entity_type_vocab, label_encoder, max_len
        )
        print(f"\nPredicted relation between '{entity1}' and '{entity2}': {relation}\n")


def evaluate_with_length_analysis(model, test_data, data_loader, device, label_encoder, export_csv=False):
    model.eval()

    sample_lengths = [len(sample["token"]) for sample in test_data]

    # all predictions and labels
    predictions_with_indices = []
    labels_with_indices = []

    with torch.no_grad():
        batch_idx = 0
        for batch in data_loader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

            # forward pass
            outputs = model(
                batch["token_ids"], batch["pos_ids"], batch["ner_ids"],
                batch["subj_positions"], batch["obj_positions"],
                batch["subj_type"], batch["obj_type"], batch["lengths"]
            )

            # predictions
            _, preds = torch.max(outputs, 1)

            for i, (pred, label, orig_idx) in enumerate(zip(
                    preds.cpu().numpy(),
                    batch["labels"].cpu().numpy(),
                    batch["original_indices"]
            )):
                predictions_with_indices.append((orig_idx, pred))
                labels_with_indices.append((orig_idx, label))

    predictions_with_indices.sort(key=lambda x: x[0])
    labels_with_indices.sort(key=lambda x: x[0])

    # convert to numpy arrays
    all_preds = np.array([p[1] for p in predictions_with_indices])
    all_labels = np.array([l[1] for l in labels_with_indices])

    all_preds_unencoded = label_encoder.inverse_transform(all_preds)
    all_labels_unencoded = label_encoder.inverse_transform(all_labels)

    if export_csv:
        results_df = pd.DataFrame({
            'True Labels': all_labels_unencoded,
            'Predictions': all_preds_unencoded
        })
        results_df.to_csv('predictions.csv', index=False)

    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='macro')

    print("\nTest Results:")
    print(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")

    # Classification report
    print("\nClassification Report:")
    print(classification_report(
        all_labels,
        all_preds,
        labels=range(len(label_encoder.classes_)),
        target_names=label_encoder.classes_,
        zero_division=0
    ))


    # compute quantiles for 0.33 and 0.66 to create 3 equal groups
    lengths = np.array(sample_lengths[:len(all_preds)])
    q33 = np.quantile(lengths, 0.33)
    q66 = np.quantile(lengths, 0.66)

    # lists for each category
    short_preds = []
    short_labels = []
    mid_preds = []
    mid_labels = []
    long_preds = []
    long_labels = []

    # categorize samples based on sentence length
    for length, pred, label in zip(lengths, all_preds, all_labels):
        if length <= q33:
            short_preds.append(pred)
            short_labels.append(label)
        elif length <= q66:
            mid_preds.append(pred)
            mid_labels.append(label)
        else:
            long_preds.append(pred)
            long_labels.append(label)

    # category scores
    print("\n# category-wise scores")
    cal_category_score("Short", short_labels, short_preds)
    cal_category_score("Mid", mid_labels, mid_preds)
    cal_category_score("Long", long_labels, long_preds)


def cal_category_score(name, true_labels, pred_labels):
    if len(true_labels) > 0:
        precision = precision_recall_fscore_support(true_labels, pred_labels, average="macro", zero_division=0)[0]
        recall = precision_recall_fscore_support(true_labels, pred_labels, average="macro", zero_division=0)[1]
        f1 = precision_recall_fscore_support(true_labels, pred_labels, average="macro", zero_division=0)[2]
        print(f"{name} Sentences - Precision: {precision:.4f} | Recall: {recall:.4f} | F1-score: {f1:.4f}")
    else:
        print(f"No {name} sentences in the test set.")

if __name__ == "__main__":

    # configurations
    max_len = 60
    dropout_rate = 0.3
    balance_ratio = 3.0
    train_filepath = "../data/train.json"
    dev_filepath = "../data/dev.json"
    test_filepath = "../data/test.json"

    if os.path.exists("../models_parameters/re_lstm_model.pt"):
        print("Loading model")
        model, word_vocab, pos_vocab, ner_vocab, entity_type_vocab, label_encoder, max_len = load_model(
            "../models_parameters/re_lstm_model.pt")

        # evaluate on test set
        if os.path.exists(test_filepath):
            print("\nEvaluating on test set")
            test_data = load_data(test_filepath)
            processed_test = preprocess_data(test_data, word_vocab, pos_vocab, ner_vocab, entity_type_vocab, max_len)
            test_dataset = RelationDataset(processed_test, label_encoder)
            test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)

            device = get_device()
            model.to(device)

            evaluate_with_length_analysis(model, test_data, test_loader, device, label_encoder)
    else:
        print("Training new model - LSTM")
        # Load and process data
        train_data = load_data(train_filepath)
        dev_data = load_data(dev_filepath)

        # load test data
        test_data = None
        if os.path.exists(test_filepath):
            print("Loading test set")
            test_data = load_data(test_filepath)

        # train model
        model, word_vocab, pos_vocab, ner_vocab, entity_type_vocab, label_encoder = train_and_evaluate(
            train_data, dev_data, test_data, batch_size=32, epochs=20, max_len=max_len,
            dropout_rate=dropout_rate, balance_ratio=balance_ratio
        )

        if test_data:
            device = get_device()
            model.to(device)
            processed_test = preprocess_data(test_data, word_vocab, pos_vocab, ner_vocab, entity_type_vocab, max_len)
            test_dataset = RelationDataset(processed_test, label_encoder)
            test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)
            evaluate_with_length_analysis(model, test_data, test_loader, device, label_encoder)

    prediction_result(
        model, word_vocab, pos_vocab, ner_vocab, entity_type_vocab, label_encoder, max_len
    )