import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence



# Get device for running model, load LSTM model, process input
class RelationProcessor:
    def __init__(self, model_path="./models_parameters/re_lstm_model_no_relation_filtered.pt"):
        self.device = self._get_device()
        (self.model, self.word_vocab, self.pos_vocab, self.ner_vocab, self.entity_type_vocab,
         self.label_encoder, self.max_len) = self._initialize_model(model_path)

    def _get_device(self):
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
            print("Using Apple Silicon GPU")
        else:
            device = torch.device("cpu")
            print("Using CPU")

        return device

    def _initialize_model(self, model_path):
        print("Loading Model - LSTM")

        # load the model on CPU first
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'), weights_only=False)

        word_vocab = checkpoint['word_vocab']
        pos_vocab = checkpoint['pos_vocab']
        ner_vocab = checkpoint['ner_vocab']
        entity_type_vocab = checkpoint['entity_type_vocab']
        label_encoder = checkpoint['label_encoder']
        max_len = checkpoint['max_len']

        # initialize model
        model = LSTM(
            word_vocab_size=len(word_vocab),
            pos_vocab_size=len(pos_vocab),
            ner_vocab_size=len(ner_vocab),
            entity_type_vocab_size=len(entity_type_vocab),
            num_classes=len(label_encoder.classes_),
            max_len=max_len
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        # print(model)
        return model, word_vocab, pos_vocab, ner_vocab, entity_type_vocab, label_encoder, max_len

    def process_text(self, sentence, entity1, entity2):
        self.model.eval()

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

        # extract position features - how words are positioned relatives to entities
        subj_center = (subj_start + subj_end) // 2
        obj_center = (obj_start + obj_end) // 2
        subj_positions = [i - subj_center for i in range(len(tokens))]
        obj_positions = [i - obj_center for i in range(len(tokens))]

        # generate placeholders for POS and NER tags
        pos_ids = [self.pos_vocab["<UNK>"]] * len(tokens)
        ner_ids = [self.ner_vocab["<UNK>"]] * len(tokens)

        # guess entity types of input entities
        subj_type = self.entity_type_vocab.get("PERSON", 0)
        obj_type = self.entity_type_vocab.get("ORGANIZATION", 0)

        token_ids = [self.word_vocab.get(t, self.word_vocab["<UNK>"]) for t in tokens]

        # Handle sequence length if > 60
        if len(tokens) > self.max_len:
            token_ids = token_ids[:self.max_len]
            pos_ids = pos_ids[:self.max_len]
            ner_ids = ner_ids[:self.max_len]
            subj_positions = subj_positions[:self.max_len]
            obj_positions = obj_positions[:self.max_len]

        # pad sequences if < 60
        token_len = len(token_ids)
        pad_len = self.max_len - token_len

        token_ids += [self.word_vocab["<PAD>"]] * pad_len
        pos_ids += [self.pos_vocab["<PAD>"]] * pad_len
        ner_ids += [self.ner_vocab["<PAD>"]] * pad_len
        subj_positions += [0] * pad_len
        obj_positions += [0] * pad_len

        # convert to tensors and add batch dimension
        token_ids = torch.tensor([token_ids], dtype=torch.long).to(self.device)
        pos_ids = torch.tensor([pos_ids], dtype=torch.long).to(self.device)
        ner_ids = torch.tensor([ner_ids], dtype=torch.long).to(self.device)
        subj_positions = torch.tensor([subj_positions], dtype=torch.long).to(self.device)
        obj_positions = torch.tensor([obj_positions], dtype=torch.long).to(self.device)
        subj_type = torch.tensor([subj_type], dtype=torch.long).to(self.device)
        obj_type = torch.tensor([obj_type], dtype=torch.long).to(self.device)
        lengths = torch.tensor([token_len], dtype=torch.long).to(self.device)

        # predict the relationship
        with torch.no_grad():
            outputs = self.model(
                token_ids, pos_ids, ner_ids,
                subj_positions, obj_positions,
                subj_type, obj_type, lengths
            )

            _, predicted = torch.max(outputs, 1)
            relation = self.label_encoder.inverse_transform(predicted.cpu().numpy())[0]

        return relation


# re-defined the architecture to make it independent of the training script
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

        # Fully connected layers
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

        # pack padded sequence for efficiency
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

        output = self.fc1(combined)
        output = self.batch_norm(output)
        output = self.relu(output)
        output = self.dropout(output)
        output = self.fc2(output)

        return output


# if __name__ == "__main__":
#     processor = RelationProcessor()
#     relation = processor.process_text(
#         "Trump loves America",
#         "Trump",
#         "America"
#     )
#     print(f"Predicted relation: {relation}")
