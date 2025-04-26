import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR
from transformers import BertModel, BertTokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
id_to_label ={0: 'per:religion',
 1: 'per:stateorprovinces_of_residence',
 2: 'per:stateorprovince_of_death',
 3: 'per:other_family',
 4: 'per:country_of_birth',
 5: 'per:cities_of_residence',
 6: 'org:founded',
 7: 'per:city_of_death',
 8: 'per:charges',
 9: 'per:stateorprovince_of_birth',
 10: 'org:members',
 11: 'per:alternate_names',
 12: 'org:parents',
 13: 'per:city_of_birth',
 14: 'org:country_of_headquarters',
 15: 'org:alternate_names',
 16: 'org:subsidiaries',
 17: 'org:member_of',
 18: 'per:spouse',
 19: 'per:date_of_birth',
 20: 'org:political/religious_affiliation',
 21: 'per:age',
 22: 'org:founded_by',
 23: 'per:date_of_death',
 24: 'per:children',
 25: 'per:siblings',
 26: 'per:title',
 27: 'org:city_of_headquarters',
 28: 'org:number_of_employees/members',
 29: 'per:parents',
 30: 'per:country_of_death',
 31: 'per:origin',
 32: 'org:shareholders',
 33: 'org:stateorprovince_of_headquarters',
 34: 'no_relation',
 35: 'per:schools_attended',
 36: 'org:website',
 37: 'org:dissolved',
 38: 'per:cause_of_death',
 39: 'per:employee_of',
 40: 'org:top_members/employees',
 41: 'per:countries_of_residence'}
# ================== Define BERT + FFNN Model ==================
# ================== Define BERT + FFNN Model ==================
class BertWithFFNN(nn.Module):
    def __init__(self, num_labels):
        super(BertWithFFNN, self).__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.extra_layer = nn.TransformerEncoderLayer(d_model=768, nhead=12, dim_feedforward=3072)
        self.dropout = nn.Dropout(0.2)
        self.fc1 = nn.Linear(768 * 2, 256)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256, num_labels)

    def forward(self, input_ids, attention_mask, entity1_idx, entity2_idx):
        bert_outputs = self.bert(input_ids, attention_mask=attention_mask)
        hidden_states = bert_outputs.last_hidden_state
        extra_layer_output = self.extra_layer(hidden_states)

        batch_size = extra_layer_output.shape[0]
        entity1_output = extra_layer_output[torch.arange(batch_size), entity1_idx]
        entity2_output = extra_layer_output[torch.arange(batch_size), entity2_idx]

        combined_representation = torch.cat([entity1_output, entity2_output], dim=-1)
        x = self.dropout(combined_representation)
        x = self.fc1(x)
        x = self.relu(x)
        logits = self.fc2(x)

        return logits
# ======= HERE WE PRE PROCESS AND EVALUTE INPUT FOR MODEL =======

# ================== Data Processing ==================
def process_data(data):
    processed = []
    for item in data:
        sentence = " ".join(item['token'])
        subject = " ".join(item['token'][item['subj_start']:item['subj_end'] + 1])
        obj = " ".join(item['token'][item['obj_start']:item['obj_end'] + 1])
        processed.append({
            'text': sentence,
            'relation': item['relation'],
            'subject': subject,
            'object': obj
        })
    return processed

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")
def predict_relation(sentence, subject, obj):
    """
    Predicts the relation between subject and object in the given sentence.
    """
    # Ensure model is on the correct device
    model = BertWithFFNN(41)
    model.load_state_dict(torch.load('./models_parameters/bertmodelnorelation.pt',
                                     map_location=torch.device("cpu")))
    model.eval()
    print("MODEL LOADED SUCCESSFULLY")
    model.to(device)

    # Simulate the structure of a data sample
    words = sentence.split()
    subj_start = words.index(subject.split()[0])
    subj_end = words.index(subject.split()[-1])
    obj_start = words.index(obj.split()[0])
    obj_end = words.index(obj.split()[-1])

    # Simulate structured input data
    sample_data = [{
        'token': words,
        'relation': "unknown",  # Placeholder, not used in inference
        'subj_start': subj_start,
        'subj_end': subj_end,
        'obj_start': obj_start,
        'obj_end': obj_end
    }]
    # Process input to match training data format
    processed_sample = process_data(sample_data)[0]

    # Tokenize using the same encoding method as training data
    encoded = tokenizer(
        processed_sample["text"],
        padding="max_length",
        truncation=True,
        max_length=64,
        return_tensors="pt"
    )

    # Move tensors to the correct device
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    entity1_idx = torch.tensor([[subj_start, subj_end]], dtype=torch.long).to(device)
    entity2_idx = torch.tensor([[obj_start, obj_end]], dtype=torch.long).to(device)
    # Predict relation
    with torch.no_grad():
        logits = model(input_ids, attention_mask=attention_mask, entity1_idx = entity1_idx, entity2_idx=entity2_idx)
        logits_for_subject = logits[0, subj_start:subj_end + 1, :]
        predicted_label_id = torch.argmax(logits_for_subject, dim=-1).item()

    # Convert label ID to actual relation name
    predicted_relation = id_to_label.get(predicted_label_id)

    return predicted_relation


if __name__ == "__main__":
    # Example Usage
    input_sentence = input("Enter a sentence: ")
    subject_entity = input("Enter first entity: ")
    object_entity = input("Enter second entity: ")

    predicted_relation = predict_relation(input_sentence, subject_entity, object_entity)
    print(f"Predicted Relation: {predicted_relation}")
