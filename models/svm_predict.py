import json
import numpy as np
import nltk
import re
import string
from nltk.tokenize import word_tokenize
import pickle
import ssl
ssl._create_default_https_context = ssl.create_default_context


nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt_tab')

def preprocess_text(text):
    stop_words = set(nltk.corpus.stopwords.words('english'))
    text = text.lower()
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
    words = word_tokenize(text)  # Tokenize words
    words = [word for word in words if word not in stop_words]  # Remove stopwords
    return " ".join(words)


# Load models
with open('./models_parameters/svm_binary_model.pickle', 'rb') as file:
    svm_binary = pickle.load(file)

with open('./models_parameters/svm_multi_model.pickle', 'rb') as file:
    svm_multi = pickle.load(file)

# Load vectorizers
with open('./models_parameters/svm_vectorizer_text.pickle', 'rb') as file:
    vectorizer_text = pickle.load(file)

with open('./models_parameters/svm_vectorizer_pos.pickle', 'rb') as file:
    vectorizer_pos = pickle.load(file)

with open('./models_parameters/svm_vectorizer_dep.pickle', 'rb') as file:
    vectorizer_dep = pickle.load(file)

with open('./models_parameters/svm_vectorizer_ent.pickle', 'rb') as file:
    vectorizer_ent = pickle.load(file)


def predict_relation(user_text, entity1, entity2):
    processed_input = preprocess_text(user_text)
    input_vector_text = vectorizer_text.transform([processed_input])
    input_vector_pos = vectorizer_pos.transform(["unknown"])
    input_vector_dep = vectorizer_dep.transform(["unknown"])
    input_vector_ent = vectorizer_ent.transform(["unknown unknown"])

    input_vector_combined = np.hstack(
        [input_vector_text.toarray(), input_vector_pos.toarray(), input_vector_dep.toarray(),
         input_vector_ent.toarray()])
    binary_prediction = svm_binary.predict(input_vector_combined)[0]
    # binary_prediction = svm_multi.predict(input_vector_combined)[0]

    if binary_prediction == "no_relation":
        return "No relation detected"

    multi_prediction = svm_multi.predict(input_vector_combined)[0]
    return multi_prediction


if __name__ == "__main__":
    # User input for sentence and entities
    user_sentence = input("Please input a sentence: ")
    entity1 = input("Please input the first entity: ")
    entity2 = input("Please input the second entity: ")

    predicted_relation = predict_relation(user_sentence, entity1, entity2)
    print("Predicted Relation:", predicted_relation)
