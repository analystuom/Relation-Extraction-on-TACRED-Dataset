import sys
import argparse
import models.lstm as lstm
import models.bert as bert
import models.lstm_relation_only as lstm_2
import models.bert_relation_only as bert_2


def svm_load_and_predict(sentence, entity1, entity2):
    import models.svm_predict as svm
    return svm.predict_relation(sentence, entity1, entity2)


def load_and_extract(sentence, entity1, entity2, model_type):
    model_processors = {
        'lstm': lambda: lstm.RelationProcessor().process_text(sentence, entity1, entity2),
        'bert': lambda: bert.predict_relation(sentence, entity1, entity2),
        'svm': lambda: svm_load_and_predict(sentence, entity1, entity2),
        'lstm-relation': lambda: lstm_2.RelationProcessor().process_text(sentence, entity1, entity2),
        'bert-relation': lambda: bert_2.predict_relation(sentence, entity1, entity2)
    }

    try:
        model_type = model_type.lower()
        processor_func = model_processors.get(model_type)
        if processor_func is None:
            raise ValueError(f"Unsupported model type: {model_type}. Choose from 'lstm', "
                             f"'bert','svm', or 'lstm-relation', 'bert-relation'.")
        return processor_func()
    except Exception as e:
        print(f"Error in load_and_extract: {e}")
        raise


def validate_inputs(sentence, entity1, entity2, model_type):
    if not sentence:
        return False, "Sentence cannot be empty."
    if not entity1:
        return False, "First entity cannot be empty."
    if not entity2:
        return False, "Second entity cannot be empty."

    valid_models = ['lstm', 'bert', 'svm', 'lstm-relation', 'bert-relation']
    if model_type.lower() not in valid_models:
        return False, f"Invalid model type: {model_type}. Please choose from {', '.join(valid_models)}."

    warnings = []
    if entity1.lower() not in sentence.lower():
        warnings.append(f"Warning: Entity 1 '{entity1}' not found in the sentence.")
    if entity2.lower() not in sentence.lower():
        warnings.append(f"Warning: Entity 2 '{entity2}' not found in the sentence.")

    return True, warnings


def predict_relation(sentence, entity1, entity2, model_type):
    result = {
        'success': False,
        'warnings': [],
        'error': None,
        'data': {
            'sentence': sentence,
            'entity1': entity1,
            'entity2': entity2,
            'model_type': model_type,
            'relation': None
        }
    }

    is_valid, message = validate_inputs(sentence, entity1, entity2, model_type)
    if not is_valid:
        result['error'] = message
        return result

    if isinstance(message, list):
        result['warnings'] = message

    try:
        relation = load_and_extract(sentence, entity1, entity2, model_type)

        result['success'] = True
        result['data']['relation'] = relation

    except Exception as e:
        result['error'] = f"Error predicting relation: {e}"

    return result


def format_result(result):
    output = []

    for warning in result.get('warnings', []):
        output.append(warning)

    if result.get('error'):
        output.append(f"Error: {result['error']}")
        return "\n".join(output)

    # Format successful result
    data = result.get('data', {})
    output.append("\nResults:")
    output.append(f"  Model: {data.get('model_type', '').upper()}")
    output.append(f"  Sentence: {data.get('sentence', '')}")
    output.append(f"  Entity 1 (Subject): {data.get('entity1', '')}")
    output.append(f"  Entity 2 (Object): {data.get('entity2', '')}")
    output.append(f"  Predicted relation: {data.get('relation', 'Unknown')}")

    return "\n".join(output)


def get_available_models():
    return ['lstm', 'bert', 'svm', 'lstm-relation', 'bert-relation']



def user_interact():
    print("\n=== Relation Prediction ===")
    print("Type 'exit' at any prompt to quit the program")

    # Get model type from user
    print("Select a model type (lstm, bert, svm, lstm-relation, bert-relation):")
    model_type = input().strip().lower()
    if model_type == 'exit':
        return

    if model_type not in ['lstm', 'bert', 'svm', 'lstm-relation', 'bert-relation']:
        print(
            f"Invalid model type: {model_type}. Please choose from 'lstm', 'bert','svm' "
            f"or 'lstm-relation', 'bert-relation'.")
        return

    # Main interaction loop
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

        try:
            # Use the new prediction function
            result = predict_relation(sentence, entity1, entity2, model_type)

            # Print formatted result
            print(format_result(result))
            print("\n" + "-" * 45 + "\n")
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    user_interact()