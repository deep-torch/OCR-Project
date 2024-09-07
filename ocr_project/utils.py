import torch


def get_device():
    """
    Determines the device to use for PyTorch operations. 
    Returns:
        device (torch.device): The device object representing the hardware ('cuda' or 'cpu').
    """
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_longest_word_length(annotation_file):
    max_length = 0
    with open(annotation_file, 'r') as file:
        for line in file:
            parts = line.strip().split(' ')
            label = parts[-1]
            if len(label) > max_length:
                max_length = len(label)

    return max_length


def decode(pred):
    prev_char = None
    decoded_str = []

    for idx in pred:
        if idx != prev_char:
            decoded_str.append(idx)
            prev_char = idx

    decoded_str = [idx for idx in decoded_str if idx != 0]
    return decoded_str


def translate(decoded_preds, index_to_token):
    return [''.join([index_to_token[idx] for idx in decoded_preds if idx in index_to_token])]


def calculate_cer(cer_metric, predictions, targets, index_to_token):
    assert len(predictions) == len(targets), "The number of predictions and targets must be the same."
    decoded_preds = [translate(decode(pred), index_to_token) for pred in predictions]
    decoded_targets = [translate(target, index_to_token) for target in targets]
    cer_metric.update(decoded_preds, decoded_targets)


def calculate_accuracy(predictions, targets, index_to_token):
    decoded_preds = [translate(decode(pred), index_to_token) for pred in predictions]
    decoded_targets = [translate(target, index_to_token) for target in targets]
    correct_count = sum(pred == target for pred, target in zip(decoded_preds, decoded_targets))
    accuracy = correct_count / len(decoded_preds)
    return accuracy * 100
