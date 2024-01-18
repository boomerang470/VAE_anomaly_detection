from typing import Tuple
import torch
from torch.utils.data import Dataset, TensorDataset
from torchvision.datasets import MNIST
from transformers import BertTokenizer, BertModel
from tqdm import tqdm


def read_logs_from_file(file_path, num_lines=5000):
    with open(file_path, 'r', encoding='utf-8') as file:
        logs = [next(file).strip() for _ in range(num_lines)]
    return logs

def preprocess_logs(logs):
    # Tokenize the logs
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    tokenized_logs = [tokenizer(log, return_tensors='pt') for log in tqdm(logs, desc="Tokenizing")]

    return tokenized_logs

def convert_logs_to_vectors(tokenized_logs):
    # Load pre-trained BERT model
    model = BertModel.from_pretrained('bert-base-uncased')

    # Set the model to evaluation mode
    model.eval()

    # Convert logs to semantic vectors
    semantic_vectors = []
    with torch.no_grad():
        for log in tqdm(tokenized_logs, desc="Converting to Vectors"):
            outputs = model(**log)
            pooled_output = outputs.pooler_output
            semantic_vectors.append(pooled_output)

    # Stack the vectors into a tensor
    semantic_vectors = torch.stack(semantic_vectors)

    return semantic_vectors

def rand_dataset(num_rows=60_000, num_columns=100) -> Dataset:
    log_file_path = 'HDFS.log'
    logs = read_logs_from_file(log_file_path)
    tokenized_logs = preprocess_logs(logs)
    semantic_vectors = convert_logs_to_vectors(tokenized_logs)
    semantic_vectors = torch.stack([torch.as_tensor(output.squeeze()) for output in semantic_vectors])

# Save the tensor to a file
    torch.save(semantic_vectors, 'saved_tensor.pth')
    loaded_tensor = torch.load('saved_tensor.pth')
    return loaded_tensor


def mnist_dataset(train=True) -> Dataset:
    """
    Returns the MNIST dataset for training or testing.
    
    Args:
    train (bool): If True, returns the training dataset. Otherwise, returns the testing dataset.
    
    Returns:
    Dataset: The MNIST dataset.
    """
    return MNIST(root='./data', train=train, download=True, transform=None)
