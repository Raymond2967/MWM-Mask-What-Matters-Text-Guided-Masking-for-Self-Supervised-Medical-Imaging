from transformers import AutoModel, AutoTokenizer, AutoProcessor
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


model = AutoModel.from_pretrained(model_path, trust_remote_code=True).to(device)
processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

print(model.__class__)
print(tokenizer.__class__)
print(processor.__class__)
