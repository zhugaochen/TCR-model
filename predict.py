import numpy as np
from data_preprocess import load_data
import torch
from model import Tudui

# 定义训练的设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data, labels = load_data('../t2')
# labels = torch.tensor(labels, dtype=torch.float32)
data_tensor = np.array(data)
data_tensor = torch.tensor(data_tensor, dtype=torch.float32)
print(data_tensor.shape)
size = data_tensor.size(0)

labels = np.array(labels)
labels = labels.reshape(-1, 1)
labels = torch.tensor(labels, dtype=torch.float32)

model = Tudui().to(device)
model_path = 'model_fold_0_best.pth'
model.load_state_dict(torch.load(model_path))

data_tensor = data_tensor.to(device)
model.eval()
with torch.no_grad():
    predictions = model(data_tensor)
    outputs = torch.where(predictions < 0.5, torch.tensor(0.0).to(device), torch.tensor(1.0).to(device))
    targets = labels.to(device)
    accuracy = (outputs == targets).sum()
    print(f'Test set size: {size}; Predicted correctly: {accuracy}')
    print(f'Accuracy on test set: {accuracy/size}')
    print(predictions)
    print(targets)
