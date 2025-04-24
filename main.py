import torch
import torch.nn as nn
import torch.nn.functional as F

# 常量
AA_List = "ACDEFGHIKLMNPQRSTVWY"
PADDING = 0
embedding_dim = 64
aa_size = 21
max_length = 5636

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 氨基酸字典
aa_dict = {aa: idx + 1 for idx, aa in enumerate(AA_List)}

def encode_sequence(seq):
    encoded = [aa_dict.get(aa, PADDING) for aa in seq]
    if len(encoded) > max_length:
        encoded = encoded[:max_length]
    else:
        encoded += [PADDING] * (max_length - len(encoded))
    return torch.tensor(encoded, dtype=torch.long).unsqueeze(0).to(device)

# 模型定义
class ProteinCNN(nn.Module):
    def __init__(self, embedding_dim, num_classes):
        super(ProteinCNN, self).__init__()
        self.embedding = nn.Embedding(aa_size, embedding_dim, padding_idx=PADDING)
        self.conv1 = nn.Conv1d(embedding_dim, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        self.conv4 = nn.Conv1d(256, 512, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(512 * 1409, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.embedding(x).permute(0, 2, 1)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool1(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool2(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 加载模型
model = ProteinCNN(embedding_dim, 2).to(device)
model.load_state_dict(torch.load("best_model.pth", weights_only=True))
model.eval()

# 预测函数
def predict_solubility(sequence):
    with torch.no_grad():
        encoded_seq = encode_sequence(sequence)
        output = model(encoded_seq)
        _, predicted = output.max(1)
        return "Soluble" if predicted.item() == 1 else "Insoluble"

# 用户交互
if __name__ == "__main__":
    seq = input("请输入蛋白质氨基酸序列: ")
    result = predict_solubility(seq)
    print(f"预测水溶性: {result}")
