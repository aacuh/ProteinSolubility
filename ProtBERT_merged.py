import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score
from transformers import BertModel, BertTokenizer

# ======== 设置参数 ========
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 8  # ProtBERT 占显存大，建议小一点
EPOCHS = 10
LEARNING_RATE = 1e-4

# ======== 加载 ProtBERT 模型 ========
print("Loading ProtBERT...")
tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False)
bert_model = BertModel.from_pretrained("Rostlab/prot_bert").to(device)
bert_model.eval()

@torch.no_grad()
def protbert_encode(seq):
    seq = ' '.join(list(seq))  # MKT -> M K T
    tokens = tokenizer(seq, return_tensors="pt")
    tokens = {k: v.to(device) for k, v in tokens.items()}
    outputs = bert_model(**tokens)
    embedding = outputs.last_hidden_state.squeeze(0)  # [L, 1024]
    return embedding

# ======== 定义 Dataset 类 ========
class ProteinDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = sequences
        self.labels = labels

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        label = self.labels[idx]
        embedding = protbert_encode(seq)
        return embedding, label

# ======== 读取并准备数据 ========
file_path = "F:/ProteinWithFasta_WS/uniprot_soluble_proteins_with_fasta_dropped.csv"
data = pd.read_csv(file_path)
data['Solubility'] = data['Solubility'].map({'Soluble': 1, 'Insoluble': 0})

# 用 Clear_FASTA（原始序列）
x = data['Clear_FASTA'].tolist()
y = data['Solubility'].tolist()

X_train, XX, y_train, yy = train_test_split(x, y, test_size=0.3, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(XX, yy, test_size=0.5, random_state=42, stratify=yy)

train_dataset = ProteinDataset(X_train, y_train)
val_dataset = ProteinDataset(X_val, y_val)
test_dataset = ProteinDataset(X_test, y_test)

# ======== 自定义 collate_fn ========
def collate_fn_bert(batch):
    sequences, labels = zip(*batch)
    sequences = torch.nn.utils.rnn.pad_sequence(sequences, batch_first=True)  # [B, L, 1024]
    labels = torch.tensor(labels, dtype=torch.float)
    return sequences.to(device), labels.to(device)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn_bert)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn_bert)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn_bert)

# ======== Transformer 模型（接受 ProtBERT embedding） ========
class ProteinTransformer(nn.Module):
    def __init__(self, emb_size=1024, nhead=8, nhid=512, nlayers=3):
        super(ProteinTransformer, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=emb_size, nhead=nhead, dim_feedforward=nhid)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=nlayers)
        self.classifier = nn.Linear(emb_size, 1)

    def forward(self, x):  # [B, L, 1024]
        x = x.permute(1, 0, 2)  # [L, B, 1024]
        out = self.transformer(x)
        out = out.mean(dim=0)  # [B, 1024]
        out = self.classifier(out)
        return torch.sigmoid(out).squeeze()

# ======== 训练和评估函数 ========
def train(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0
    for x, y in loader:
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
    return total_loss / len(loader.dataset)

def evaluate(model, loader):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x, y in loader:
            output = model(x)
            preds = (output > 0.5).int().cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(y.cpu().numpy())
    f1 = f1_score(all_labels, all_preds)
    acc = accuracy_score(all_labels, all_preds)
    return f1, acc

# ======== 初始化模型并训练 ========
model = ProteinTransformer().to(device)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = nn.BCELoss()

for epoch in range(EPOCHS):
    train_loss = train(model, train_loader, optimizer, criterion)
    val_f1, val_acc = evaluate(model, val_loader)
    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {train_loss:.4f}, F1: {val_f1:.4f}, Acc: {val_acc:.4f}")

test_f1, test_acc = evaluate(model, test_loader)
print(f"\n✅ Test F1: {test_f1:.4f}, Test Accuracy: {test_acc:.4f}")
