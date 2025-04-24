import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
#import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score

# 常量
AA_List = "ACDEFGHIKLMNPQRSTVWY"
PADDING = 0
aa_size = 21
embedding_dim = 64
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32
EPOCHS = 35
LEARNING_RATE =0.001

# 数据集
class ProteinDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = sequences
        self.labels = labels

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]

# 数据加载
file_path = "F:/ProteinWithFasta_WS/uniprot_soluble_proteins_with_fasta_dropped.csv"
data = pd.read_csv(file_path)
aa_dict = {aa: idx + 1 for idx, aa in enumerate(AA_List)}
data['Solubility'] = data['Solubility'].map({'Soluble': 1, 'Insoluble': 0})
data['Encoded_Seq'] = data['Clear_FASTA'].apply(lambda seq: [aa_dict.get(aa, PADDING) for aa in seq])
#下面两行用的是静态填充，现改为动态填充，便注释掉
#max_length = 5636
#data['Encoded_Seq'] = data['Encoded_Seq'].apply(lambda x: x + [PADDING] * (max_length - len(x)))

x = data['Encoded_Seq'].tolist()
y = data['Solubility'].tolist()
X_train, XX, y_train, yy = train_test_split(x, y, test_size=0.3, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(XX, yy, test_size=0.5, random_state=42, stratify=yy)

train_dataset = ProteinDataset(X_train, y_train)
val_dataset = ProteinDataset(X_val, y_val)
test_dataset = ProteinDataset(X_test, y_test)

def collate_fn(batch):
    sequences, labels = zip(*batch)
    sequences = [torch.tensor(seq, dtype=torch.long) for seq in sequences]
    sequences = torch.nn.utils.rnn.pad_sequence(sequences, batch_first=True, padding_value=PADDING)
    labels = torch.tensor(labels, dtype=torch.float)
    return sequences, labels

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

#定义transformer模型
class ProteinTransformer(nn.Module):
    def __init__(self, AA_size=aa_size + 1, emb_size = embedding_dim, nhead=8, nhid=256, nlayers=5):#nlayer=2,代表将EncoderLayer复制2次，可以改变数量来改变Encoder的层数
        #上一行 nhid为Attention后的前馈神经网络（FFN）的隐藏层的维度
        #FFN(x) = Linear(emb_size → dim_feedforward) → ReLU → Linear(dim_feedforward → emb_size)
        #通常dim_feedforward ≈ 4 × d_model
        super(ProteinTransformer, self).__init__()
        self.embedding = nn.Embedding(AA_size, emb_size,padding_idx=PADDING)
        encoder_layer = nn.TransformerEncoderLayer(d_model=emb_size, nhead=nhead ,dim_feedforward=nhid,dropout=0.1)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=nlayers)
        self.classifier = nn.Linear(emb_size, 1)


    def forward(self, x):
        '''
        下面是无AttentionMask的版本
        x = self.embedding(x)   #embedding_dim=64将每个氨基酸转换为一个64维的向量
        x = x.permute(1,0,2)    #调整三个维度的顺序
        x = self.transformer(x)
        x = x.mean(dim=0)   #在第0维取平均，把蛋白质序列压缩成一个固定长度的向量
        x = self.classifier(x)
        return torch.sigmoid(x).squeeze()   #sequeeze把维度1去掉，得到可溶性概率
        '''
        #加上Attention Mask
        mask = (x == PADDING)
        x = self.embedding(x)
        x = x.permute(1,0,2)

        out = self.transformer(x, src_key_padding_mask=mask)    #应用mask
        out = out.mean(dim=0)
        out = self.classifier(out)
        return torch.sigmoid(out).squeeze()

def train(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0
    for batch in loader:
        x, y = batch
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        output = model(x)
        y = y.float()
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()*x.size(0) #
    return total_loss / len(loader.dataset)

def evaluate(model, loader):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():   #
        for batch in loader:
            x, y = batch
            x = x.to(device)
            output = model(x)
            preds = (output > 0.5).int().cpu().numpy()  #
            all_preds.extend(preds)
            all_labels.extend(y.cpu().numpy())
    f1 = f1_score(all_labels, all_preds)
    acc = accuracy_score(all_labels, all_preds)
    #print("Preds:", np.unique(all_preds, return_counts=True))
    #print("Labels:", np.unique(all_labels, return_counts=True))
    return f1,acc

model = ProteinTransformer().to(device)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = nn.BCELoss()

for epoch in range(EPOCHS):
    train_loss = train(model, train_loader, optimizer, criterion)
    val_f1,val_acc = evaluate(model, val_loader)
    print(f"Epoch {epoch+1}/{EPOCHS},Loss:{train_loss},F1:{val_f1},Accuracy:{val_acc}")

test_f1,test_acc = evaluate(model, test_loader)
print(f"Test F1:{test_f1},Test Accuracy:{test_acc}")
