import json

from tqdm.auto import trange
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


class RQAutoEncoder(nn.Module):
    def __init__(self, input_dim=768, latent_dim=32, output_dim=768, num_codes=256):
        super().__init__()
        self.k_dim = num_codes
        self.z_dim = latent_dim
        self.input_dim = input_dim
        self.encode = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim),
            nn.ReLU(),
        )
        # rq_vae
        self.cookbook1 = nn.Embedding(self.k_dim, self.z_dim)
        self.cookbook2 = nn.Embedding(self.k_dim, self.z_dim)
        self.cookbook3 = nn.Embedding(self.k_dim, self.z_dim)

        self.decode = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim)
        )

    def find_nearest(self, query, target):
        Q = query.unsqueeze(1).repeat(1, target.size(0), 1)
        T = target.unsqueeze(0).repeat(query.size(0), 1, 1)
        index = (Q - T).pow(2).sum(2).sqrt().min(1)[1]
        return target[index], index

    def forward(self, X):
        Z_enc = self.encode(X.view(-1, self.input_dim))
        Z_dec1, index1 = self.find_nearest(Z_enc, self.cookbook1.weight)
        z_res1 = Z_enc - Z_dec1
        z_dec2, index2 = self.find_nearest(z_res1, self.cookbook2.weight)
        z_res2 = z_res1 - z_dec2
        z_dec3, index3 = self.find_nearest(z_res2, self.cookbook2.weight)
        z_res3 = z_res2 - z_dec3

        X_recon = self.decode(z_res3).view(-1,self.z_dim)
        return Z_enc, Z_dec1, z_res1, z_dec2, z_res2, z_dec3, z_res3, X_recon, index1, index2, index3


class EmbeddingDataset(Dataset):
    def __init__(self, embedding_tensor):
        self.embedding_tensor = embedding_tensor

    def __len__(self):
        return len(self.embedding_tensor)

    def __getitem__(self, idx):
        return self.embedding_tensor[idx]

def train(model, train_loader, opt, beta, train_iterations=1000):
    def iterate_dataset(data_loader):
        data_iter = iter(data_loader)
        while True:
            try:
                x, y = next(data_iter)
            except StopIteration:
                data_iter = iter(data_loader)
                x, y = next(data_iter)
            yield x.to(device), y.to(device)

    for _ in (pbar := trange(train_iterations)):
        opt.zero_grad()
        x, _ = next(iterate_dataset(train_loader))
        out, indices, cmt_loss = model(x)
        rec_loss = (out - x).abs().mean()
        (rec_loss + beta * cmt_loss).backward()

        opt.step()
        pbar.set_description(
            f"rec loss: {rec_loss.item():.3f} | "
            + f"cmt loss: {cmt_loss.item():.3f}"
        )
    return


if __name__ == "__main__":
    with open('datasets/asin_to_embedding.json', 'r') as infile:
        asin_to_embedding = json.load(infile)

    # 假设数据是一个字典，asin为key，embedding为value
    embeddings = [value for key, value in asin_to_embedding.items()]
    embeddings_tensor = torch.tensor(embeddings)
    train_dataset = EmbeddingDataset(embeddings_tensor)

    lr = 0.4
    batch_size = 1024
    train_iter = 20000
    seed = 1234
    beta = 0.25
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.random.manual_seed(seed)
    model = RQAutoEncoder().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    train(model, train_dataset, opt, train_iterations=train_iter, beta=beta)