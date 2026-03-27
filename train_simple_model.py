import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from load_eth_ucy_data import ETHUCYDataset

class SimpleLSTM(nn.Module):
    """Simple LSTM model for trajectory prediction"""
    
    def __init__(self, input_dim=2, hidden_dim=64, output_dim=2, 
                 obs_len=8, pred_len=12, num_layers=2):
        super().__init__()
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.hidden_dim = hidden_dim
        
        self.encoder = nn.LSTM(input_dim, hidden_dim, num_layers, 
                               batch_first=True, dropout=0.2)
        self.decoder = nn.LSTM(input_dim, hidden_dim, num_layers,
                               batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, obs):
        # obs: [batch, obs_len, 2]
        batch_size = obs.size(0)
        
        # Encode observed trajectory
        _, (hidden, cell) = self.encoder(obs)
        
        # Initialize decoder input with last observed position
        decoder_input = obs[:, -1:, :]  # [batch, 1, 2]
        
        outputs = []
        for _ in range(self.pred_len):
            # Decode one step
            out, (hidden, cell) = self.decoder(decoder_input, (hidden, cell))
            pred = self.fc(out)  # [batch, 1, 2]
            outputs.append(pred)
            decoder_input = pred
        
        # Concatenate all predictions
        return torch.cat(outputs, dim=1)  # [batch, pred_len, 2]

# ============================================
# TRAINING SETUP
# ============================================
print("=" * 50)
print("TRAINING SIMPLE TRAJECTORY PREDICTOR")
print("=" * 50)

# Load data
train_dataset = ETHUCYDataset(split='train')
val_dataset = ETHUCYDataset(split='val')

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Create model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SimpleLSTM().to(device)
print(f"\n✅ Model created on {device}")

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 5
print("\nStarting training...")

for epoch in range(num_epochs):
    # Training
    model.train()
    train_loss = 0
    for batch_idx, (obs, fut) in enumerate(train_loader):
        obs, fut = obs.to(device), fut.to(device)
        
        # Forward pass
        predictions = model(obs)
        loss = criterion(predictions, fut)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        
        if batch_idx % 10 == 0:
            print(f"  Epoch {epoch+1}, Batch {batch_idx}: Loss = {loss.item():.4f}")
    
    avg_train_loss = train_loss / len(train_loader)
    
    # Validation
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for obs, fut in val_loader:
            obs, fut = obs.to(device), fut.to(device)
            predictions = model(obs)
            loss = criterion(predictions, fut)
            val_loss += loss.item()
    
    avg_val_loss = val_loss / len(val_loader)
    
    print(f"\n📊 Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}")
    print("-" * 50)

print("\n✅ Training complete!")
print("\n🎉 Your ETH/UCY dataset is ready and working with PyTorch!")