import torch
import torch.nn as nn
from torch.nn import functional as F

# Hyperparameters: Define training configuration
# batch_size = 64 #number of idependent sequences precessed in parallel
# block_size = 256 #maximum context length for prediction
# max_iters = 5000
# eval_interval = 500
# learning_rate = 3e-4
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# eval_iters = 200
# n_embed = 384
# n_head = 6
# n_layer = 6
# dropout = 0.2

batch_size = 128  # number of idependent sequences precessed in parallel
block_size = 512  # maximum context length for prediction
eval_interval = 100  # Interval for evaluation during training
learning_rate = 3e-4  # Learning rate for the optimizer
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200  # Number of iterations for loss estimation
n_embed = 512  # Dimensionality of token embeddings
n_head = 8  # Number of attention heads
n_layer = 8  # Number of transformer blocks
dropout = 0.2  # Dropout rate to avoid overfitting

# max_iters = 5000
# save_path = 'bigram_model.pth'
# with open('input.txt', 'r', encoding='utf-8') as f:
#     text = f.read()

# max_iters = 10000
# save_path = 'bigram_model_twitter.pth'
# with open('Twitter Datasubset.txt', 'r', encoding='utf-8') as f:
#     text = f.read()

max_iters = 2000
save_path = 'lyrics.pth'
with open('Eminem Lyrics.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# all unique characters
chars = sorted(list(set(text)))
vocab_size = len(chars)

# Mapping characters to integers (stoi: string-to-integer) and vice versa (itos: integer-to-string)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
# Encoder: Converts a string into a list of integers
encode = lambda s: [stoi[c] for c in s]
# Decoder: Converts a list of integers back into a string
decode = lambda l: ''.join([itos[i] for i in l])

# Convert the entire dataset into a tensor of integers
data = torch.tensor(encode(text), dtype=torch.long)

# split into train and test data
n = int(0.9 * (len(data)))
train_data = data[:n]
val_data = data[n:]


# Function to create input-output batches for training
def get_batch(split):
    data = train_data if split == 'train' else val_data
    # Randomly select starting indices for each batch
    ix = torch.randint(len(data) - block_size, (batch_size,))
    # Create input (x) and target (y) tensors
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    # Move tensors to the specified device (CPU or GPU)
    x, y = x.to(device), y.to(device)
    return x, y


# Function to estimate training and validation loss
@torch.no_grad()  # Disable gradient computation for performance
def estimate_loss():
    out = {}
    model.eval()  # Set the model to evaluation mode
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()  # Average loss across iterations
    model.train()  # Return the model to training mode
    return out


# Self-Attention Mechanism: Defines a single attention head
class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))  # Causal masking
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)  # Compute keys
        q = self.query(x)  # Compute queries
        # Compute scaled dot-product attention
        wei = q @ k.transpose(-2, -1) * C ** -0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))  # Apply masking
        wei = F.softmax(wei, dim=-1)  # Normalize weights
        wei = self.dropout(wei)  # Apply dropout
        v = self.value(x)  # Compute values
        out = wei @ v  # Weighted sum of values
        return out


# Multi-Head Attention: Combines multiple attention heads
class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])  # List of attention heads
        self.proj = nn.Linear(n_embed, n_embed)  # Linear projection layer
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Concatenate outputs from all attention heads and project
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        return out


# Feed-Forward Network: Applies transformations after self-attention
class FeedForward(nn.Module):
    def __init__(self, n_embed):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed),  # Expand dimensions
            nn.ReLU(),  # Apply non-linearity
            nn.Linear(4 * n_embed, n_embed),  # Reduce dimensions back
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


# Transformer Block: Combines multi-head attention and feed-forward network
class Block(nn.Module):
    def __init__(self, n_embed, n_head):
        super().__init__()
        head_size = n_embed // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embed)
        self.ln1 = nn.LayerNorm(n_embed)  # Layer normalization for attention
        self.ln2 = nn.LayerNorm(n_embed)  # Layer normalization for feed-forward

    def forward(self, x):
        x = x + self.sa(self.ln1(x))  # Add residual connection for attention
        x = x + self.ffwd(self.ln2(x))  # Add residual connection for feed-forward
        return x


# Bigram Language Model: Combines all components into a full transformer
class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)  # Token embeddings
        self.position_embedding_table = nn.Embedding(block_size, n_embed)  # Positional embeddings
        self.blocks = nn.Sequential(
            *[Block(n_embed, n_head=n_head) for _ in range(n_layer)])  # Stack transformer blocks
        self.ln_f = nn.LayerNorm(n_embed)  # Final layer normalization
        self.lm_head = nn.Linear(n_embed, vocab_size)  # Output layer for logits

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)  # Token embeddings
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))  # Positional embeddings
        x = tok_emb + pos_emb  # Combine embeddings
        x = self.blocks(x)  # Pass through transformer blocks
        x = self.ln_f(x)  # Apply final layer normalization
        logits = self.lm_head(x)  # Compute logits for each token

        if targets is None:
            loss = None  # No loss during inference
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)  # Reshape for cross-entropy loss
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)  # Compute loss
        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]  # Ensure context length doesn't exceed block_size
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :]  # Focus on the last time step
            probs = F.softmax(logits, dim=-1)  # Convert logits to probabilities
            idx_next = torch.multinomial(probs, num_samples=1)  # Sample the next token
            idx = torch.cat((idx, idx_next), dim=1)  # Append the sampled token to the context
        return idx


# Check if training or inference
train_model = False  # Set this to False for inference

if train_model:
    # Training phase
    model = BigramLanguageModel().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)  # AdamW optimizer
    for iter in range(max_iters):
        if iter % eval_interval == 0:  # Periodically estimate loss
            losses = estimate_loss()
            print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        xb, yb = get_batch('train')  # Get a batch of training data
        logits, loss = model(xb, yb)  # Forward pass
        optimizer.zero_grad()  # Clear gradients
        loss.backward()  # Backward pass
        optimizer.step()  # Update model parameters
    torch.save(model.state_dict(), save_path)  # Save the trained model
    print(f"Model saved to {save_path}")
else:
    model = BigramLanguageModel()
    # model.load_state_dict(torch.load(save_path, weights_only=True))
    model.load_state_dict(torch.load(save_path, map_location=torch.device('cpu')))
    model.to(device)
    model.eval()

    context = torch.zeros((1, 1), dtype=torch.long, device=device)  # Start with an empty context

    # Generate text and print one character at a time
    for _ in range(1000):  # Generate 1000 characters
        # Get the logits for the current context
        logits, _ = model(context[:, -block_size:])
        logits = logits[:, -1, :]  # Focus on the last time step
        probs = F.softmax(logits, dim=-1)  # Convert logits to probabilities
        next_idx = torch.multinomial(probs, num_samples=1)  # Sample next character
        context = torch.cat((context, next_idx), dim=1)  # Append to the context

        # Decode and print the latest character
        print(decode([next_idx.item()]), end='', flush=True)

