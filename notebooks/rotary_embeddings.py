import torch
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

# Define the Rotary class
class Rotary(torch.nn.Module):

    def __init__(self, dim, base=10000):
        super().__init__()
        self.inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.seq_len_cached = None
        self.cos_cached = None
        self.sin_cached = None

    def forward(self, x):
        seq_len = x.shape[1]
        if seq_len != self.seq_len_cached:
            self.seq_len_cached = seq_len
            t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
            freqs = torch.outer(t, self.inv_freq).to(x.device)
            self.cos_cached = freqs.cos().float()
            self.sin_cached = freqs.sin().float()
        return self.cos_cached[None, :, None, :], self.sin_cached[None, :, None, :]

# Define the apply_rotary_emb function
def apply_rotary_emb(x, cos, sin):
    assert x.ndim == 4  # multi-head attention input
    d = x.shape[3] // 2
    x1 = x[..., :d]
    x2 = x[..., d:]
    # Corrected rotation equations
    y1 = x1 * cos - x2 * sin
    y2 = x1 * sin + x2 * cos
    return torch.cat([y1, y2], dim=-1).type_as(x)

# Generate sample data
batch_size = 1
seq_len = 10
num_heads = 1
dim = 4  # Using 4D vectors because rotary embeddings work on pairs of dimensions

# Create random 4D vectors (we'll visualize only the first 3 dimensions)
x = torch.randn(batch_size, seq_len, num_heads, dim)

# Initialize Rotary embedding
rotary = Rotary(dim=dim)

# Get cosine and sine values
cos, sin = rotary(x)

# Apply rotary embedding
x_rotated = apply_rotary_emb(x, cos, sin)

# Prepare data for plotting
x_original = x.squeeze(0).squeeze(1).numpy()  # Shape: [seq_len, dim]
x_rotated = x_rotated.squeeze(0).squeeze(1).detach().numpy()  # Shape: [seq_len, dim]

# Plotting
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')

# Plot original vectors (blue)
for i in range(seq_len):
    ax.quiver(0, 0, 0, x_original[i, 0], x_original[i, 1], x_original[i, 2], 
              color='blue', alpha=0.6, arrow_length_ratio=0.1)

# Plot rotated vectors (red)
for i in range(seq_len):
    ax.quiver(0, 0, 0, x_rotated[i, 0], x_rotated[i, 1], x_rotated[i, 2], 
              color='red', alpha=0.6, arrow_length_ratio=0.1)

# Set plot limits and labels
max_val = max(np.abs(x_original).max(), np.abs(x_rotated).max()) + 0.5
ax.set_xlim([-max_val, max_val])
ax.set_ylim([-max_val, max_val])
ax.set_zlim([-max_val, max_val])

# Labels and title
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('3D Visualization of Original (Blue) and Rotated (Red) Vectors')

# Show plot
plt.show()
