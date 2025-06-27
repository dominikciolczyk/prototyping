import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
block = torch.nn.Conv1d(
    in_channels=256,
    out_channels=256,
    kernel_size=5,
    stride=1,
    padding="same",
    dilation=1,
    groups=1,
    bias=True,
    padding_mode="zeros",
    device=device,
    dtype=torch.float32,
)

# input shape: (batch, in_channels, length)
input = torch.randn((1, 256, 16), device=device, dtype=torch.float32)
output = block(input)
print("Output shape:", output.shape)