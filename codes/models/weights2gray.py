import torch

# Path to your pretrained .pth file
ckpt_path = "/mnt/home_liu/degradation/method_4/Deflow-grayscale-main/codes/models/RRDB_PSNR_x4.pth"

# Load the checkpoint
ckpt = torch.load(ckpt_path, map_location="cpu")

# Access the state_dict (some .pth files store directly, some under a key)
state_dict = ckpt.get('state_dict', ckpt)

# Fix the first and last convolution layers for grayscale
def convert_rgb_to_gray_weights(w_rgb):
    # Average across the color channels to simulate grayscale
    return w_rgb.mean(dim=1, keepdim=True)

# conv_first.weight: [64, 3, 3, 3] → [64, 1, 3, 3]
if "conv_first.weight" in state_dict:
    w = state_dict["conv_first.weight"]
    state_dict["conv_first.weight"] = convert_rgb_to_gray_weights(w)

# conv_last.weight: [3, 64, 3, 3] → [1, 64, 3, 3]
if "conv_last.weight" in state_dict:
    w = state_dict["conv_last.weight"]
    state_dict["conv_last.weight"] = w.mean(dim=0, keepdim=True)

# conv_last.bias: [3] → [1]
if "conv_last.bias" in state_dict:
    b = state_dict["conv_last.bias"]
    state_dict["conv_last.bias"] = b.mean().unsqueeze(0)

# Save the modified checkpoint
torch.save(state_dict, "model_grayscale.pth")

print("✅ Saved modified grayscale-compatible checkpoint: model_grayscale.pth")
