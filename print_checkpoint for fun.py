import argparse
import torch

def inspect_checkpoint(path, device='cpu'):
    ckpt = torch.load(path, map_location=device, weights_only=False)

    print(f"\nLoaded checkpoint from {path}\n" + "-"*60)
    for key, value in ckpt.items():
        print(f"{key!r}: ", end="")
        if isinstance(value, (int, float, bool, str)):
            print(value)
        elif isinstance(value, dict):
            print(f"<dict with {len(value)} keys>")
        else:
            cls = value.__class__.__name__
            if hasattr(value, 'state_dict'):
                sd = value.state_dict()
                print(f"<{cls} with {len(sd)} state_dict entries>")
            else:
                print(f"<{cls} object>")

    print("-"*60 + "\n")

if __name__ == "__main__":

    checkpoint = "/home/mihai/workspace/output_data/Checkpoints/BEST_checkpoint_flickr30k_5_cap_per_img_5_min_word_freq.pth.tar"
    #checkpoint = "/home/mihai/workspace/output_data/Checkpoints/checkpoint_flickr30k_5_cap_per_img_5_min_word_freq.pth.tar"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    inspect_checkpoint(checkpoint, device)