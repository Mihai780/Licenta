import torch
import torch.nn.functional as F
import numpy as np
import torchvision.transforms as T
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import skimage.transform as skt
import imageio
from skimage.transform import resize as resize_image
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def beam_search_captioning(cnn_encoder, lstm_decoder, img_path, mapping, beam_sz=3):
    """
    Generate a caption and attention weights for an image via beam search.
    Returns token indices and per-step attention maps.
    """
    k = beam_sz
    vocab_size = len(mapping)

    # Load & preprocess image
    raw = imageio.imread(img_path)
    if raw.ndim == 2:
        raw = np.stack([raw]*3, axis=2)
    raw = resize_image(raw, (256, 256), preserve_range=True).astype(np.uint8)
    raw = raw.transpose(2, 0, 1) / 255.0
    img_tensor = torch.FloatTensor(raw).to(device)
    norm = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    img_norm = norm(img_tensor)
    img_batch = img_norm.unsqueeze(0)

    # Encode
    feats = cnn_encoder(img_batch)
    H, W, D = feats.size(1), feats.size(2), feats.size(3)
    feats = feats.view(1, -1, D)
    num_pixels = feats.size(1)
    feats = feats.expand(k, num_pixels, D)

    # Init beams
    start_idx = mapping['<start>']
    beams = torch.full((k, 1), start_idx, dtype=torch.long, device=device)
    beam_scores = torch.zeros(k, 1, device=device)
    beam_alphas = torch.ones(k, 1, H, W, device=device)

    completed_seqs = []
    completed_alphas = []
    completed_scores = []

    h, c = lstm_decoder.initial_states(feats)
    step = 0

    while True:
        emb = lstm_decoder.embedding(beams[:, -1]).squeeze(1)
        context, alpha = lstm_decoder.attention(feats, h)
        alpha_maps = alpha.view(-1, H, W)
        gate = torch.sigmoid(lstm_decoder.beta_gate(h))
        attended = gate * context

        h, c = lstm_decoder.decode_step(torch.cat([emb, attended], dim=1), (h, c))
        logits = lstm_decoder.fcl(h)
        log_probs = F.log_softmax(logits, dim=1)

        scores = beam_scores.expand_as(log_probs) + log_probs
        if step == 0:
            top_scores, top_words = scores[0].topk(k, dim=0)
        else:
            top_scores, top_words = scores.view(-1).topk(k, dim=0)

        prev_beam = top_words // vocab_size
        next_word = top_words % vocab_size

        new_beams = torch.cat([beams[prev_beam], next_word.unsqueeze(1)], dim=1)
        new_alphas = torch.cat([beam_alphas[prev_beam], alpha_maps[prev_beam].unsqueeze(1)], dim=1)

        # Identify completed
        mask_end = next_word == mapping['<end>']
        if mask_end.any():
            for idx in mask_end.nonzero(as_tuple=False).flatten():
                completed_seqs.append(new_beams[idx].tolist())
                completed_alphas.append(new_alphas[idx].detach().cpu().numpy())
                completed_scores.append(top_scores[idx])
            k -= mask_end.sum().item()

        # Continue with incomplete
        if k == 0 or step > 50:
            break
        beams = new_beams[~mask_end]
        beam_alphas = new_alphas[~mask_end]
        beam_scores = top_scores[~mask_end].unsqueeze(1)
        h = h[prev_beam[~mask_end]]
        c = c[prev_beam[~mask_end]]
        feats = feats[prev_beam[~mask_end]]
        step += 1

    # Choose best
    best = completed_scores.index(max(completed_scores))
    return completed_seqs[best], completed_alphas[best]


def plot_attention(img_path, seq_indices, alpha_maps, idx_to_word, smooth=True):
    """
    Overlay attention maps for each generated word on the image.
    """
    H, W = alpha_maps.shape[1], alpha_maps.shape[2]
    img = Image.open(img_path).resize((H*24, W*24), Image.LANCZOS)
    words = [idx_to_word[i] for i in seq_indices]

    n_rows = int(np.ceil(len(words) / 5.0))
    for t, word in enumerate(words):
        if t >= 50:
            break
        plt.subplot(n_rows, 5, t + 1)
        plt.text(0, 1, word, color='black', backgroundcolor='white')
        plt.imshow(img)
        alpha = alpha_maps[t]
        if smooth:
            overlay = skt.pyramid_expand(alpha, upscale=24, sigma=8)
        else:
            overlay = skt.resize(alpha, (H*24, W*24))
        plt.imshow(overlay, alpha=0.8)
        plt.set_cmap(cm.Greys_r)
        plt.axis('off')
    plt.show()

