import os
import json
import h5py
import numpy as np
import torch
from tqdm import tqdm
from random import seed,choice,sample,shuffle
import imageio
from PIL import Image
from collections import Counter,defaultdict

train_ratio = 0.6
val_ratio = 0.2
test_ratio = 0.2

def preprocess_dataset(dataset_name, split_spec, image_dir, caps_per_img, min_freq, output_dir, max_length=100):
    """
    Prepare HDF5 archives and JSON caption data for training, validation, and testing.
    dataset_name: 'flickr8k', 'flickr30k', 'coco'
    split_spec: path to Karpathy JSON (for COCO/Flickr30k) or captions.txt (for Flickr8k)
    image_dir: directory containing raw image files
    caps_per_img: number of captions per image
    min_freq: threshold below which words become '<unk>'
    output_dir: directory to write HDF5/JSON outputs
    max_length: maximum allowed token length for captions
    """
    assert dataset_name in {'coco', 'flickr8k', 'flickr30k'}, f"Unsupported dataset: {dataset_name}"

    paths = {'train': [], 'val': [], 'test': []}
    captions = {'train': [], 'val': [], 'test': []}
    word_freq = Counter()

    if dataset_name == 'flickr8k':
        # Read all captions
        caption_map = defaultdict(list)
        with open(split_spec, 'r') as cfile:
            for line in cfile:
                img_fn, caption = line.strip().split(',', 1)
                tokens = caption.strip().rstrip('.').lower().split()
                if tokens and len(tokens) <= max_length:
                    caption_map[img_fn].append(tokens)
                    word_freq.update(tokens)

        # Create a reproducible 60/20/20 split
        seed(28)
        img_fns = list(caption_map.keys())
        shuffle(img_fns)
        total = len(img_fns)
        train_end = int(0.6 * total)
        val_end = int(0.8 * total)

        split_map = {}
        for idx, fn in enumerate(img_fns):
            if idx < train_end:
                split_map[fn] = 'train'
            elif idx < val_end:
                split_map[fn] = 'val'
            else:
                split_map[fn] = 'test'

        # Assign images and captions to splits
        for img_fn, token_lists in caption_map.items():
            split = split_map[img_fn]
            img_path = os.path.join(image_dir, img_fn)
            paths[split].append(img_path)
            captions[split].append(token_lists)

    else:
        # Load Karpathy JSON
        with open(split_spec, 'r') as jf:
            data = json.load(jf)
        for img in data['images']:
            token_lists = [s['tokens'] for s in img['sentences'] if 1 <= len(s['tokens']) <= max_length]
            if not token_lists:
                continue
            # update frequency
            for toks in token_lists:
                word_freq.update(toks)
            # file path
            if dataset_name == 'coco':
                img_path = os.path.join(image_dir, img['filepath'], img['filename'])
            else:  # flickr30k
                img_path = os.path.join(image_dir, img['filename'])
            # split key normalization
            split = img['split']
            if split in {'restval', 'train'}:
                split = 'train'
            captions[split].append(token_lists)
            paths[split].append(img_path)

    # Sanity checks
    for split in ['train', 'val', 'test']:
        assert len(paths[split]) == len(captions[split]), f"Mismatch in {split} count"

    # --- 2. Build vocabulary ---
    vocab = [w for w, cnt in word_freq.items() if cnt > min_freq]
    index_map = {w: i+1 for i, w in enumerate(vocab)}
    index_map['<pad>'] = 0
    for special in ['<start>', '<end>', '<unk>']:
        index_map[special] = len(index_map)

    # Save word map
    os.makedirs(output_dir, exist_ok=True)
    base_tag = f"{dataset_name}_{caps_per_img}_cap_per_img_{min_freq}_min_word_freq"
    with open(os.path.join(output_dir, f"WORDMAP_{base_tag}.json"), 'w') as wm:
        json.dump(index_map, wm)

    # --- 3. Encode captions and write HDF5/JSON per split ---
    seed(28)
    for split in ['train', 'val', 'test']:
        split_key = split.upper()
        img_list = paths[split]
        cap_lists = captions[split]

        h5_path = os.path.join(output_dir, f"{split_key}_IMAGES_{base_tag}.h5")
        with h5py.File(h5_path, 'w') as h5f:
            h5f.attrs['captions_per_image'] = caps_per_img
            ds = h5f.create_dataset('images', (len(img_list), 3, 256, 256), dtype='uint8')

            encoded_caps = []
            cap_lengths = []

            for idx, img_file in enumerate(tqdm(img_list, desc=f"Writing {split_key}")):
                # sample or duplicate to exactly caps_per_img
                toks_for_img = cap_lists[idx]
                if len(toks_for_img) < caps_per_img:
                    picks = toks_for_img + [choice(toks_for_img) for _ in range(caps_per_img - len(toks_for_img))]
                else:
                    picks = sample(toks_for_img, caps_per_img)

                # load and resize image
                img_array = imageio.imread(img_file)
                if img_array.ndim == 2:
                    img_array = np.stack([img_array]*3, axis=2)
                pil_img = Image.fromarray(img_array).resize((256, 256), Image.BILINEAR)
                arr = np.array(pil_img).transpose(2, 0, 1).astype('uint8')
                ds[idx] = arr

                # encode captions
                for sent in picks:
                    enc = [index_map['<start>']] + [index_map.get(w, index_map['<unk>']) for w in sent] + [index_map['<end>']]
                    padding = [index_map['<pad>']] * (max_length - len(sent))
                    encoded_caps.append(enc + padding)
                    cap_lengths.append(len(sent) + 2)

            assert len(encoded_caps) == len(cap_lengths) == len(img_list)*caps_per_img

            # write JSON
            with open(os.path.join(output_dir, f"{split_key}_ENCODEDCAPS_{base_tag}.json"), 'w') as cf:
                json.dump(encoded_caps, cf)
            with open(os.path.join(output_dir, f"{split_key}_CAPLENGTHS_{base_tag}.json"), 'w') as lf:
                json.dump(cap_lengths, lf)

def random_weights(tensor):
    """
    Populate the tensor with values sampled uniformly from [-limit, limit]
    where limit = sqrt(1 / hidden_size).
    tensor: torch tensor to initialize
    """
    hidden_size = tensor.size(1)
    limit = np.sqrt(1.0 / hidden_size)
    with torch.no_grad():
        tensor.uniform_(-limit, limit)


def build_embedding_matrix(embeddings_path, index_map):
    """
    Constructs an embedding matrix for tokens in index_map from a GloVe-formatted file.

    embeddings_path: path to GloVe file
    index_map: mapping from token string to row index in output matrix
    Returns: (matrix of shape (vocab_size, dim), embedding dimension)
    """
    # Determine dimension by inspecting the first line
    with open(embeddings_path, 'r', encoding='utf-8') as f:
        embedding_dim = len(f.readline().strip().split())-1

    vocab_size = len(index_map)
    emb_matrix = torch.FloatTensor(vocab_size, embedding_dim)
    random_weights(emb_matrix)

    print("Loading pre-trained embeddings...")
    with open(embeddings_path, 'r', encoding='utf-8') as f:
        for entry in f:
            parts = entry.strip().split()
            token = parts[0]
            if token in index_map:
                vec = torch.tensor([float(x) for x in parts[1:]], dtype=torch.float)
                emb_matrix[index_map[token]] = vec

    return emb_matrix, embedding_dim


def save_checkpoint(data_name,current_epoch,epochs_no_improve,encoder,decoder,enc_optimiser,dec_optimiser,bleu4,is_best):
    """
    Encapsulates everything we need to persist our training state at the end of an epoch, so we can resume later or roll back to our best-performing model.
    data_name: identifier for this dataset
    current_epoch: current epoch index
    epochs_no_improve: epochs since last BLEU-4 improvement
    bleu_score: validation BLEU-4 score
    is_best: shows if the current checkpoint is the best that has been or not
    """
    # Assemble checkpoint dictionary
    checkpoint_data = {
        'epoch': current_epoch,
        'epochs_since_improvement': epochs_no_improve,
        'bleu-4': bleu4,
        'encoder': encoder,
        'decoder': decoder,
        'encoder_optimizer': enc_optimiser,
        'decoder_optimizer': dec_optimiser,
    }
    base_filename = f"checkpoint_{data_name}.pth.tar"
    torch.save(checkpoint_data, base_filename)

    #If it is the best, we have to store it to not lose it
    if is_best:
        best_filename = f"BEST_{base_filename}"
        torch.save(checkpoint_data, best_filename)

def clip_gradient(optimizer, grad_clip):
    """
    Clips gradients computed during backpropagation to avoid explosion of gradients.
    optimizer: optimizer with the gradients to be clipped
    grad_clip: clip value
    """
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


class AverageMeter:
    """
    Tracks the latest value, average, total, and count for any metric.
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.current = 0.0
        self.total = 0.0
        self.count = 0
        self.average = 0.0

    def update(self, value, instances= 1):
        self.current = value
        self.total += value * instances
        self.count += instances
        self.average = self.total / self.count if self.count else 0.0

    def adjust_learning_rate(optimizer, factor):
        """
        Multiply learning rate of each parameter group by 'factor'.
        """
        print("\nApplying LR decay by factor: {:.4f}".format(factor))
        for group in optimizer.param_groups:
            group['lr'] *= factor
        print(f"Updated LR: {optimizer.param_groups[0]['lr']:.6f}\n")

    def accuracy(output_scores,ground_truth,top_k):
        """
        Compute top-k accuracy percentage for classification predictions.
        """
        batch = ground_truth.size(0)
        _, top_indices = output_scores.topk(top_k, dim=1, largest=True, sorted=True)
        matches = top_indices.eq(ground_truth.view(-1, 1).expand_as(top_indices))
        correct = matches.float().sum().item()
        return 100.0 * (correct / batch)