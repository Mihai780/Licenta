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

    caption_map = defaultdict(list)
    if dataset_name == 'coco':
        # Load train annotations
        train_ann = os.path.join(split_spec, 'captions_train2017.json')
        with open(train_ann, 'r') as f:
            data = json.load(f)
        images_info = {img['id']: img['file_name'] for img in data['images']}
        for ann in data['annotations']:
            fn = images_info[ann['image_id']]
            tokens = ann['caption'].rstrip('.').lower().split()
            word_freq.update(tokens)
            if tokens and len(tokens) <= max_length:
                caption_map[('train', fn)].append(tokens)

        # Load val annotations
        val_ann = os.path.join(split_spec, 'captions_val2017.json')
        with open(val_ann, 'r') as f:
            data = json.load(f)
        images_info = {img['id']: img['file_name'] for img in data['images']}
        for ann in data['annotations']:
            fn = images_info[ann['image_id']]
            tokens = ann['caption'].rstrip('.').lower().split()
            word_freq.update(tokens)
            if tokens and len(tokens) <= max_length:
                caption_map[('val', fn)].append(tokens)

        # Populate validation set
        for fn, toks_list in caption_map.items():
            if fn[0] == 'val':
                caps = toks_list[:caps_per_img]
                img_path = os.path.join(image_dir, 'val2017', fn[1])
                paths['val'].append(img_path)
                captions['val'].append(caps)

        seed(1234)
        train_fns = [fn for (split, fn) in caption_map if split == 'train']
        shuffle(train_fns)
        test_fns = set(train_fns[:5000])
        train_only_fns = set(train_fns[5000:])

        # Populate test set
        for fn in test_fns:
            toks = caption_map[('train', fn)][:caps_per_img]
            img_path = os.path.join(image_dir, 'train2017', fn)
            paths['test'].append(img_path)
            captions['test'].append(toks)

        # Populate train set with the remaining images
        for fn in train_only_fns:
            toks = caption_map[('train', fn)][:caps_per_img]
            img_path = os.path.join(image_dir, 'train2017', fn)
            paths['train'].append(img_path)
            captions['train'].append(toks)

    else:
        if dataset_name == 'flickr8k':
            with open(split_spec, 'r') as cfile:
                for line in cfile:
                    img_fn, caption = line.strip().split(',', 1)
                    tokens = caption.strip().rstrip('.').lower().split()
                    word_freq.update(tokens)
                    if tokens and len(tokens) <= max_length:
                        caption_map[img_fn].append(tokens)
        else:
            with open(split_spec, 'r') as f:
                for line in f:
                    if not line.strip():
                        continue
                    img_fn, caption = line.strip().split(',', 1)
                    tokens = caption.rstrip('.').lower().split()
                    word_freq.update(tokens)
                    if tokens and len(tokens) <= max_length:
                        caption_map[img_fn].append(tokens)

        seed(28)
        img_fns = list(caption_map.keys())
        shuffle(img_fns)
        total = len(img_fns)
        train_end = int(0.8 * total)
        val_end   = int(0.9 * total)
        split_map = {fn: ('train' if i < train_end else 'val'   if i < val_end else 'test') for i, fn in enumerate(img_fns)}

        for img_fn, token_lists in caption_map.items():
            split = split_map[img_fn]
            img_path = os.path.join(image_dir, img_fn)
            token_lists = token_lists[:caps_per_img]
            paths[split].append(img_path)
            captions[split].append(token_lists)
    

    # Checks
    for split in ['train', 'val', 'test']:
        assert len(paths[split]) == len(captions[split]), f"Mismatch in {split} count"

    # Build vocabulary
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

    # Encode captions and write HDF5 and JSON per split
    seed(28)
    for split in ['train', 'val', 'test']:
        split_key = split.upper()
        img_list = paths[split]
        cap_lists = captions[split]

        h5_path = os.path.join(output_dir, f"{split_key}_IMAGES_{base_tag}.h5")
        with h5py.File(h5_path, 'w') as h5f:
            h5f.attrs['captions_per_image'] = caps_per_img
            ds = h5f.create_dataset('images', (len(img_list), 3, 380, 380), dtype='uint8')

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
                pil_img = Image.fromarray(img_array).resize((380, 380), Image.BILINEAR)
                arr = np.array(pil_img).transpose(2, 0, 1).astype('uint8')
                ds[idx] = arr

                # encode captions
                for sent in picks:
                    enc = [index_map['<start>']] + [index_map.get(w, index_map['<unk>']) for w in sent] + [index_map['<end>']]
                    padding = [index_map['<pad>']] * (max_length - len(sent))
                    encoded_caps.append(enc + padding)
                    cap_lengths.append(len(sent) + 2)

            assert len(encoded_caps) == len(cap_lengths) == len(img_list)*caps_per_img

            with open(os.path.join(output_dir, f"{split_key}_ENCODEDCAPS_{base_tag}.json"), 'w') as cf:
                json.dump(encoded_caps, cf)
            with open(os.path.join(output_dir, f"{split_key}_CAPLENGTHS_{base_tag}.json"), 'w') as lf:
                json.dump(cap_lengths, lf)


def save_checkpoint(data_name, current_epoch, epochs_no_improve,encoder, decoder, enc_optimiser, dec_optimiser,bleu4, is_best):
    """
    Encapsulates everything we need to persist our training state at the end of an epoch,
    so we can resume later or roll back to our best-performing model.
    """
    # Target directory
    ckpt_dir = '/home/mihai/workspace/output_data/Checkpoints'
    os.makedirs(ckpt_dir, exist_ok=True)

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

    # Base filename
    base_name = f"checkpoint_{data_name}.pth.tar"
    ckpt_path = os.path.join(ckpt_dir, base_name)
    torch.save(checkpoint_data, ckpt_path)

    # If it's the best, also save a copy prefixed with BEST_
    if is_best:
        best_path = os.path.join(ckpt_dir, f"BEST_{base_name}")
        torch.save(checkpoint_data, best_path)

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
        correct_total = matches.float().sum().item()
        return 100.0 * (correct_total / batch)