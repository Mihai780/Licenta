import os
import json
import h5py
import numpy as np
import torch
from tqdm import tqdm
from random import seed, choice, sample
import imageio
from PIL import Image
from collections import Counter,defaultdict

def preprocess_dataset(dataset_name, split_spec, image_dir, caps_per_img, min_freq, output_dir, max_length=100):
    """
    Prepare HDF5 image archives and JSON caption data for training, validation, and testing.
    dataset_name: one of 'coco', 'flickr8k', 'flickr30k'
    split_pec: path to the captions
    image_dir: directory containing raw image files
    caps_per_img: number of captions to include per image
    min_freq: threshold below which words become '<unk>'
    output_dir: directory to write HDF5 and JSON outputs
    max_length: maximum allowed token length for captions
    """

    assert dataset_name in {'coco', 'flickr8k', 'flickr30k'}, "Unsupported dataset: {}".format(dataset_name)

    if dataset_name == 'flickr8k':
        caption_map = defaultdict(list)
        with open(split_spec, 'r') as cfile:
            for line in cfile:
                img_fn, caption = line.strip().split(',', 1)
                tokens = caption.strip().rstrip('.').lower().split()
                if len(tokens) <= max_length:
                    caption_map[img_fn].append(tokens)

        paths = {'train': [], 'val': [], 'test': []}
        tokens = {'train': [], 'val': [], 'test': []}
        vocab_counter = Counter()

        for img_fn, caps in caption_map.items():
            if not caps:
                continue
            paths['train'].append(os.path.join(image_dir, img_fn))
            tokens['train'].append(caps)
            for sent in caps:
                vocab_counter.update(sent)
    else:
        # Load split description
        with open(split_spec, 'r') as f:
            split_data = json.load(f)

        # Collections for each partition
        paths = {'train': [], 'val': [], 'test': []}
        tokens = {'train': [], 'val': [], 'test': []}
        vocab_counter = Counter()

        # Gather paths and tokens, build vocabulary counts
        for entry in split_data['images']:
            image_tokens = [s['tokens'] for s in entry['sentences'] if len(s['tokens']) <= max_length]
            if not image_tokens:
                continue
            vocab_counter.update([tok for sent in image_tokens for tok in sent])

            # Determine file path
            if dataset_name == 'coco':
                img_path = os.path.join(image_dir, entry['filepath'], entry['filename'])
            else:
                img_path = os.path.join(image_dir, entry['filename'])

            # Assign to partition
            split_key = 'train' if entry['split'] in {'train', 'restval'} else entry['split']
            paths[split_key].append(img_path)
            tokens[split_key].append(image_tokens)

    # Filter tokens by frequency and create index map
    vocab = [w for w, c in vocab_counter.items() if c > min_freq]
    index_map = {word: idx+1 for idx, word in enumerate(vocab)}
    index_map['<pad>'] = 0
    for special in ['<start>', '<end>','<unk>']:
        index_map[special] = len(index_map) + 1

    # Base filename for outputs
    tag = f"{dataset_name}_{caps_per_img}_cap_per_img_{min_freq}_min_word_freq"

    # Save word map
    os.makedirs(output_dir,exist_ok=True)
    with open(os.path.join(output_dir, f"WORDMAP_{tag}.json"), 'w') as f:
        json.dump(index_map, f)

    seed(28)
    ok=1
    # Process each split
    for key in ['train', 'val', 'test']:
        split_key = key.upper()
        imgs = paths[key]
        caption_lists = tokens[key]
        h5_path = os.path.join(output_dir, f"{split_key}_IMAGES_{tag}.h5")
        with h5py.File(h5_path, 'w') as h5f:
            h5f.attrs['captions_per_image'] = caps_per_img
            ds = h5f.create_dataset('images', (len(imgs), 3, 256, 256), dtype='uint8')

            encoded_caps = []
            lengths = []

            for idx, img_file in enumerate(tqdm(imgs, desc=f"Writing {split_key}")):
                # Choose captions
                captions = caption_lists[idx]
                if len(captions) < caps_per_img:
                    picks = captions + [choice(captions) for _ in range(caps_per_img - len(captions))]
                else:
                    picks = sample(captions, caps_per_img)

                # Load and resize image
                img = imageio.imread(img_file)                       
                if img.ndim == 2:
                    img = np.stack([img, img, img], axis=2)          
                pil_img = Image.fromarray(img)                       
                pil_img = pil_img.resize((256, 256), Image.BILINEAR) 
                img_resized = np.array(pil_img)                     
                img_resized = img_resized.transpose(2, 0, 1).astype('uint8')
                ds[idx] = img_resized

                # Encode captions
                for sent in picks:
                    enc = [index_map['<start>']] + [index_map.get(w, index_map['<unk>']) for w in sent] + [index_map['<end>']]
                    # pad to max_length+2
                    padding = [index_map['<pad>']] * (max_length - len(sent))
                    encoded_caps.append(enc + padding)
                    lengths.append(len(sent) + 2)

            # Verify counts
            assert len(encoded_caps) == len(lengths) == len(imgs) * caps_per_img

            # Write JSON outputs
            caps_path = os.path.join(output_dir, f"{split_key}_ENCODEDCAPS_{tag}.json")
            lens_path = os.path.join(output_dir, f"{split_key}_CAPLENGTHS_{tag}.json")
            with open(caps_path, 'w') as f:
                json.dump(encoded_caps, f)
            with open(lens_path, 'w') as f:
                json.dump(lengths, f)

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



preprocess_dataset(
    dataset_name='flickr8k',
    split_spec='/home/mihai/workspace/data/Flickr8k/captions.txt',        
    image_dir='/home/mihai/workspace/data/Flickr8k/Images',               
    caps_per_img=5,                           
    min_freq=5,                                
    output_dir='/home/mihai/workspace/output_data/Flickr8k',
    max_length=50                              
)
