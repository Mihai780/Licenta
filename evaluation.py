import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from datasets import *
from utils import *
from nltk.translate.bleu_score import corpus_bleu
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# Parameters
# data_folder = '/home/mihai/workspace/output_data/Flickr30k'
# data_name = 'flickr30k_5_cap_per_img_5_min_word_freq'
# checkpoint = "/home/mihai/workspace/output_data/Checkpoints/BEST_checkpoint_flickr30k_5_cap_per_img_5_min_word_freq.pth.tar"
# word_map_file = "/home/mihai/workspace/output_data/Flickr30k/WORDMAP_flickr30k_5_cap_per_img_5_min_word_freq.json"

data_folder = '/home/mihai/workspace/output_data/Flickr8k'
data_name = 'flickr8k_5_cap_per_img_5_min_word_freq'
checkpoint = "/home/mihai/workspace/output_data/Checkpoints/BEST_checkpoint_flickr8k_5_cap_per_img_5_min_word_freq.pth.tar"
word_map_file = "/home/mihai/workspace/output_data/Flickr8k/WORDMAP_flickr8k_5_cap_per_img_5_min_word_freq.json"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cudnn.benchmark = True

# Load model
checkpoint = torch.load(checkpoint,map_location=device,weights_only=False)
decoder = checkpoint['decoder']
decoder = decoder.to(device)
decoder.eval()
encoder = checkpoint['encoder'].to(device)
encoder = encoder.to(device)
encoder.eval()

# Load word map (word2ix)
with open(word_map_file, 'r') as j:
    word_map = json.load(j)
rev_word_map = {v: k for k, v in word_map.items()}
vocab_size = len(word_map)

# Normalization transform
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])


def evaluate(beam_size):
    """
    Evaluation
    """
    # DataLoader
    loader = torch.utils.data.DataLoader(
        ImageCaptionDataset(data_folder, data_name, 'TEST', transforms.Compose([normalize])),
        batch_size=1, shuffle=True, num_workers=1, pin_memory=True)

    references = list()
    hypotheses = list()

    # For each image
    for i, (image, caps, caplens, allcaps) in enumerate(
            tqdm(loader, desc="EVALUATING AT BEAM SIZE " + str(beam_size))):

        k = beam_size

        image = image.to(device)

        # Encode
        encoder_out = encoder(image)
        enc_image_size = encoder_out.size(1)
        encoder_dim = encoder_out.size(3)

        encoder_out = encoder_out.view(1, -1, encoder_dim)
        num_pixels = encoder_out.size(1)
        encoder_out = encoder_out.expand(k, num_pixels, encoder_dim)
        k_prev_words = torch.LongTensor([[word_map['<start>']]] * k).to(device)
        seqs = k_prev_words
        top_k_scores = torch.zeros(k, 1).to(device)
        complete_seqs = list()
        complete_seqs_scores = list()
        step = 1
        h, c = decoder.initial_states(encoder_out)
        while True:

            embeddings = decoder.embedding(k_prev_words).squeeze(1)
            awe, _ = decoder.attention(encoder_out, h)
            gate = decoder.sigmoid(decoder.beta_gate(h))
            awe = gate * awe
            h, c = decoder.decode_step(torch.cat([embeddings, awe], dim=1), (h, c))
            scores = decoder.fcl(h)
            scores = F.log_softmax(scores, dim=1)
            scores = top_k_scores.expand_as(scores) + scores
            if step == 1:
                top_k_scores, top_k_words = scores[0].topk(k, 0, True, True)
            else:
                top_k_scores, top_k_words = scores.view(-1).topk(k, 0, True, True)

            prev_word_inds = top_k_words // vocab_size
            next_word_inds = top_k_words % vocab_size
            seqs = torch.cat([seqs[prev_word_inds], next_word_inds.unsqueeze(1)], dim=1)
            incomplete_inds = [ind for ind, next_word in enumerate(next_word_inds) if
                               next_word != word_map['<end>']]
            complete_inds = list(set(range(len(next_word_inds))) - set(incomplete_inds))

            if len(complete_inds) > 0:
                complete_seqs.extend(seqs[complete_inds].tolist())
                complete_seqs_scores.extend(top_k_scores[complete_inds])
            k -= len(complete_inds) 

            if k == 0:
                break
            seqs = seqs[incomplete_inds]
            h = h[prev_word_inds[incomplete_inds]]
            c = c[prev_word_inds[incomplete_inds]]
            encoder_out = encoder_out[prev_word_inds[incomplete_inds]]
            top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)
            k_prev_words = next_word_inds[incomplete_inds].unsqueeze(1)

            if step > 50:
                break
            step += 1

        if len(complete_seqs_scores) == 0:
            complete_seqs = seqs.tolist()
            complete_seqs_scores = top_k_scores.squeeze(1).tolist()

        i = complete_seqs_scores.index(max(complete_seqs_scores))
        seq = complete_seqs[i]

        img_caps = allcaps[0].tolist()
        img_captions = list(
            map(lambda c: [w for w in c if w not in {word_map['<start>'], word_map['<end>'], word_map['<pad>']}],
                img_caps))
        references.append(img_captions)

        hypotheses.append([w for w in seq if w not in {word_map['<start>'], word_map['<end>'], word_map['<pad>']}])

        assert len(references) == len(hypotheses)

    bleu1 = corpus_bleu(references, hypotheses, weights=(1.0, 0, 0, 0))
    bleu2 = corpus_bleu(references, hypotheses, weights=(0.5, 0.5, 0, 0))
    bleu3 = corpus_bleu(references, hypotheses, weights=(0.33, 0.33, 0.33, 0))
    bleu4 = corpus_bleu(references, hypotheses, weights=(0.25, 0.25, 0.25, 0.25))
    return bleu1, bleu2, bleu3, bleu4



def plot_bleu_scores(beam_size: int, data_folder: str):
    bleu1, bleu2, bleu3, bleu4 = evaluate(beam_size)

    bleu_scores = [bleu1*100, bleu2*100, bleu3*100, bleu4*100]
    labels = ['BLEU-1', 'BLEU-2', 'BLEU-3', 'BLEU-4']

    plt.figure(figsize=(8, 5))
    bars = plt.bar(labels, bleu_scores)

    for bar, score in zip(bars, bleu_scores):
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,  
            height + 0.02,                       
            f"{score:.2f}",                      
            ha='center',                         
            va='bottom'                          
        )

    plt.xlabel('BLEU Metric')
    plt.ylabel('Score')
    plt.ylim(0, 100)
    plt.title(f'BLEU Scores for {os.path.basename(data_folder)}')
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    beam_size = 1
    plot_bleu_scores(beam_size, data_folder)