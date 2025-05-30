import os
import torch
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from datasets import ImageCaptionDataset
from utils import *
from nltk.translate.bleu_score import corpus_bleu
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm
import json

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cudnn.benchmark = True

def evaluate_dataset(data_folder, data_name, checkpoint_path, word_map_file, beam_size=1):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    encoder = ckpt['encoder'].to(device).eval()
    decoder = ckpt['decoder'].to(device).eval()

    with open(word_map_file, 'r') as j:
        word_map = json.load(j)
    vocab_size = len(word_map)

    #modificam cand schimbam encoder-ul
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    loader = torch.utils.data.DataLoader(
        ImageCaptionDataset(data_folder, data_name, 'TEST', transforms.Compose([normalize])),
        batch_size=1, shuffle=False, num_workers=1, pin_memory=True)

    references, hypotheses = [], []

    for image, caps, caplens, allcaps in tqdm(loader, desc=f"Eval {os.path.basename(data_folder)}"):
        k = beam_size
        image = image.to(device)

        encoder_out = encoder(image)                                    
        enc_dim = encoder_out.size(3)
        encoder_out = encoder_out.view(1, -1, enc_dim)                  
        encoder_out = encoder_out.expand(k, -1, enc_dim)                

        k_prev_words = torch.LongTensor([[word_map['<start>']]] * k).to(device)
        seqs = k_prev_words
        top_k_scores = torch.zeros(k, 1).to(device)
        complete_seqs, complete_seqs_scores = [], []
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
                top_k_scores, top_k_words = scores[0].topk(k, dim=0)
            else:
                top_k_scores, top_k_words = scores.view(-1).topk(k, dim=0)

            prev_word_inds = top_k_words // vocab_size
            next_word_inds = top_k_words % vocab_size
            seqs = torch.cat([seqs[prev_word_inds], next_word_inds.unsqueeze(1)], dim=1)
            incomplete_inds = [i for i, w in enumerate(next_word_inds) if w != word_map['<end>']]
            complete_inds   = list(set(range(len(next_word_inds))) - set(incomplete_inds))
            if complete_inds:
                for idx in complete_inds:
                    complete_seqs.append(seqs[idx].tolist())
                    complete_seqs_scores.append(top_k_scores[idx].item())
            k -= len(complete_inds)
            if k == 0: break

            seqs = seqs[incomplete_inds]
            h    = h[prev_word_inds[incomplete_inds]]
            c    = c[prev_word_inds[incomplete_inds]]
            encoder_out = encoder_out[prev_word_inds[incomplete_inds]]
            top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)
            k_prev_words = next_word_inds[incomplete_inds].unsqueeze(1)

            if step > 50: break
            step += 1

        if not complete_seqs_scores:
            complete_seqs = seqs.tolist()
            complete_seqs_scores = top_k_scores.squeeze(1).tolist()

        best = complete_seqs[complete_seqs_scores.index(max(complete_seqs_scores))]
        img_caps = allcaps[0].tolist()
        refs = [[w for w in cap if w not in {word_map['<start>'], word_map['<end>'], word_map['<pad>']}] 
                for cap in img_caps]
        references.append(refs)
        hyp = [w for w in best if w not in {word_map['<start>'], word_map['<end>'], word_map['<pad>']}]
        hypotheses.append(hyp)

    bleu1 = corpus_bleu(references, hypotheses, weights=(1, 0, 0, 0))
    bleu2 = corpus_bleu(references, hypotheses, weights=(0.5,0.5,0,0))
    bleu3 = corpus_bleu(references, hypotheses, weights=(0.33,0.33,0.33,0))
    bleu4 = corpus_bleu(references, hypotheses, weights=(0.25,0.25,0.25,0.25))
    return [b*100 for b in (bleu1, bleu2, bleu3, bleu4)]

def compare_and_plot():
    flickr8k = evaluate_dataset(
        '/home/mihai/workspace/output_data/Flickr8k',
        'flickr8k_5_cap_per_img_5_min_word_freq',
        '/home/mihai/workspace/output_data/Checkpoints/BEST_checkpoint_flickr8k_5_cap_per_img_5_min_word_freq.pth.tar',
        '/home/mihai/workspace/output_data/Flickr8k/WORDMAP_flickr8k_5_cap_per_img_5_min_word_freq.json',
        beam_size=1
    )
    flickr30k = evaluate_dataset(
        '/home/mihai/workspace/output_data/Flickr30k',
        'flickr30k_5_cap_per_img_5_min_word_freq',
        '/home/mihai/workspace/output_data/Checkpoints/BEST_checkpoint_flickr30k_5_cap_per_img_5_min_word_freq.pth.tar',
        '/home/mihai/workspace/output_data/Flickr30k/WORDMAP_flickr30k_5_cap_per_img_5_min_word_freq.json',
        beam_size=1
    )

    labels = ['BLEU-1','BLEU-2','BLEU-3','BLEU-4']
    x = range(len(labels))
    width = 0.35

    plt.figure(figsize=(10,6))
    plt.bar([i-width/2 for i in x], flickr8k, width, label='Flickr8k')
    plt.bar([i+width/2 for i in x], flickr30k, width, label='Flickr30k')

    for i, v in enumerate(flickr8k):
        plt.text(i-width/2, v+1, f"{v:.1f}", ha='center')
    for i, v in enumerate(flickr30k):
        plt.text(i+width/2, v+1, f"{v:.1f}", ha='center')

    plt.xticks(x, labels)
    plt.ylim(0,100)
    plt.xlabel('BLEU Metric')
    plt.ylabel('Score (%)')
    plt.title('BLEU Comparison: Flickr8k vs Flickr30k of beam size 1')
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    compare_and_plot()