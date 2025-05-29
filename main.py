import os
import json
import torch
import torch.optim as optim
import torchvision.transforms as transforms
import torch.utils.data as data
import time
from models import Encoder, DecoderWithAttention
from torch.nn.utils.rnn import pack_padded_sequence
from torch import nn
from datasets import ImageCaptionDataset
from utils import clip_gradient, save_checkpoint, AverageMeter
from nltk.translate.bleu_score import corpus_bleu

data_folder = '/home/mihai/workspace/output_data/Flickr8k'
data_name = 'flickr8k_5_cap_per_img_5_min_word_freq'

# data_folder = '/home/mihai/workspace/output_data/Flickr30k'
# data_name = 'flickr30k_5_cap_per_img_5_min_word_freq' 

# data_folder = '/home/mihai/workspace/output_data/COCO'
# data_name = 'coco_5_cap_per_img_5_min_word_freq'  

# Network hyperparameters
embedding_size = 512
attention_size = 512
decoder_size = 512
dropout_rate = 0.5

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.backends.cudnn.benchmark = True

# Training params
initial_epoch = 0
total_epochs = 120
epochs_no_improve = 0
best_bleu_score = 0.0
batch_sz = 32
num_workers = 1
enc_lr = 1e-4
dec_lr = 4e-4
#resume_ckpt = None
resume_ckpt = "/home/mihai/workspace/output_data/Checkpoints/checkpoint_flickr8k_5_cap_per_img_5_min_word_freq.pth.tar"
fine_tune_enc = False

def main():
    """
    Training and validation.
    """
    global best_bleu_score, epochs_no_improve, checkpoint, initial_epoch, fine_tune_enc, data_name, word_to_idx

    # Load word-to-index map
    wordmap_path = os.path.join(data_folder, f'WORDMAP_{data_name}.json')
    with open(wordmap_path, 'r') as jm:
        word_to_idx = json.load(jm)
    
    if resume_ckpt is None:
        encoder = Encoder()
        encoder.fine_tune(fine_tune_enc)
        decoder = DecoderWithAttention(
            embedding_dim=embedding_size,
            vocab_size=len(word_to_idx),
            attention_dim=attention_size,
            decoder_dim=decoder_size,
            dropout=dropout_rate
        )
        enc_optimizer = optim.Adam(filter(lambda p: p.requires_grad, encoder.parameters()), lr=enc_lr) if fine_tune_enc else None
        dec_optimizer = optim.Adam(filter(lambda p: p.requires_grad, decoder.parameters()), lr=dec_lr)
    else:
        checkpoint = torch.load(resume_ckpt,map_location=device,weights_only=False)
        initial_epoch = checkpoint['epoch'] + 1
        epochs_no_improve = checkpoint['epochs_since_improvement']
        best_bleu_score = checkpoint['bleu-4']
        encoder = checkpoint['encoder']
        decoder = checkpoint['decoder']
        enc_optimizer = checkpoint['encoder_optimizer']
        dec_optimizer = checkpoint['decoder_optimizer']
        if fine_tune_enc and enc_optimizer is None:
            encoder.fine_tune(True)
            enc_optimizer = optim.Adam(filter(lambda p: p.requires_grad, encoder.parameters()), lr=enc_lr)
    
    encoder = encoder.to(device)
    decoder = decoder.to(device)

    # Loss
    criterion = nn.CrossEntropyLoss().to(device)

    # Data loaders
    normalize = transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    train_loader = data.DataLoader(
        ImageCaptionDataset(data_folder, data_name, 'TRAIN', transforms.Compose([normalize])),
        batch_size=batch_sz, shuffle=True, num_workers=num_workers, pin_memory=True
    )
    val_loader = data.DataLoader(
        ImageCaptionDataset(data_folder, data_name, 'VAL', transforms.Compose([normalize])),
        batch_size=batch_sz, shuffle=False, num_workers=num_workers, pin_memory=True
    )

    # Epoch loop
    for epoch in range(initial_epoch, total_epochs):
        if epochs_no_improve == 20:
            print('No improvement for 20 epochs, stopping.')
            break
        if epochs_no_improve > 0 and epochs_no_improve % 8 == 0:
            AverageMeter.adjust_learning_rate(dec_optimizer, 0.8)
            if fine_tune_enc:
                AverageMeter.adjust_learning_rate(enc_optimizer, 0.8)

        # Train one epoch
        train(train_loader, encoder, decoder, criterion, enc_optimizer, dec_optimizer, epoch)

        # Validate
        current_bleu4 = validate(val_loader, encoder, decoder, criterion)

        # Check improvement
        is_best = current_bleu4 > best_bleu_score
        if is_best:
            best_bleu_score = current_bleu4
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
        print(f"Epoch {epoch}: {'Improved' if is_best else 'No improvement'} (BLEU-4 = {current_bleu4:.4f})")

        # Save
        save_checkpoint(
            data_name, epoch, epochs_no_improve,
            encoder, decoder, enc_optimizer,
            dec_optimizer, best_bleu_score, is_best
        )

def train(train_data_loader, enc_model, dec_model, loss_fn, opt_enc=None, opt_dec=None, epoch_idx=0, print_interval=100, clip_value=5, alpha_c=1.0, device=device):
    """
    Execute one full training epoch.
    """
    enc_model.train()
    dec_model.train()

    timers = {
        'batch': AverageMeter(),
        'load': AverageMeter(),
        'loss': AverageMeter(),
        'top5': AverageMeter()
    }

    start_time = time.time()
    for batch_idx, (images, captions, lengths) in enumerate(train_data_loader):
        # measure data loading time
        timers['load'].update(time.time() - start_time)

        images = images.to(device)
        captions = captions.to(device)
        lengths = lengths.to(device)

        # forward pass
        features = enc_model(images)
        scores, sorted_caps, decode_lens, alphas, _ = dec_model(features, captions, lengths)

        # prepare targets (skip <start>) and pack sequences
        targets = sorted_caps[:, 1:]
        packed_scores  = pack_padded_sequence(scores,  decode_lens, batch_first=True).data
        packed_targets = pack_padded_sequence(targets, decode_lens, batch_first=True).data

        # compute loss with attention regularization
        loss = loss_fn(packed_scores, packed_targets)
        loss += alpha_c * ((1. - alphas.sum(dim=1)) ** 2).mean()

        # backward pass
        if opt_dec: opt_dec.zero_grad()
        if opt_enc: opt_enc.zero_grad()
        loss.backward()

        # gradient clipping
        if clip_value:
            clip_gradient(opt_dec, clip_value)
            if opt_enc:
                clip_gradient(opt_enc, clip_value)

        if opt_dec: opt_dec.step()
        if opt_enc: opt_enc.step()

        top5 = AverageMeter.accuracy(packed_scores, packed_targets, 5)
        num_tokens = sum(decode_lens)
        timers['loss'].update(loss.item(), num_tokens)
        timers['top5'].update(top5, num_tokens)
        timers['batch'].update(time.time() - start_time)

        start_time = time.time()

        if batch_idx % print_interval == 0:
            print(f"Epoch [{epoch_idx}][{batch_idx}/{len(train_data_loader)}] "
                  f"Load {timers['load'].current:.3f} ({timers['load'].average:.3f}) "
                  f"Batch {timers['batch'].current:.3f} ({timers['batch'].average:.3f}) "
                  f"Loss {timers['loss'].current:.4f} ({timers['loss'].average:.4f}) "
                  f"Top-5 {timers['top5'].current:.3f} ({timers['top5'].average:.3f})")
            
def validate(valid_data_loader, enc_model, dec_model, loss_fn,print_interval=100, alpha_c=1.0, device=device):
    """
    Run validation and return BLEU-4 score.
    """
    enc_model.eval()
    dec_model.eval()

    timers = {
        'batch': AverageMeter(),
        'loss': AverageMeter(),
        'top5': AverageMeter()
    }

    references, hypotheses = [], []
    start_time = time.time()

    with torch.no_grad():
        for batch_idx, (images, captions, lengths, all_captions) in enumerate(valid_data_loader):
            images = images.to(device)
            captions = captions.to(device)
            lengths = lengths.to(device)

            features = enc_model(images)
            scores, sorted_caps, decode_lens, alphas, sort_idx = dec_model(features, captions, lengths)

            targets = sorted_caps[:, 1:]

            # keep a copy for prediction
            raw_scores = scores.clone()

            packed_scores  = pack_padded_sequence(scores,  decode_lens, batch_first=True).data
            packed_targets = pack_padded_sequence(targets, decode_lens, batch_first=True).data

            loss = loss_fn(packed_scores, packed_targets)
            loss += alpha_c * ((1. - alphas.sum(dim=1)) ** 2).mean()

            # update metrics
            top5 = AverageMeter.accuracy(packed_scores, packed_targets, 5)
            num_tokens = sum(decode_lens)
            timers['loss'].update(loss.item(), num_tokens)
            timers['top5'].update(top5, num_tokens)
            timers['batch'].update(time.time() - start_time)
            start_time = time.time()

            if batch_idx % print_interval == 0:
                print(f"Validation [{batch_idx}/{len(valid_data_loader)}] "
                      f"Batch {timers['batch'].current:.3f} ({timers['batch'].average:.3f}) "
                      f"Loss {timers['loss'].current:.4f} ({timers['loss'].average:.4f}) "
                      f"Top-5 {timers['top5'].current:.3f} ({timers['top5'].average:.3f})")

            # gather references
            all_captions = all_captions.to(device)
            sorted_all_caps = all_captions[sort_idx]
            _, pred_idxs = torch.max(raw_scores, dim=2)

            for img_idx in range(len(decode_lens)):
                caps_for_img = sorted_all_caps[img_idx]
                refs = []
                for ref_tensor in caps_for_img:
                    tokens = [w for w in ref_tensor.tolist()
                            if w not in {word_to_idx['<start>'], word_to_idx['<pad>']}]
                    refs.append(tokens)
                references.append(refs)
                
                hyp_tokens = pred_idxs[img_idx, :decode_lens[img_idx]].tolist()
                hypotheses.append(hyp_tokens)

    # compute BLEU-4
    bleu4 = corpus_bleu(references, hypotheses)
    print(f"\n * Validation BLEU-4: {bleu4:.4f}, Loss: {timers['loss'].average:.4f}, Top-5: {timers['top5'].average:.4f}\n")
    return bleu4

if __name__ == '__main__':
    main()