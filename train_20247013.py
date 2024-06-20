#-*- coding: utf-8 -*-

import os
import json
import argparse
import numpy as np
import editdistance
from tqdm import tqdm
import soundfile as sf
from functools import partial

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F

from torch.utils.tensorboard import SummaryWriter
import torchaudio
from torchaudio.models import Conformer

## ===================================================================
# Model archictecture
## ===================================================================
class Conv2dSubsampling(nn.Module):
  '''
    Subsampling mel spectrogram 4x size for reducing length.
    Constituted with two 2D convolution layers.
  '''
  def __init__(self, d_model=144):
    super(Conv2dSubsampling, self).__init__()
    self.module = nn.Sequential(
      nn.Conv2d(in_channels=1, out_channels=d_model, kernel_size=3, stride=2),
      nn.ReLU(),
      nn.Conv2d(in_channels=d_model, out_channels=d_model, kernel_size=3, stride=2),
      nn.ReLU(),
    )

  def forward(self, x):
    output = self.module(x.unsqueeze(1)) # (batch_size, 1, time, d_input)
    batch_size, d_model, subsampled_time, subsampled_freq = output.size()
    output = output.permute(0, 2, 1, 3)
    output = output.contiguous().view(batch_size, subsampled_time, d_model * subsampled_freq)
    return output


class Conformer_Decoder(nn.Module):
  '''
    Conv2dsubsampling
    Conformer
    Decoder-LSTM
  '''
  def __init__(
          self,
          d_input=80,
          d_model=144,
          num_layers=16,
          conv_kernel_size=31, 
          feed_forward_residual_factor=.5,
          feed_forward_expansion_factor=4,
          num_heads=4,
          dropout=.1,
          n_classes = 11

  ):
    super(Conformer_Decoder, self).__init__()
    self.conv_subsample = Conv2dSubsampling(d_model=d_model)
    self.linear_proj = nn.Linear(d_model * (((d_input - 1) // 2 - 1) // 2), d_model) # project subsamples to d_model
    self.dropout = nn.Dropout(p=dropout)

    self.conformer = Conformer(            
            input_dim = d_model,
            num_heads = num_heads,
            ffn_dim = d_model*feed_forward_expansion_factor,
            num_layers=num_layers,
            depthwise_conv_kernel_size=conv_kernel_size,
            dropout= dropout)

    self.lstm = nn.LSTM(input_size=d_model, hidden_size=320, num_layers=1, batch_first=True, bidirectional=True)
    self.linear = nn.Linear(640, n_classes)

  def forward(self, x, x_len):
    # Conv2d Subsampling
    x = self.conv_subsample(x)
    x_len_sub = (((x_len - 1) // 2 - 1) // 2) 
    x = self.linear_proj(x)
    x = self.dropout(x)

    # Conformer
    x, output_length = self.conformer(x,lengths = x_len_sub)

    # Decoder
    x, _ = self.lstm(x)
    logits = self.linear(x)
    return logits, output_length

## ===================================================================
## Load labels
#### labels_path에서 순서대로 char마다 idx쌍 부여
## ===================================================================

def load_label_json(labels_path):
    with open(labels_path, encoding="utf-8") as label_file:
        labels = json.load(label_file)
        char2index = dict()
        index2char = dict()

        for index, char in enumerate(labels):
            char2index[char] = index
            index2char[index] = char
            
        return char2index, index2char

## ===================================================================
## Data loader
## ===================================================================

class SpeechDataset(torch.utils.data.Dataset):
    def __init__(self, data_list, data_path, max_length, char2index):
        super(SpeechDataset, self).__init__()

        # load data from JSON
        with open(data_list,'r') as f:
            data = json.load(f)

        # convert seconds to frames
        max_length *= 16000

        # sort data in length order and filter data less than max_length
        data = sorted(data, key=lambda d: d['len'], reverse=True)
        self.data = [x for x in data if x['len'] <= max_length]

        self.dataset_path   = data_path
        self.char2index     = char2index

    def __getitem__(self, index):

        # read audio using soundfile.read
        audio, _ = sf.read(os.path.join(self.dataset_path,self.data[index]['file']))
        
        # read transcript and convert to indices
        transcript = self.data[index]['text']
        transcript = self.parse_transcript(transcript)

        return torch.FloatTensor(audio), torch.LongTensor(transcript)

    def parse_transcript(self, transcript):
        transcript = list(filter(None, [self.char2index.get(x) for x in list(transcript)]))
        return transcript

    def __len__(self):
        return len(self.data)


## ===================================================================
## Define collate function
## ===================================================================

def preprocess_audio_collate_fn(batch, batch_type = 'train'):
    train_audio_transform = torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_mels=80, hop_length=160)
    valid_audio_transform = torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_mels=80, hop_length=160)
    if batch_type =='train':
        audio_transform = train_audio_transform
    elif batch_type == 'valid':
        audio_transform = valid_audio_transform
    
    # 
    spectograms = []
    (xx, yy) = zip(*batch)
    for x in xx:
        spec = audio_transform(x).squeeze(0).transpose(0,1) # (seq, freq)
        spectograms.append(spec)
    xx_pad = pad_sequence([x for x in spectograms],batch_first=True)
    x_lens = [x.shape[0] for x in spectograms]

    y_lens = [y.shape[0] for y in yy]
    yy_pad = pad_sequence([y for y in yy],batch_first=True)
    return xx_pad, yy_pad, x_lens, y_lens

train_collate_fn = partial(preprocess_audio_collate_fn,batch_type='train')
valid_collate_fn = partial(preprocess_audio_collate_fn,batch_type='valid')


## ===================================================================
## Define sampler 
## ===================================================================

class BucketingSampler(torch.utils.data.sampler.Sampler):
    def __init__(self, data_source, batch_size=1):
        """
        Samples batches assuming they are in order of size to batch similarly sized samples together.
        """
        super(BucketingSampler, self).__init__(data_source)
        self.data_source = data_source
        ids = list(range(0, len(data_source)))
        self.bins = [ids[i:i + batch_size] for i in range(0, len(ids), batch_size)]

    def __iter__(self):

        # Shuffle bins in random order
        np.random.shuffle(self.bins)

        # For each bin
        for ids in self.bins:
            # Shuffle indices in random order
            np.random.shuffle(ids)
            yield ids

    def __len__(self):
        return len(self.bins)

## ===================================================================
## Train an epoch on GPU
## ===================================================================

def process_epoch(args, epoch, model,loader,criterion,optimizer, writer, trainmode=True):
    # Set the model to training or eval mode
    if trainmode:
        model.train() 
    else:
        model.eval() 

    with tqdm(loader, unit="batch") as tepoch:
        print('LR: ',optimizer.param_groups[0]['lr'])
        ep_loss = 0
        ep_cnt  = 0
        for i, data in enumerate(tepoch):
            step = epoch*len(tepoch)+i
            ## Load x and y
            x = data[0].cuda()
            y = data[1].cuda()

            x_len = torch.LongTensor(data[2])
            y_len = torch.LongTensor(data[3])

            output, _ = model(x, x_len.cuda()) 
            output = F.log_softmax(output, dim=-1)
            output = output.transpose(0,1) # seq batch

            downsamp_factor = x.size(1) / output.size(0)
            x_len = torch.LongTensor(data[2])
            x_len_down = torch.clamp(torch.ceil(x_len / downsamp_factor),max=output.size(0)).type(torch.LongTensor)
            loss = criterion(output, y, x_len_down, y_len)
            
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"Skipping backward pass at iteration {i} due to NaN or Inf loss")
                continue

            if trainmode:
              writer.add_scalar('Loss/train', loss.item(), step)
              loss = loss / args.accumulate_iters
              loss.backward()
              
              if (i+1) % args.accumulate_iters == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.max_norm)
                optimizer.step()
                optimizer.zero_grad()

            # keep running average of loss
            ep_loss += loss.item() * len(x) * args.accumulate_iters
            ep_cnt  += len(x)

            # print value to TQDM
            tepoch.set_postfix(loss=ep_loss/ep_cnt)
            
    return ep_loss/ep_cnt
## ===================================================================
## Greedy CTC Decoder
## ===================================================================

class GreedyCTCDecoder(torch.nn.Module):
    def __init__(self, blank=0):
        super().__init__()
        self.blank = blank

    def forward(self, emission: torch.Tensor):
        """
        Given a sequence emission over labels, get the best path.
        """
        # find the index of the maximum probability output at each time step

        sf_max = nn.Softmax(dim=-1)
        max_emission = sf_max(emission)
        max_index = torch.argmax(max_emission, dim=1)
        # remove the repeats

        def remove_repeats(indices_list):
          reduced_list = []
          for i, val in enumerate(indices_list.tolist()):
            if i ==0:
              reduced_list.append(val)
              prev = val
            elif val == prev:
              prev = val
            else:
              reduced_list.append(val)
              prev = val
          return torch.Tensor(reduced_list)

        reduced_max_index = remove_repeats(max_index)
        # convert to numpy array

        reduced_max_index = reduced_max_index.numpy()
        # remove the blank symbols

        indices = np.delete(reduced_max_index, np.where(reduced_max_index==self.blank))
        return indices

## ===================================================================
## Evaluation script
## ===================================================================

def process_eval_loss(model,epoch, val_loader, criterion, writer):

    # set model to evaluation mode
    model.eval()
    val_loss =0
    val_cnt = 0

    with tqdm(val_loader, unit='batch') as vepoch:
        with torch.no_grad():
            for i,data in enumerate(vepoch):
                step = epoch*len(vepoch)+i
                x = data[0].cuda()
                y = data[1].cuda()

                x_len = torch.LongTensor(data[2])
                y_len = torch.LongTensor(data[3])
                
                output, _ = model(x, x_len.cuda()) 
                output = F.log_softmax(output, dim=-1)
                output = output.transpose(0,1) # seq batch

                downsamp_factor = x.size(1) / output.size(0)
                x_len = torch.LongTensor(data[2])
                x_len_down = torch.clamp(torch.ceil(x_len / downsamp_factor),max=output.size(0)).type(torch.LongTensor)
                
                loss = criterion(output, y, x_len_down, y_len)
                
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"Skipping backward pass at iteration {i} due to NaN or Inf loss")
                    continue
                writer.add_scalar('Loss/valid', loss.item(), step)
                val_loss += loss.item() * len(x)
                val_cnt  += len(x)
                # print value to TQDM
                if val_cnt ==0:
                   val_cnt =1
                vepoch.set_postfix(loss=val_loss/val_cnt)   
    if val_cnt ==0:
       val_cnt =1
    return val_loss/val_cnt

def process_eval_cer(model,data_path,data_list,index2char,save_path=None):

    # set model to evaluation mode
    model.eval()
    # initialise the greedy decoder
    greedy_decoder = GreedyCTCDecoder(blank=len(index2char))
    valid_audio_transform = torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_mels=80, hop_length=160)
    # load data from JSON
    with open(data_list,'r') as f:
        data = json.load(f)

    results = []

    for file in tqdm(data):

        # read the wav file and convert to PyTorch format
        audio, sample_rate = sf.read(os.path.join(data_path, file['file']))
        audio = torch.FloatTensor(audio)
        audio = valid_audio_transform(audio.unsqueeze(0)).squeeze(0).transpose(0,1) # (seq, freq)

        audio = audio.unsqueeze(0).cuda()
        audio_len = torch.LongTensor([audio.shape[1]]).cuda()
        # forward pass through the model
        with torch.no_grad():
            output, _ = model(audio, audio_len)# < fill your code here >
            output = F.log_softmax(output,dim=-1)
            output = output.transpose(0,1)
        # decode using the greedy decoder
        pred = greedy_decoder(output.cpu().detach().squeeze()) # < fill your code here >

        # convert to text
        out_text = ''.join([index2char[x] for x in pred])

        # keep log of the results
        file['pred'] = out_text
        if 'text' in file:
            file['edit_dist']   = editdistance.eval(out_text.replace(' ',''),file['text'].replace(' ',''))
            file['gt_len']     = len(file['text'].replace(' ',''))
        results.append(file)
    
    # save results to json file
    with open(os.path.join(save_path,'results.json'), 'w', encoding='utf-8') as outfile:
        json.dump(results, outfile, ensure_ascii=False, indent=2)

    # print CER if there is ground truth
    if 'text' in file:
        cer = sum([x['edit_dist'] for x in results]) / sum([x['gt_len'] for x in results])
        print('Character Error Rate is {:.2f}%'.format(cer*100))

## ===================================================================
## Main execution script
## ===================================================================

def main():
    parser = argparse.ArgumentParser(description='EE738 Exercise')

    ## related to data loading
    parser.add_argument('--max_length', type=int, default=10,   help='maximum length of audio file in seconds')
    parser.add_argument('--train_list', type=str, default='./data/ks_train.json')
    parser.add_argument('--val_list',   type=str, default='./data/ks_val.json')
    parser.add_argument('--test_list',   type=str, default='./data/ks_test_rel.json')
    parser.add_argument('--labels_path',type=str, default='./data/label.json')
    parser.add_argument('--train_path', type=str, default='./data/kspon_train')
    parser.add_argument('--val_path',   type=str, default='./data/kspon_eval')

    ## related to training
    parser.add_argument('--max_epoch',  type=int, default=10,       help='number of epochs during training')
    parser.add_argument('--start_epoch',  type=int, default=0,       help='number of epochs starts training')
    parser.add_argument('--batch_size', type=int, default=64,      help='batch size')
    parser.add_argument('--lr',         type=float, default=1e-3,     help='learning rate')
    parser.add_argument('--seed',       type=int, default=2222,     help='random seed initialisation')
    parser.add_argument('--accumulate_iters', type=int, default=1,     help='gradient accumulate iters, makes batch size bs*accumulate iter')
    parser.add_argument('--max_norm', type=int, default=10,     help='max norm for gradient clipping')
    
    ## relating to loading and saving
    parser.add_argument('--initial_model',  type=str, default='',   help='load initial model, e.g. for finetuning')
    parser.add_argument('--save_path',      type=str, default='./eval',   help='location to save checkpoints')
    parser.add_argument('--log_dir',      type=str, default='./runs/experiment1',   help='location to save checkpoints')

    ## related to inference
    parser.add_argument('--eval',   dest='eval',    action='store_true', help='Evaluation mode')
    parser.add_argument('--gpu',    type=int,       default=0,      help='GPU index');

    args = parser.parse_args()
    # log writer
    log_dir = args.log_dir # "runs/experiment1"
    writer = SummaryWriter(log_dir)

    # set gpu
    os.environ["CUDA_VISIBLE_DEVICES"]='{}'.format(args.gpu)

    # load labels
    char2index, index2char = load_label_json(args.labels_path)

    ## make an instance of the model on GPU
    model = Conformer_Decoder(
        d_input = 80,
        d_model = 144,
        num_layers = 16,
        conv_kernel_size=31, 
        feed_forward_residual_factor=.5,
        feed_forward_expansion_factor=4,
        num_heads=4,
        dropout=.1,
        n_classes=len(char2index)+1
    ).cuda()


    print('model loaded. Number of parameters:',sum(p.numel() for p in model.parameters()))
    ## define the optimizer with args.lr learning rate and appropriate weight decay
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-6, betas=(0.9,0.98), eps = 1e-9)


    ## load from initial model
    if args.initial_model != '':
        checkpoint = torch.load(args.initial_model)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-6, betas=(0.9,0.98), eps = 1e-9)
    # make directory for saving models and output
    assert args.save_path != ''
    os.makedirs(args.save_path,exist_ok=True)

    ## code for inference - this uses val_path and val_list
    if args.eval:
        process_eval_cer(model, args.val_path, args.test_list, index2char, save_path=args.save_path)
        quit();

    # initialise seeds
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    # define datasets
    trainset  = SpeechDataset(args.train_list, args.train_path, args.max_length, char2index)
    valset    = SpeechDataset(args.val_list,   args.val_path,   args.max_length, char2index)

    # initiate loader for each dataset with 'collate_fn' argument
    # do not use more than 6 workers
    trainloader = torch.utils.data.DataLoader(trainset, 
        batch_sampler=BucketingSampler(trainset, args.batch_size), 
        num_workers=4, 
        collate_fn=train_collate_fn,
        prefetch_factor=4)
    
    valloader = torch.utils.data.DataLoader(valset,   
        batch_sampler=BucketingSampler(valset, args.batch_size), 
        num_workers=4, 
        collate_fn=valid_collate_fn,
        prefetch_factor=4)

    ## set loss function with blank index
    criterion = nn.CTCLoss(blank=len(index2char))
    
    ## initialise training log file
    f_log = open(os.path.join(args.save_path,'train.log'),'a+')
    f_log.write('{}\n'.format(args))
    f_log.flush()

    ## Train for args.max_epoch epochs
    for epoch in range(args.start_epoch, args.max_epoch):
        process_eval_cer(model, args.val_path,args.val_list,index2char,save_path=args.save_path)
        tloss = process_epoch(args, epoch, model,trainloader,criterion,optimizer, writer, trainmode=True)
        
        # save checkpoint to file
        save_file = '{}/model{:05d}.pt'.format(args.save_path,epoch)
        print('Saving model {}'.format(save_file))
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }, save_file)

        vloss = process_eval_loss(model,epoch, valloader,criterion, writer)
        f_log.write('val loss {:.3f}\n'.format(vloss))
        # write training progress to log
        f_log.write('Epoch {:03d}, train loss {:.3f}, val loss {:.3f}\n'.format(epoch, tloss, vloss))
        f_log.flush()

    f_log.close()
    writer.close()

if __name__ == "__main__":
    main()
