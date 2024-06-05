#-*- coding: utf-8 -*-

import os
import json
import pdb
import argparse
import time
import torch
import torch.nn as nn
import torchaudio
import soundfile
import numpy as np
import editdistance
import pickle
from tqdm import tqdm
import soundfile as sf

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
        # < fill your code here >
        audio, sample_rate = sf.read(os.path.join(self.dataset_path,self.data[index]['file']))
        
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
from torch.nn.utils.rnn import pad_sequence
def pad_collate(batch):
    (xx, yy) = zip(*batch)

    ## compute lengths of each item in xx and yy
    x_lens = [x.shape[0] for x in xx]# < fill your code here >
    y_lens = [y.shape[0] for y in yy]# < fill your code here >

    ## zero-pad to the longest length
    max_x = max(x_lens)
    max_y = max(y_lens)
    xx_pad = pad_sequence([x for x in xx],batch_first=True)# < fill your code here >
    yy_pad = pad_sequence([y for y in yy],batch_first=True) # < fill your code here >

    return xx_pad, yy_pad, x_lens, y_lens

## ===================================================================
## Define sampler 
#### 같은 길이의 sample들이 하나의 batch로 묶이도록 미리 설정
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
## Baseline speech recognition model
## ===================================================================

class SpeechRecognitionModel(nn.Module):

    def __init__(self, n_classes=11):
        super(SpeechRecognitionModel, self).__init__()
        
        cnns = [nn.Dropout(0.1),  
                nn.Conv1d(40,64,3, stride=1, padding=1),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Dropout(0.1),  
                nn.Conv1d(64,64,3, stride=1, padding=1),
                nn.BatchNorm1d(64),
                nn.ReLU()] 

        for i in range(2):
          cnns += [nn.Dropout(0.1),  
                   nn.Conv1d(64,64, 3, stride=1, padding=1),
                   nn.BatchNorm1d(64),
                   nn.ReLU()]

        ## define CNN layers
        self.cnns = nn.Sequential(*nn.ModuleList(cnns))

        ## define RNN layers as self.lstm - use a 3-layer bidirectional LSTM with 256 output size and 0.1 dropout
        self.lstm = nn.LSTM(input_size=64,hidden_size=256, num_layers=3, batch_first=True, bidirectional=True,dropout=0.1)# < fill your code here >

        ## define the fully connected layer
        self.classifier = nn.Linear(512,n_classes)

        self.preprocess   = torchaudio.transforms.MFCC(sample_rate=8000, n_mfcc=40)
        self.instancenorm = nn.InstanceNorm1d(40)

    def forward(self, x):

        ## compute MFCC and perform mean variance normalisation
        with torch.no_grad():
          x = self.preprocess(x)+1e-6
          x = self.instancenorm(x).detach()

        ## pass the network through the CNN layers
        
        x = self.cnns(x)# < fill your code here >
        x=x.transpose(1,2)
        #print(x.shape , 'cnn')
        ## pass the network through the RNN layers - check the input dimensions of nn.LSTM()
        x, _ = self.lstm(x) # < fill your code here >

        ## pass the network through the classifier
        x = self.classifier(x) # < fill your code here >

        return x

## ===================================================================
## Train an epoch on GPU
## ===================================================================

def process_epoch(model,loader,criterion,optimizer,trainmode=True):

    # Set the model to training or eval mode
    if trainmode:
        model.train() # < fill your code here >
    else:
        model.eval() # < fill your code here >

    ep_loss = 0
    ep_cnt  = 0

    with tqdm(loader, unit="batch") as tepoch:

        for data in tepoch:

            ## Load x and y
            x = data[0].cuda()
            y = data[1].cuda()
            y_len = torch.LongTensor(data[3])

            output = model(x) # < fill your code here >
            output = output.transpose(0,1)
            ## compute the loss using the CTC objective
            #x_len = torch.LongTensor([output.size(1)]).repeat(output.size(0))
            downsamp_factor = x.size(1) / output.size(0)
            x_len = torch.LongTensor(data[2])
            x_len_down = torch.clamp(torch.ceil(x_len / downsamp_factor),max=output.size(0)).type(torch.LongTensor)
            #print('x_len', x_len.shape)
            #print('y_len', y_len.shape)
            #print('output',output.shape)
            loss = criterion(output, y, x_len_down, y_len)

            if trainmode:
              loss.backward() # < fill your code here >
              optimizer.step()
              optimizer.zero_grad()
            # keep running average of loss
            ep_loss += loss.item() * len(x)
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
        
        # < fill your code here >
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

def process_eval(model,data_path,data_list,index2char,save_path=None):

    # set model to evaluation mode
    model.eval()

    # initialise the greedy decoder
    greedy_decoder = GreedyCTCDecoder(blank=len(index2char))

    # load data from JSON
    with open(data_list,'r') as f:
        data = json.load(f)

    results = []

    for file in tqdm(data):

        # read the wav file and convert to PyTorch format
        audio, sample_rate = soundfile.read(os.path.join(data_path, file['file']))
        audio = torch.FloatTensor(audio)# < fill your code here >
        audio = audio.unsqueeze(0).cuda()
        # forward pass through the model
        with torch.no_grad():
            output = model(audio)# < fill your code here >
            sf_max = nn.Softmax(dim=-1)
            output = sf_max(output)
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
    parser.add_argument('--train_list', type=str, default='data/ks_train.json')
    parser.add_argument('--val_list',   type=str, default='data/ks_val.json')
    parser.add_argument('--labels_path',type=str, default='data/label.json')
    parser.add_argument('--train_path', type=str, default='data/kspon_train')
    parser.add_argument('--val_path',   type=str, default='data/kspon_eval')


    ## related to training
    parser.add_argument('--max_epoch',  type=int, default=10,       help='number of epochs during training')
    parser.add_argument('--batch_size', type=int, default=20,      help='batch size')
    parser.add_argument('--lr',         type=int, default=1e-4,     help='learning rate')
    parser.add_argument('--seed',       type=int, default=2222,     help='random seed initialisation')
    
    ## relating to loading and saving
    parser.add_argument('--initial_model',  type=str, default='',   help='load initial model, e.g. for finetuning')
    parser.add_argument('--save_path',      type=str, default='./eval',   help='location to save checkpoints')

    ## related to inference
    parser.add_argument('--eval',   dest='eval',    action='store_true', help='Evaluation mode')
    parser.add_argument('--gpu',    type=int,       default=1,      help='GPU index');

    args = parser.parse_args()

    # set gpu
    os.environ["CUDA_VISIBLE_DEVICES"]='{}'.format(args.gpu)

    # load labels
    char2index, index2char = load_label_json(args.labels_path)

    ## make an instance of the model on GPU
    model = SpeechRecognitionModel(n_classes=len(char2index)+1).cuda()
    print('Model loaded. Number of parameters:',sum(p.numel() for p in model.parameters()))

    ## load from initial model
    if args.initial_model != '':
        model.load_state_dict(torch.load(args.initial_model))

    # make directory for saving models and output
    assert args.save_path != ''
    os.makedirs(args.save_path,exist_ok=True)

    ## code for inference - this uses val_path and val_list
    if args.eval:
        process_eval(model, args.val_path, args.val_list, index2char, save_path=args.save_path)
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
        collate_fn=pad_collate,
        prefetch_factor=4)
    valloader   = torch.utils.data.DataLoader(valset,   
        batch_sampler=BucketingSampler(valset, args.batch_size), 
        num_workers=4, 
        collate_fn=pad_collate,
        prefetch_factor=4)

    ## define the optimizer with args.lr learning rate and appropriate weight decay
    optimizer = torch.optim.Adam(model.parameters(),lr=args.lr)# < fill your code here >

    ## set loss function with blank index
    criterion = nn.CTCLoss(blank=len(index2char))
    
    ## initialise training log file
    f_log = open(os.path.join(args.save_path,'train.log'),'a+')
    f_log.write('{}\n'.format(args))
    f_log.flush()

    ## Train for args.max_epoch epochs
    for epoch in range(0, args.max_epoch):
        
        vloss = process_eval(model,args.val_path,args.val_list,index2char,save_path=args.save_path)

        tloss = process_epoch(model,trainloader,criterion,optimizer,trainmode=True)# < fill your code here >
        
        # save checkpoint to file
        save_file = '{}/model{:05d}.pt'.format(args.save_path,epoch)
        print('Saving model {}'.format(save_file))
        torch.save(model.state_dict(), save_file)

        # write training progress to log
        f_log.write('Epoch {:03d}, train loss {:.3f}, val loss {:.3f}\n'.format(epoch, tloss, vloss))
        f_log.flush()

    f_log.close()


if __name__ == "__main__":
    main()
