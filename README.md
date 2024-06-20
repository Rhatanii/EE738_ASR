Automatic Speech Recognition Project

Dataset: Kspon Dataset
Model: 16-layer Conformer

Baseline Validation CER: 31.85%
Conformer Validation CER: 14.67%


- 2DConvSubsampling enables large batch size training reducing mel spectrogram lengths 4x.
- Additional mel spectorgram masking augmentation would help training. 
- (To do) Transformer Learning rate Scheduler is mentioned in the paper, but I didn't find the optimal point.
