encoder_name = 'timm-efficientnet-b2'
decoder_name = 'UnetPlusPlus'

tags = [
    'reduceonplateau',
    'simpleeaug',
    'channelflip'
]

bs = 16
val_bs=4
LR = 1e-4
FOCAL_WEIGHT = 0.1
N_EPOCHS = 2
nfolds = 4
fold = 0
SEED = 666
TRAIN = 'data/train_features'
LABELS = 'data/train_labels'
DUMP_STORAGE = 'dumps/'
NUM_WORKERS = 8
SAVE_N_BEST = 5

assert all(['.' not in tag for tag in tags])
run_prefix = '.'.join([encoder_name]+[decoder_name]+tags)