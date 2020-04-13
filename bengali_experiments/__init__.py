from pathlib import Path

DATA = Path.home()/'data'/'bengaliai'
LABELS = DATA/'train.csv'
LABELS_SMALL = DATA/'labels_small.csv'
TRN_NUMPY = DATA/'tmp'/'train.npy'
TRN_IMAGE_IDS = DATA/'tmp'/'image_ids.npy'

IMG_SIZE = 137, 236
NUM_PIX_COLS = IMG_SIZE[0] * IMG_SIZE[1]
N_CLASSES = 168, 11, 7
