import pickle

with open('/sdb1/piper_5hz/train/Align_the_cups/68/episode.pickle', 'rb') as f:
    data = pickle.load(f)

print('success')