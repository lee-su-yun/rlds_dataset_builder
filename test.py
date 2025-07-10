import numpy as np
import cv2
import matplotlib.pyplot as plt

path = '/data/piper_grape0626/pick_the_grape_and_put_it_to_the_basket/1057/episode.pickle'
data = np.load(path, allow_pickle=True)

img = cv2.imdecode(data['observation.images.table'][200], cv2.IMREAD_COLOR)
plt.imshow(img)
plt.show()