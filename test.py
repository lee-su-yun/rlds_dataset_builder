import numpy as np
import cv2
import matplotlib.pyplot as plt

path = '/data/piper_grape0626/pick_the_grape_and_put_it_to_the_basket/1057/episode.pickle'
data = np.load(path, allow_pickle=True)

img100 = cv2.imdecode(data['observation.images.table'][100], cv2.IMREAD_COLOR)
img150 = cv2.imdecode(data['observation.images.table'][150], cv2.IMREAD_COLOR)
img200 = cv2.imdecode(data['observation.images.table'][200], cv2.IMREAD_COLOR)
img250 = cv2.imdecode(data['observation.images.table'][250], cv2.IMREAD_COLOR)
img300 = cv2.imdecode(data['observation.images.table'][300], cv2.IMREAD_COLOR)


print(f'100 frame state : {data['observation.state'][100]}')
print(f'150 frame state : {data['observation.state'][150]}')
print(f'200 frame state : {data['observation.state'][200]}')
print(f'250 frame state : {data['observation.state'][250]}')
print(f'300 frame state : {data['observation.state'][300]}')

cv2.imwrite('100.png', img100)
cv2.imwrite('150.png', img150)
cv2.imwrite('200.png', img200)
cv2.imwrite('250.png', img250)
cv2.imwrite('300.png', img300)
