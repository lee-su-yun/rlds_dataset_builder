import numpy as np
import cv2
import matplotlib.pyplot as plt

path = '/data/piper_grape0626/pick_the_grape_and_put_it_to_the_basket/1057/episode.pickle'
data = np.load(path, allow_pickle=True)

img50 = cv2.imdecode(data['observation.images.table'][50], cv2.IMREAD_COLOR)
img60 = cv2.imdecode(data['observation.images.table'][60], cv2.IMREAD_COLOR)
img70 = cv2.imdecode(data['observation.images.table'][70], cv2.IMREAD_COLOR)
img80 = cv2.imdecode(data['observation.images.table'][80], cv2.IMREAD_COLOR)
img90 = cv2.imdecode(data['observation.images.table'][90], cv2.IMREAD_COLOR)
img100 = cv2.imdecode(data['observation.images.table'][100], cv2.IMREAD_COLOR)
img110 = cv2.imdecode(data['observation.images.table'][110], cv2.IMREAD_COLOR)
img120 = cv2.imdecode(data['observation.images.table'][120], cv2.IMREAD_COLOR)
img130 = cv2.imdecode(data['observation.images.table'][130], cv2.IMREAD_COLOR)
img140 = cv2.imdecode(data['observation.images.table'][140], cv2.IMREAD_COLOR)
img150 = cv2.imdecode(data['observation.images.table'][150], cv2.IMREAD_COLOR)
img160 = cv2.imdecode(data['observation.images.table'][160], cv2.IMREAD_COLOR)
img200 = cv2.imdecode(data['observation.images.table'][200], cv2.IMREAD_COLOR)

st50 = data['observation.state'][50]
st60 = data['observation.state'][60]
st70 = data['observation.state'][70]
st80 = data['observation.state'][80]
st90 = data['observation.state'][90]
st100 = data['observation.state'][100]
st110 = data['observation.state'][110]
st120 = data['observation.state'][120]
st130 = data['observation.state'][130]
st140 = data['observation.state'][140]
st150 = data['observation.state'][150]
st160 = data['observation.state'][160]
st200 = data['observation.state'][200]

print(f'50 frame state : {st50}')
print(f'60 frame state : {st60}')
print(f'70 frame state : {st70}')
print(f'80 frame state : {st80}')
print(f'90 frame state : {st90}')
print(f'100 frame state : {st100}')
print(f'110 frame state : {st110}')
print(f'120 frame state : {st120}')
print(f'130 frame state : {st130}')
print(f'140 frame state : {st140}')
print(f'150 frame state : {st150}')
print(f'160 frame state : {st160}')
print(f'200 frame state : {st200}')

cv2.imwrite('50.png', img50)
cv2.imwrite('60.png', img60)
cv2.imwrite('70.png', img70)
cv2.imwrite('80.png', img80)
cv2.imwrite('90.png', img90)
cv2.imwrite('100.png', img100)
cv2.imwrite('110.png', img110)
cv2.imwrite('120.png', img120)
cv2.imwrite('130.png', img130)
cv2.imwrite('140.png', img140)
cv2.imwrite('150.png', img150)
cv2.imwrite('160.png', img160)
cv2.imwrite('200.png', img200)
