from utils.model_implementation import *

import numpy as np
import cv2

handler = Handler('./2d106det', 0, ctx_id=-1,
                  det_size=120)  # чем меньше размер картинки тем быстрее инференс, но точность ниже, норм при 120..

source = cv2.imread('images/source.jpg')
cv2.resize(source, (640, 480))
target = cv2.imread('images/1.jpg')
cv2.resize(target, (640, 480))
preds_source = handler.get(source, get_all=False)
preds_target = handler.get(target, get_all=False)

color = (200, 160, 75)

for pred in preds_source:
    pred = np.round(pred).astype(np.int)
    for i in range(pred.shape[0]):
        p = tuple(pred[i])
        cv2.circle(source, p, 1, color, 1, cv2.LINE_AA)


for pred in preds_target:
    pred = np.round(pred).astype(np.int)
    for i in range(pred.shape[0]):
        p = tuple(pred[i])
        cv2.circle(target, p, 1, color, 1, cv2.LINE_AA)
print(len(preds_source[0]))
cv2.imshow('photo', target)
cv2.imshow('photo1', source)


while True:
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()