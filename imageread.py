import numpy as np
import cv2
img = np.load('data/train/fake/IMG_0.npy')
print(img.shape)
cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
