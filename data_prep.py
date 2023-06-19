import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from skimage import img_as_ubyte
from sklearn.model_selection import train_test_split

def prepare_train_dataset(dataframe,LOCATION):

    y = dataframe['labels']
    x = dataframe.copy()
    x.drop('labels',axis=1,inplace=True)

    X = x.to_numpy()
    Y = y.to_numpy()

    X = np.reshape(X,(len(x),20,20,3))
    print("Shape of Image-Dataset: {}".format(X.shape))

    REAL = 'real/'
    FAKE = 'fake/'
    
    IMGCNT = 0
    FAKECNT = 0
    REALCNT = 0
    for image in X:
        if(Y[IMGCNT] == 0):
            np.save(os.path.join(LOCATION,FAKE,'IMG_'+str(FAKECNT)),image)
            FAKECNT = FAKECNT + 1
        else:
            np.save(os.path.join(LOCATION,REAL,'IMG_'+str(REALCNT)),image)
            REALCNT = REALCNT + 1
        # print("saving image{}...".format(cnt))
        IMGCNT = IMGCNT + 1

    print("SUCCESS...")
    print("{} Images saved".format(IMGCNT))
    print("{} Fake-Images saved".format(FAKECNT))
    print("{} Real-Images saved".format(REALCNT))


def prepare_test_dataset(dataframe,LOCATION):
    x = dataframe.copy()
    x.drop('id',axis=1,inplace=True)

    X = x.to_numpy()
    X = np.reshape(X,(len(x),20,20,3))
    print("Shape of Test-Image-Dataset: {}".format(X.shape))

    IMGCNT = 0
    for image in X:
        np.save(os.path.join(LOCATION,'IMG_'+str(IMGCNT)),image)
        # print("saving image{}...".format(cnt))
        IMGCNT = IMGCNT + 1

    print("SUCCESS...")
    print("{} Test-Images saved".format(IMGCNT))



# There are no null values in this dataframe:

# Creating Train and Validation dataset:
df = pd.read_csv('train.csv')
df_train, df_val = train_test_split(df, test_size=.1, random_state=42)

# Test dataset:
df_test = pd.read_csv('test.csv')


# Prepapring Dataset
ROOT_DIR = 'data/'
TRAIN = 'train/'
TEST = 'test/'
VAL = 'val/'

prepare_train_dataset(df_train,os.path.join(ROOT_DIR,TRAIN))
prepare_train_dataset(df_val,os.path.join(ROOT_DIR,VAL))
prepare_test_dataset(df_test,os.path.join(ROOT_DIR,TEST))









# print(cnt)

# np.save('img0',ImageDataset[0])
# cv2.imshow('Image',ImageDataset[0])
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# cv2.imwrite('img_0.jpg',ImageDataset[0],cv2.IMWRITE_EXR_TYPE_FLOAT)


# with torch.no_grad():
#     X_train = torch.from_numpy(X)
#     y_train = torch.from_numpy(Y)

# # our array of images...
# X_train = torch.reshape(X_train,(len(x),20,20,3))


# plt.imshow(cv2.cvtColor(X_train[0], cv2.COLOR_BGR2RGB))
# plt.show()