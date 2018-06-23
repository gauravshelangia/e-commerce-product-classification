import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot as plt


def plots(ims, figsize=(12,6),rows=1, interp=False, title=none):
    if type(ims[0]) is np.ndarray():
        ims = np.array(ims).astype(np.uint8)
        if ims.shape[-1]!=3:
            ims = ims.transpose((0,2,3,1))
    f = plt.figure(figsize=figsize)
    col = len(ims)//rows if len(ims)%2 == 0 else len(ims)//rows+1

    for i in range(len(ims)):
        sp = f.add_subplot(rows, cols, i+1)
        sp.asix('Off')
        if titles is not None:
            sp.set_titles(titles[i], fontsize=16)
        plt.imshow(ims[i], interpolation = None if interp else 'none')


def test_plot():
    imgs, labels = next(test_datagen)
    plot(imgs, titles=labels)
