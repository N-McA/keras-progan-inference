
# Keras inference for "Progressive Growing of GANs" with weights.

This module contains a ported version of the Celeb-A HQ weights from the official ProGAN repo [0].

Unlike the original implementation, it can be run easily on CPU.

It reproduces the results very well (median error of 1e-7 in final output).

It does not contain any of the logic required for training.

```python
from progan_layers import custom_objects
from keras.models import load_model

import matplotlib.pyplot as plt
%matplotlib inline
import numpy as np

progan = load_model('../weights/keras_progan.hdf5', custom_objects=custom_objects)
latents = np.random.RandomState(1000).randn(1000, 512) # 1000 random latents
latents = latents[[477, 56]] # hand-picked top-10 from the authors, to verify replication.

imgs = progan.predict(latents)
# Predicts in range -1 to 1
imgs = (imgs + 1) / 2 # Now 0 to 1
imgs = np.clip(imgs, 0, 1)

plt.imshow(imgs[0])
plt.axis('off')
plt.savefig('example.png')
```
Gives:
![alt text](notebooks/test.png)

[0] https://github.com/tkarras/progressive_growing_of_gans