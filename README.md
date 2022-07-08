# Ordinal Regression PyTorch Hub



This is a GitHub repository containing some deep learning models for ordinal regression (with pre-trained weights) in the PyTorch Hub / Torch Hub format. Note that this repository is not going to be a comprehensive Hub for ordinal regression models but more of a way to quickly access models from a specific manuscript:

- Xintong Shi, Wenzhi Cao, and Sebastian Raschka 
  *Deep Neural Networks for Rank-Consistent Ordinal Regression Based On Conditional Probabilities.* [https://arxiv.org/abs/2111.08851](https://arxiv.org/abs/2111.08851) 

(More models may be added later, but I don't want to make any promises 😅.)



## PyTorch Hub / Torch Hub Resources

- For more information on (Py)Torch Hub, see the documentation at [https://pytorch.org/docs/stable/hub.html](https://pytorch.org/docs/stable/hub.html)



## Using the Models


You can load the model via the following syntax:

```python
import torch

model = torch.hub.load(
    "rasbt/ord-torchhub",
    model="resnet34_corn_afad",
    source='github',
    pretrained=True
)
```

Note that the pretrained versions may only perform well on images from the [AFAD](https://afad-dataset.github.io) dataset, which is the dataset that was used to train the models. For more usage examples and transfer learning instructions, please see the examples in [./examples](./examples).



## Which Models Are Currently Supported

- `"resnet34_corn_afad"` (an ordinal model trained via the [CORN](https://arxiv.org/abs/2111.08851) loss)
- `"resnet34_coral_afad"` (an ordinal model trained via the [CORAL](http://arxiv.org/abs/1901.07884) loss)
- `"resnet34_niu_afad"` (an ordinal model trained via [Niu et al.'s](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Niu_Ordinal_Regression_With_CVPR_2016_paper.pdf) loss)
- `"resnet34_crossentr_afad"` (a regular classifier trained via cross entropy loss)




## Training (Optional)

In case you want to reproduce the model training, you can find the respective instructions and files in the [`_train`](./_train) subfolder.
