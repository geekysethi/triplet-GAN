
# Triplet GAN

This is the code  for *Training Triplet Networks with GAN* ([arXiv](https://arxiv.org/abs/1704.02227)).

The code is based on *Improved Techniques with GAN* ([arXiv](https://arxiv.org/abs/1606.03498)) ([code](https://github.com/Sleepychord/ImprovedGAN-pytorch))

Official Code Repo - [Github](https://github.com/maciejzieba/tripletGAN)



## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install all required libraries.


```bash
pip install requirements.txt 
```

or

```bash

conda create --name <env> --file requirements.txt 
```



## Pretraining
For pretraining the Triplet GAN we have used *Improved Techniques with GAN* for without the triplet loss function.

For Pretraining for MNIST data use `pretrain.py --config_file configs/pretrain_gan.yml`
Results of pretraining are shown in Figure - 

## Training Triplet GAN

For training the triplet GAN after *pretraining* use `main.py --config_file configs/triplet_gan.yaml`


## Results

### Pretraining

Input Image             |  Ground Truth
:-------------------------:|:-------------------------:

<img src="images/pretrain_d.png" width="150" /> |  <img src="images/pretrain_g.png" width="150">


<!-- ![Drag Racing]() -->








<!-- ## Results



<table>
  <tr>
    <th></th>
    <th>Validation IOU</th>
    <th>Validation Loss</th>
  </tr>
  <tr>
    <td>Upsample</td>
    <td>0.752</td>
    <td>0.144</td>
  </tr>
  <tr>
    <td>Convtrans2d</td>
    <td>0.783</td>
    <td>0.134</td>
  </tr>
  <tr>
    <td>Skip Connections</td>
    <td>0.743</td>
    <td>0.149</td>
  </tr>
</table>

### Training IOU Plot

<img src="images/train_acc.png" width="500">


### Validation IOU Plot
<img src="images/val_acc.png" width="500">


### Training Error Plot
<img src="images/train_loss.png" width="500">

 -->


<!-- ### Validation Error Plot

<img src="images/tsne_1.png" width="500"> -->









## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.



