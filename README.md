# Welcome to MTracker

**MTracker** is a tool for automatic splining tongue shapes in ultrasound images by harnessing the power of deep convolutional neural networks. It is developed at the Department of Linguistics, University of Michgian to address the need of spining a large-scale ultrasound project. MTracker also allows for human correction by interfacing with the [GetContours Suite](https://github.com/mktiede/GetContours) developed by Mark Tiede at Haskins Laboratory.


## About MTracker

You can refer to this paper for more information. [A CNN-based tool for automatic tongue contour tracking in ultrasound images](https://arxiv.org/abs/1907.10210). 
[This paper was peer-reviewed and accepted by Interspeech 2019 but was withdrawn due to my unability to travel to the conference].

MTracker was developed by Jian Zhu, [Will Styler](http://savethevowels.org/will/), and Ian Calloway at the University of Michigan, on the basis of data collected with [Patrice Beddor](https://sites.lsa.umich.edu/beddor/) and [Andries Coetzee](https://sites.lsa.umich.edu/coetzee/).  


It was first described in [a poster](https://github.com/lingjzhu/mtracker.github.io/blob/master/mtracker_asa2018_poster%202.pdf) at the [175th Meeting of the Acoustical Society of America in Minneapolis](https://acousticalsociety.org/program-of-175th-meeting-of-the-acoustical-society-of-america/).  The tools, and trained model, will be made available below.

This work is inspired by a [Deep Learning Tutorial for Kaggle Ultrasound Nerve Segmentation competition](https://github.com/jocicmarko/ultrasound-nerve-segmentation). 



### Attribution and Citation

For now, you can cite this software as:

```
@article{zhu2019cnn,
  title={A CNN-based tool for automatic tongue contour tracking in ultrasound images},
  author={Zhu, Jian and Styler, Will and Calloway, Ian},
  journal={arXiv preprint arXiv:1907.10210},
  year={2019}
}
```

Or

Zhu, J., Styler, W., and Calloway, I. C. (2018). Automatic tongue contour extraction in ultrasound images with convolutional neural networks. The Journal of the Acoustical Society of America, 143(3):1966–1966. [https://doi.org/10.1121/1.5036466](https://doi.org/10.1121/1.5036466)

You can also send people to this website, [https://github.com/lingjzhu/mtracker.github.io/](https://github.com/lingjzhu/mtracker.github.io/).

## Installing MTracker

To install MTracker for your own use, follow the instructions below:

### Dependencies

- Python >= 3.6
- Keras >= 2.0.8
- Tensorflow/Tensorflow-gpu==1.14, CUDA and CUDNN（if you need GPU support）
- Scikit-image 1.3.0
- Imageio
- Praatio
- tqdm

Installing Tensorflow can be a painful process. Please refer to the official documentation of Tensorflow and Keras for installation guides.

### Downloading the trained model

#### The following models are the models that produce some of the results in Figure 2.
- [dense_aug.hdf5](https://drive.google.com/file/d/1r1PnFw8KihmJnhXLUsKiw9BxJTSm3PhM/view?usp=sharing) -- Dense UNet model trained with 50% of the training data with augmentation
- [unet_aug.hdf5](https://drive.google.com/file/d/181unum8CBgpzoGs-4KFqOwj7k7ELlK3R/view?usp=sharing) -- UNet model trained with 50% of the training data with augmentation


### Preparing your data

A muted ultrasound video for demonstration purpose only is available in the "demo" folder.

The test data are available [here](https://drive.google.com/drive/folders/14x-lG-gDZE-3qzMI5MPYJ7q2-8vu5Xc1?usp=sharing). Please do not redistribute.

### Quickstart guide
#### tracking tongue contours in a video
```
python track_video.py -v ./demo/demo_video.mp4 -t du -m ./model/dense_aug.hdf5 -o ./demo/out.csv -f ./demo -n 5
```

#### tracking tongue contours in an image sequence
```
python track_frames.py -i ./test_data/test_x -t du -m ./model/dense_aug.hdf5 -o ./demo/out.csv -f ./demo -n 5
```
Note. Both scripts accept the following arguments. All arguments come with default values. So the script can be run directly using "python track_video.py" or "python track_frames.py"
```
   -v path to the video file;
   
   -i path to the images

   -t model type; du for Dense UNet and u for UNet;

   -m path to model;

   -o path to the output folder;

   -n plot the every Nth predicted contour on its corresponding raw frame for sanity check;

   -b specify the boundy for cropping the Region of Interest;
```


## Disclaimer

This software, although in production use by the authors and others at the University of Michigan, may still have bugs and quirks, alongside the difficulties and provisos which are described throughout the documentation. 

By using this software, you acknowledge:

* That you understand that this software does not produce perfect camera-ready data, and that all results should be hand-checked for sanity's sake, or at the very least, noise should be taken into account.

* That you understand that this software is a work in progress which may contain bugs.  Future versions will be released, and bug fixes (and additions) will not necessarily be advertised.

* That this software may break with future updates of the various dependencies, and that the authors are not required to repair the package when that happens.

* That you understand that the authors are not required or necessarily available to fix bugs which are encountered (although you're welcome to submit bug reports to Jian Zhu (lingjzhu@umich.edu), if needed), nor to modify the software to your needs.

* That you will acknowledge the authors of the software if you use, modify, fork, or re-use the code in your future work.  

* That rather than re-distributing this software to other researchers, you will instead advise them to download the latest version from the website.

... and, most importantly:

* That neither the authors, our collaborators, nor the the University of Michigan on the whole, are responsible for the results obtained from the proper or improper usage of the software, and that the software is provided as-is, as a service to our fellow linguists.

All that said, thanks for using our software, and we hope it works wonderfully for you!

## Support or Contact

Please contact Jian Zhu (lingjzhu@umich.edu), Will Styler (will@savethevowels.org) and Ian Calloway (iccallow@umich.edu) for support.

## References

[^1]: Ronneberger et al. 2015, U-Net: Convolutional Networks for Biomedical Image Segmentation, DOI:10.1007/978-3-319-24574-4_28
