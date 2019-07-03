# Welcome to MTracker

**MTracker** is a tool for automatic splining tongue shapes in ultrasound images by harnessing the power of deep convolutional neural networks. It is developed at the Department of Linguistics, University of Michgian to address the need of spining a large-scale ultrasound project. MTracker also allows for human correction by interfacing with the [GetContours Suite](https://github.com/mktiede/GetContours) developed by Mark Tiede at Haskins Laboratory.

## MTracker is currently being tested. The test data (4360 annotated images from two speakers) will also be released very soon!
### If you are interested in using MTracker in your research, you can send an email to Jian Zhu (lingjzhu@umich.edu). We will send a notification to you once MTracker is ready. Thanks very much for your interest.

## About MTracker

MTracker was developed by Jian Zhu, [Will Styler](http://savethevowels.org/will/), and Ian Calloway at the University of Michigan, on the basis of data collected with [Patrice Beddor](https://sites.lsa.umich.edu/beddor/) and [Andries Coetzee](https://sites.lsa.umich.edu/coetzee/).  

It was first described in [a poster](https://github.com/lingjzhu/mtracker.github.io/blob/master/mtracker_asa2018_poster%202.pdf) at the [175th Meeting of the Acoustical Society of America in Minneapolis](https://acousticalsociety.org/program-of-175th-meeting-of-the-acoustical-society-of-america/).  The tools, and trained model, will be made available below.

This work is inspired by a [Deep Learning Tutorial for Kaggle Ultrasound Nerve Segmentation competition](https://github.com/jocicmarko/ultrasound-nerve-segmentation). 

### Attribution and Citation

For now, until a peer-reviewed publication is available, you can cite this software as:

Zhu, J., Styler, W., and Calloway, I. C. (2018). Automatic tongue contour extraction in ultrasound images with convolutional neural networks. The Journal of the Acoustical Society of America, 143(3):1966–1966. [https://doi.org/10.1121/1.5036466](https://doi.org/10.1121/1.5036466)

You can also send people to this website, [https://github.com/lingjzhu/mtracker.github.io/](https://github.com/lingjzhu/mtracker.github.io/).

## Installing MTracker

To install MTracker for your own use, follow the instructions below:

### Dependencies

- Python >= 3.5
- Keras >= 2.0.8
- Tensorflow-gpu, CUDA and CUDNN（if you need GPU support）or Tensorflow
- Scikit-image 1.3.0
- Imageio
- Praatio
- tqdm

Installing Tensorflow can be a painful process. Please refer to the official documentation of Tensorflow and Keras for installation guides.

### Downloading the trained model

The pre-trained model can be found [here](https://drive.google.com/file/d/1GgUpTJ9riYAX9DPN0dHT-rIW4V67t1Ou/view?usp=sharing).

### Preparing your data

A muted ultrasound video for demonstration purpose only is available in the "demo" folder.


### Quickstart guide
- run python track_video.py -v ./demo/demo_video.py -t du -m path_to_model -o output_folder
- specify the paths to the input video, the pre-trained model and the output csv file.
   \*please also specify the area that should be cropped by inputing a four-element array.	
- each tongue contour represented by a 100-point coordinates, stored in a csv file
- every Nth frame will be ploted for sanity check

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
