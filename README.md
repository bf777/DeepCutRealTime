# DeepCutRealTime
Additional scripts for DeepLabCut to enable real-time tracking, as well as a GPIO test class adapted from pyftdi, as implemented here: https://doi.org/10.1101/482349

## Dependencies
Please note that these scripts were compiled to work with **DeepLabCut 1.11 only**; this is not the current version of DeepLabCut (2.0.x). Please follow the instructions for installing DeepLabCut 1.11 on the page below. Updates to integrate these scripts with DeepLabCut 2.0.x will come soon.

After installing [DeepLabCut](https://github.com/AlexEMG/DeepLabCut), install the dependencies for DeepCutRealTime with `pip install opencv-python pyftdi imutils`.

To set up pyftdi, follow the instructions [here](https://github.com/eblot/pyftdi).

## Usage
1. Clone this repo with `git clone https://github.com/bf777/DeepCutRealTime.git`.
1. Copy `AnalyzeVideos_streamLocalThreadedLED.py` into the `Analysis-tools` folder in `DeepLabCut`.
2. Copy `led_test.py` and `myconfig_stream.py` into your `DeepLabCut` base level folder.

## Credits
DeepCutRealTime is an adaptation and extension of [DeepLabCut](https://github.com/AlexEMG/DeepLabCut), which is covered by the [GNU Lesser General Public License v3.0](https://github.com/AlexEMG/DeepLabCut/blob/master/LICENSE). 

`AnalyzeVideos_streamLocalThreadedLED.py` is adapted from [AnalyzeVideos.py](https://github.com/AlexEMG/DeepLabCut/blob/master/Analysis-tools/AnalyzeVideos.py) and [MakingLabeledVideo.py](https://github.com/AlexEMG/DeepLabCut/blob/master/Analysis-tools/MakingLabeledVideo.py) in DeepLabCut.

`myconfig_stream.py` is adapted from [myconfig_analysis.py](https://github.com/AlexEMG/DeepLabCut/blob/master/myconfig.py) in DeepLabCut.

`led_test.py` is adapated from [gpio.py](https://github.com/eblot/pyftdi/blob/master/pyftdi/tests/gpio.py) in [pyftdi](https://github.com/eblot/pyftdi), which is covered by the [MIT License](https://opensource.org/licenses/MIT).
