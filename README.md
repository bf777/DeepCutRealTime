# DeepCutRealTime
Additional scripts for DeepLabCut to enable real-time tracking, as well as a GPIO test class adapted from pyftdi.

## Dependencies
After installing [DeepLabCut](https://github.com/AlexEMG/DeepLabCut), install with `pip install opencv-python pyftdi imutils`.

To set up pyftdi, follow the instructions [here](https://github.com/eblot/pyftdi).

## Usage
1. Clone this repo with `git clone https://github.com/eblot/pyftdi`.
1. Copy `AnalyzeVideos_streamLocalThreadedLED.py` into the `Analysis-tools` folder in `DeepLabCut`.
2. Copy `led_test.py` and `myconfig_stream.py` into your `DeepLabCut` base level folder.

## Credits
DeepCutRealTime is an adaptation and extension of [DeepLabCut](https://github.com/AlexEMG/DeepLabCut), which is covered by the [GNU Lesser General Public License v3.0](https://github.com/AlexEMG/DeepLabCut/blob/master/LICENSE). 

`AnalyzeVideos_streamLocalThreadedLED.py` is adapted from [AnalyzeVideos.py](https://github.com/AlexEMG/DeepLabCut/blob/master/Analysis-tools/AnalyzeVideos.py) and [MakingLabeledVideo.py](https://github.com/AlexEMG/DeepLabCut/blob/master/Analysis-tools/MakingLabeledVideo.py) in DeepLabCut.

`myconfig_stream.py` is adapted from [myconfig_analysis.py](https://github.com/AlexEMG/DeepLabCut/blob/master/myconfig.py) in DeepLabCut.

`led_test.py` is adapated from [gpio.py](https://github.com/eblot/pyftdi/blob/master/pyftdi/tests/gpio.py) in [pyftdi](https://github.com/eblot/pyftdi), which is covered by the [MIT License](https://opensource.org/licenses/MIT).
