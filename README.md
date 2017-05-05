# speaker_recognizer_robot
Using MFCCs and HMMs to recognize a speaker

## Dependencies

1. python_speech_features

      Install it from pypi:    `pip install python_speech_features`


## Audio Localization

We used a Raspberry Pi with two microphones to localize using a speaker's voice.

To record wav files on your RasPi, use amixer with the steps outlined in this [script](https://github.com/CirrusLogic/wiki-content/blob/master/scripts/Record_from_lineIn_Micbias.sh).

To copy your recorded file from the RasPi to your machine, use the secure copy command.  This file can now be used as input to the `angle_from_audio` function in audio_localizer.py, which calculates the angle of the speaker from the robot.

`scp pi@[IP Address]:~/test.wav .`