# speaker_recognizer_robot
Using MFCCs and HMMs to recognize a speaker

If you are looking for data or wav files to run the recognizer, request access for the drive here: https://drive.google.com/drive/u/0/folders/0BwXkbvw4LufrUzVNSUxiX1VEQUE.

## Speaker Recognizer

To run the code, simply

`roscore`

`rosrun speaker_recognizer recognizer.py`

## Audio Localization

We used a Raspberry Pi with two microphones to localize using a speaker's voice.

To record wav files on your RasPi, use amixer with the steps outlined in this [script](https://github.com/CirrusLogic/wiki-content/blob/master/scripts/Record_from_lineIn_Micbias.sh).

To copy your recorded file from the RasPi to your machine, use the secure copy command.  This file can now be used as input to the `angle_from_audio` function in audio_localizer.py, which calculates the angle of the speaker from the robot.

`scp pi@[IP Address]:~/test.wav .`

## Dependencies

1. python_speech_features

      Install it from pypi:    `pip install python_speech_features`
