---
title: How we got Audio Localization to work
layout: default
---
[back](./)

Posted on Apr 27, 2017

## Audio Localization

We are working on a conference call robot that can both localize and identify a speaker using their voice.  After spending the first two weeks of project delving into the theory behind mel-frequency cepstral coefficients and hidden markov models for speaker identification, we focused this week on audio localization.  We are using two microphones connected to a Raspberry Pi and a command-line tool called `amixer` for audio data collectio

## Initial Data Collection

We spent a good amount of time collecting audio data for localization and training our HMMs later on. We taped the microphones attached to the Raspberry Pi onto a flat table a foot apart from each other. We measured out two feet of string and taped it to the table halfway between the mics to try and keep the speaker about two feet away from the center of the microphon.

For speaker recognition data, we recorded the three of us and two other Oliners reading out a series of phonetic pangrams, phrases that try to encompass all of the sounds found in the English language. We also read part of an article about elephants aloud to collect a bulk of data since the pangrams are fairly short. We plan to train the models on the pangrams first and may end up using the much larger article files later on if the smaller files don’t prove to be diverse enough to identify each speaker.

To collect data for audio localization, we recorded one of us speaking for 2 seconds at each of 0, 45, 90, -45, and -90 degrees, approximately 2 feet away from the center of the microphones.  We used this data to test whether we could accurately predict the angle of the speaker from the microphones.  This calculated angle will be used to turn the robot to face the current speaker during a conversation

## Troubleshooting

With help from some libraries we had found that did audio localization with microphones on a Neato/Roomba, we were able to start testing the localization data we collected. We correlated the signals coming in from each of the microphones to find the ITD (interaural time difference). We were hoping to find a significant delay between sound signals reaching the two microphones. We expected one channel to have to be moved pretty significantly in correlation to match the other channel. The differences were much smaller than we expected and differences that should’ve been negative showed up positive, even at extreme angles like 90 and -90 degrees. This ended up being an issue with the data so we retook it. 

We hadn’t tried to be vertically inline with the microphones the first time we took data so we repeated our previous tests but kept our head in line with the microphones this time to try and increase the ITD. This didn’t fix our issues so we tried increasing the sampling rate. The correlations were bigger this time but not consistently and there still weren’t any clear negative differences. After talking to our instructor we realized that we had recorded our previous audio over the network. We tried recording and changing the sampling frequency directly onto the RasPi and then saved the data onto a laptop. When we ran this data, the correlation issues went away save null values which didn’t really pose a problem. We continued testing with this data and it proved very successfu

## Theory

We took the resulting time shifts (ITD) and multiplied them by the sampling interval (1/Fs) and the speed of sound (C) to get the difference in distance the sound had to travel between the two microphones. We predict the angle of a speaker using the following equation

![Approximation](images/AnglesEsti.png)

```
d1^2 - d2^2 = 2*D*r*sin(theta)

d1 - d2 = D*sin(theta) if r = (d1+d2)/2

diff_d = d1 - d2 = ITD*C/Fs

theta = arcsin(diff_d/D)
```
## Results

We were able to successfully predict the angle of a speaker, as described above, by finding the most frequently occurring angle calculation over the duration of the audio clip.  The following graphs illustrate our angle calculations for a speaker approximately -90, -45, 0, 45, and 90 degrees from the microphones, respectively.  Although the results are somewhat noisy, finding the mode angle works reliably to predict the speaker’s approximate orientation.  Our next step is to filter the output of our ITD calculations programmatically to extract this mode angle.

![Prediction Results](images/AnglesPred.png)

[back](./)
