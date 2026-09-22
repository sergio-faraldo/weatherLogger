# WeatherLogger

Simple set of scripts that reads the screen of a vevor weather station and records everything into a csv file. It takes multiple photos with a webcam and does some basic recognition using OpenCV.

The screen is located using a basic template image, and split into sections for each part of the screen, then for each section, rectangles are drawn for each bit of the 7-segment display. Based on those rectangles, the 7-segment display is interpreted.

For the direction of the wind, the compass dispaly is interpreted in a similar manner.


This software is running on a raspberry pi, and readings are uploaded to this repo periodically. An HTML, served through github pages shows the wind speed data (the only thing I really care about right now). The HTML itself is very much a WIP.

For a cron-based recorder, add this entry with `crontab -e`:

```cron
* * * * * /path/to/weatherLogger/run-record.sh >> /path/to/weatherLogger/weatherlogger.log 2>&1
```

`run-record.sh` uses `flock` to prevent overlapping camera processes and kills a recording that runs longer than 50 seconds. It requires `flock`, GNU `timeout`, and `uv` to be available in cron's `PATH`.
