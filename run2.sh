#!/bin/bash
python3 -u run_dp_one.py 20080129 --eta 106.0 --no-gif 
python3 -u run_dp_one.py 20080218 --eta 94.5 --no-gif
python3 -u run_dp_one.py 20080424 --eta 92.5 --no-gif
python3 -u run_dp_one.py 20081214 --eta 105.0 --no-gif


#make GIF
#python3 make_weather_gif.py 20080129 --pkl result_20080129.pkl --eta 106.0 --field hs
#python3 make_weather_gif.py 20080129 --pkl result_20080129.pkl --eta 106.0 --field hs --out new_weather_20080129_eta106.gif