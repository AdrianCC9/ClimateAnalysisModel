import csv
import requests
import json

api_ednpoint2 = 'https://api.weather.gc.ca/collections/climate:cmip5:historical:annual:absolute?lang=en&f=csv'

response = requests.get(api_ednpoint2)

if response.status_code == 200:
    file_path = ''
    print(response.text())

else:
    print(f"Error: {response.status_code}")