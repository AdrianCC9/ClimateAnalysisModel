import csv
import requests
import json

response = requests.get('https://api.weather.gc.ca/collections/climate:cmip5:historical:annual:absolute?lang=en?f=csv')
print(response.status_code)

if response.status_code == 200:
    file_path = 'Q:\\Users\\adria\\PycharmProjects\\Personal\\Projects\\ClimateAnalysisModel\\data\\CMIP5A_data\\CHIP5A_data.csv'

    with open(file_path, 'w') as file:
        file.write(response.text)
    print("Data downloaded successfully.")

else:
    print(f"Error: {response.status_code}")