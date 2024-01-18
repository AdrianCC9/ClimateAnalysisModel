import csv
import requests
import json

#AHCCD Data Request

# replacing 'api_endpoint' and 'parameters' with actual endpoint and parameters
api_endpoint = "https://api.weather.gc.ca/collections/ahccd-annual?lang=en"

station_ids = ['1140876','8101605','2202101','3011880','3044533','4010879','50309J6','503B1ER','506B047','6115525']

for station_id in station_ids:
    parameters = {'datetime': '2013-01-01/2023-01-01','station_id__id_station': station_id,'f': 'json'}
    response = requests.get(api_endpoint, params=parameters)

    print(f"Station ID: {station_id}, Status Code: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        with open(f'ahccd_data_{station_id}.json', 'w') as file:
            json.dump(data, file, indent=4)

    else:
        print(f"failed to fetch data for station {station_id}: {response.status_code}")

