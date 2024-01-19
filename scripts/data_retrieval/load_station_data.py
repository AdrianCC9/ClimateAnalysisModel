import json

file_path = 'Q:\\Users\\adria\\PycharmProjects\\Personal\\Projects\\ClimateAnalysisModel\\data\\station_data\\station_data.json'

with open(file_path, 'r', encoding='utf-8') as file:
    data = json.load(file)

canada_stations = [station for station in data if station.get('country') == 'CA']
meteostat_ids = [station.get('id') for station in canada_stations]

for id in meteostat_ids:
    print(id)


