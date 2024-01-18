import requests
import pandas as pd
import io

items = ['503B1ER.1963?lang=en&f=json', '506B047.1953?lang=en&f=json', '6115525.1954?lang=en&f=json',
         '8101605.1914?lang=en&f=json', '1140876.1955?lang=en&f=json', '2202101.2016?lang=en&f=json',
         '3011880.1956?lang=en&f=json', '3044533.1957?lang=en&f=json', '4010879.1996?lang=en&f=json',
         '50309J6.1965?lang=en&f=json']

combined_df = pd.DataFrame()

base_url = 'https://api.weather.gc.ca/collections/ahccd-annual/items/'

for item in items:
    url = f'{base_url}{item}'
    response = requests.get(url)
    print(response.status_code)

    if response.status_code == 200:
        file_path = 'Q:\\Users\\adria\\PycharmProjects\\Personal\\Projects\\ClimateAnalysisModel\\data\\ahccd_data\\'

        with open(file_path) as file:
            file.write(response.text)
        print("Data downloaded successfully.")

else:
    print(f"Error: {response.status_code}")
