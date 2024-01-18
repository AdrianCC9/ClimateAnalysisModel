import csv

with open('Q:\\Users\\adria\\PycharmProjects\\Personal\\Projects\\ClimateAnalysisModel\\data\\CMIP5A_data\\CHIP5A_data.csv', newline='\n') as f:
  reader = csv.reader(f)
  for row in reader:
    # do something here with `row`
    break