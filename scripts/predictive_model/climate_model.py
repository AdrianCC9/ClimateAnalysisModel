import pandas as pd
from statsmodels.tsa.arima.model import ARIMA


df = pd.read_csv(r'Q:\Users\adria\PycharmProjects\Personal\Projects\ClimateAnalysisModel\data\clean_data\clean_data.csv')
df['start'] = pd.to_datetime(df['start'], format='%Y')
df.set_index('start', inplace=True)
df.sort_index(inplace=True)



end_year = 1991
future_years = pd.date_range(end_year, 2050, freq='Y')
df_future = pd.DataFrame(index=future_years, columns=df.columns)
df_extended = pd.concat([df, df_future])
df_extended['tavg'] = df_extended['tavg'].interpolate()


p, d, q = 1, 1, 1
model = ARIMA(df_extended['tavg'], order=(p, d, q))
result = model.fit()

def predict_temperature(year):
    try:
        user_year = int(year)
        start_year = 1961
        end_year = 1991

        if user_year < start_year or user_year > 2050:
            return f"Year must be between years {start_year} and 2050."

        steps = user_year - end_year
        forecast = result.get_forecast(steps)
        prediction = forecast.predicted_mean.iloc[-1]
        return prediction

    except Exception as e:
        return f"An error occurred: {e}"


try:
    user_year = input('Please Enter the year you would like the average annual temperature for: ')
    prediction = predict_temperature(user_year)
    print(f"The Average Temperature for {user_year} is {prediction:.2f}°C")
except ValueError:
    print("Invalid input. Please enter a valid year.")

