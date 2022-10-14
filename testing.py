
import pandas as pd


## Data manipulation ##
#df = df.drop('Unnamed: 0', axis=1)
#df['OBS_TIME_LOC'] = pd.to_datetime(df['Date'])
#df = df.set_index('OBS_TIME_LOC')
#df = df.resample('H').mean()
#df = df.ffill().bfill()
#df['time'] = df.index

## Create a TimeSeries, specifying the time and value columns ##
#s#eries = TimeSeries.from_dataframe(df, "time", "HT")

## Seperating data ##
#series = series[-250:-1]

## Set aside the last 36 months as a validation series ##
#train, val = series[:-36], series[-36:]


#model = NaiveDrift()

## Training model ##
#model.fit(train)

## Establishing prediction ##
#prediction = model.predict(200)

#data = prediction.pd_dataframe()
#df = data

#print(df)
#df['time'] = pd.to_datetime(df['time'])
#plt.plot(df.index, df.HT, linewidth=5,color="r")

#sns.lineplot(x=df.index[-200:-1], y=df.HT[-200:-1],linewidth=4.0, color='b',label="Previous Data")


#plt.show()






#print(prediction)