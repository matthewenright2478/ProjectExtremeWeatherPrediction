import subprocess
from multiprocessing import freeze_support
import pandas as pd
from darts import TimeSeries
import matplotlib.pyplot as plt
from darts.models import RegressionModel, ARIMA, ExponentialSmoothing, Theta, FourTheta, Prophet, Croston, RegressionModel
from darts.models import FFT, KalmanForecaster, RandomForest, TCNModel, TransformerModel
#from darts import GaussianLikelihood
from darts.utils.likelihood_models import GaussianLikelihood
from darts.models.forecasting.nbeats import NBEATSModel
from darts.models.forecasting.tft_model import TFTModel
from darts.models.forecasting.tcn_model import TCNModel
from darts.models.forecasting.nhits import NHiTSModel
from darts.models.forecasting.block_rnn_model import BlockRNNModel
from darts.models.forecasting.transformer_model import TransformerModel
from darts.models.forecasting.rnn_model import RNNModel
from darts.models.forecasting.random_forest import RandomForest
from darts.models.forecasting.gradient_boosted_model import LightGBMModel
from darts.models.forecasting.catboost_model import CatBoostModel

def darts(model):
    import functools
  #  import partial
    import subprocess
  #  subprocess.Popen = partial(subprocess.Popen, stdin=subprocess.DEVNULL)
    stdin = subprocess.DEVNULL
    ## Read a pandas DataFrame ##
    df = pd.read_csv("FILENAME.csv")

    ## Data manipulation ##
    df = df.drop('Unnamed: 0',axis=1)
    df['OBS_TIME_LOC'] = pd.to_datetime(df['Date'])
    df = df.set_index('OBS_TIME_LOC')
    df = df.resample('H').mean()
    df = df.ffill().bfill()
    df['time'] = df.index

    ## Create a TimeSeries, specifying the time and value columns ##
    series = TimeSeries.from_dataframe(df, "time", "HT")

    ## Seperating data ##
    series = series[-350:-1]

    ## Set aside the last 36 months as a validation series ##
    train, val = series[:-36], series[-36:]

    ## For ARIMA ##
    if model == "ARIMA":
        model = ARIMA()

    ## For Expotential smoothing ##
    if model == "ExponentialSmoothing":
        model = ExponentialSmoothing()

    ## Theta model ##
    if model == "Theta":
        model = Theta()

    ## FourTheta ##
    if model == "FourTheta":
        model = FourTheta()

    ## Prophet ##
    if model == "Prophet":
        model = Prophet()

    ## RegressionModel ##
    if model == "RegressionModel":
        model = RegressionModel(lags=300)

    ## Croston ##
    if model == "Croston":
        model = Croston()

    ## Croston ##
    if model == "Neural Basis Expansion Analysis":
            model = NBEATSModel(input_chunk_length=6,output_chunk_length=4)

    ## TFTModel _ requries future covarients ##
    #if model == "TFTModel":
      #  model = TFTModel(input_chunk_length=6, output_chunk_length=4)

    ## TCN Model ##
    if model == "TCNModel":
        model = TCNModel(input_chunk_length=300, output_chunk_length=40)

    ## N-Hits Model ##
    if model == "NHiTSModel":
        model = NHiTSModel(input_chunk_length=300, output_chunk_length=4)

    ## N-Hits Model ##
    if model == "BlockRNNModel":
        model = BlockRNNModel(input_chunk_length=5, output_chunk_length=3
                              ,n_epochs=25)

    ## N-Hits Model ##
    if model == "TransformerModel":
        model = TransformerModel(input_chunk_length=200, output_chunk_length=50,n_epochs=25)

    ## N-Hits Model ##
    if model == "RNNModel":
        model = RNNModel(input_chunk_length=2, model='LSTM', n_epochs=50)

     ## N-Hits Model ##
    if model == "RandomForest":
        model = RandomForest(lags=300)

     ## N-Hits Model ##
    if model == "LightGBMModel":
        model = LightGBMModel(lags=300)

     ## N-Hits Model ##
    if model == "CatBoostModel":
        model = CatBoostModel(lags=300)

     ## N-Hits Model ##
    if model == "CatBoostModel":
        model = CatBoostModel(lags=300)

    ## Training model ##
    model.fit(train)

    ## Establishing prediction ##
    prediction = model.predict(300)

    ## Sending data to focecast_In ##
    data = prediction.pd_dataframe()
    data.to_csv('forecast_In.csv')

    ## Sending data to forecast_out ##
    forecast_out = df
    forecast_out.to_csv('forecast_Out.csv')



