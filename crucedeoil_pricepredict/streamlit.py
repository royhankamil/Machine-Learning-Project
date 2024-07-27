import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
from scipy.stats import zscore as zs
import tensorflow as tf
from keras.models import load_model
from tensorflow.keras.layers import Dense, Dropout, LSTM, GRU
from keras.callbacks import EarlyStopping
from keras.optimizers import SGD
import os

# Menampilkan path file saat ini
current_file_path = os.path.abspath(__file__)
current_directory = os.path.dirname(current_file_path)

# Membaca dataset
ds_predict = pd.read_csv(os.path.join(current_directory, 'dataset', 'ds_predict0019.csv'))
ds_predict['Date'] = pd.to_datetime(ds_predict['Date'])
ds_predict.set_index('Date', inplace=True)

st.title('Oil Price Prediction')
st.write("Check correlation")

# Visualisasi Data Seaborn
# Membuat figure heatmap
fig, ax = plt.subplots(figsize=(10, 8))

# Generate heatmap hanya untuk kolom numerik
numerical_cols = ds_predict.select_dtypes(include=[np.number]).columns
sns.heatmap(ds_predict[numerical_cols].corr(), cmap='YlGnBu', annot=True, fmt='.3f', annot_kws={"fontsize": 9}, vmin=-1)

# Menambahkan Judul
ax.set_title('Heatmap Korelasi Antar Kolom')

# Rotasi axis-y agar mudah dibaca
plt.xticks(rotation=45, ha='right')

# Show the heatmap
st.write("Heatmap Korelasi Antar Kolom")
st.pyplot(fig)

# Rolling Mean
data = ds_predict
ds_predict_list = data.columns

fig = plt.figure(figsize = (24,24))
plt.style.use('fivethirtyeight')
for i in range(4):
    ax = fig.add_subplot(4,1,i+1)
    ax.plot(data.iloc[:,i], label=ds_predict_list[i])
    data.iloc[:,i].rolling(100).mean().plot(label='Rolling Mean')
    ax.set_title(ds_predict_list[i])
    ax.set_xlabel('Date')
    ax.set_ylabel('Price')
    plt.legend()
fig.tight_layout(pad=3.0)
st.write("Rolling Mean dari Setiap Kolom")
st.pyplot(fig)

fig = plt.figure(figsize = (24,24))
plt.style.use('fivethirtyeight')
for i in range(4):
    ax = fig.add_subplot(4,1,i+1)
    sns.histplot(data.iloc[:,i], label=ds_predict_list[i], kde=True, color='dodgerblue').set_title("Distribusi Plot dari {}".format(ds_predict_list[i]), axes=ax)
    ax.tick_params(labelsize=12)
    plt.legend()
fig.tight_layout(pad=3.0)
st.write("Distribusi Plot dari Setiap Kolom")
st.pyplot(fig)

# Uji Normalitas
fig, ax = plt.subplots(figsize=(8, 5))
ax.axvline(x=np.mean(ds_predict['crudeoil_price']), c='red', ls='--', label='mean')
ax.axvline(x=np.percentile(ds_predict['crudeoil_price'],25),c='green', ls='--', label = '25th percentile:Q1')
ax.axvline(x=np.percentile(ds_predict['crudeoil_price'],75),c='orange', ls='--',label = '75th percentile:Q3' )
sns.histplot(ds_predict['crudeoil_price'], bins=30, kde=True, color='dodgerblue', alpha=0.3, edgecolor='none')

ax.set_title('Histogram and Normal Distribution Fit')
ax.set_xlabel('Price')
ax.set_ylabel('Frequency')
plt.legend()
st.write("Histogram dan Uji Normalitas Distribusi Harga Minyak Mentah")
st.pyplot(fig)

# Uji Kolmogorov Smirnov (ktest)
stat, p = stats.normaltest(ds_predict['crudeoil_price'])
st.write("Kolmogorov-Smirnov test : stat=%.3f, p=%.3f" % (stat, p))

# Menentukan hasil uji
alpha = 0.05
if p > alpha:
    st.write("Sample data berasal dari distribusi normal (Tidak bisa menolak H0)")
else:
    st.write("Sample data tidak berasal dari distribusi normal (Menolak H0)")

# Dataframe sementara
temp_data = ds_predict.copy()
temp_data['Year'] = temp_data.index.year

# Box Plot Year Wise WTI
fig, ax = plt.subplots(figsize=(15,8))
sns.boxplot(x=temp_data['Year'], y=temp_data['crudeoil_price'], ax=ax)
ax.set_title('Box Plot Year WTI', fontsize=30)
st.write("Box Plot Harga WTI per Tahun")
st.pyplot(fig)

# Seasonality decomposition
cop = ds_predict[['crudeoil_price']].copy()
decompose_result= seasonal_decompose(cop, model='additive', period=12)

fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(15, 10), sharex=True)
decompose_result.observed.plot(ax=ax1, legend=False)
ax1.set_ylabel('Observed')
decompose_result.trend.plot(ax=ax2, legend=False)
ax2.set_ylabel('Trend')
decompose_result.seasonal.plot(ax=ax3, legend=False)
ax3.set_ylabel('Seasonal')
decompose_result.resid.plot(ax=ax4, legend=False)
ax4.set_ylabel('Residual')
plt.suptitle('Decomposisi Deret Waktu')
st.write("Decomposisi Deret Waktu Harga Minyak Mentah")
st.pyplot(fig)

# Mengecek Series Lag
price_series = ds_predict['crudeoil_price']
n_lags = 8
cols = [price_series]
for i in range(1, (n_lags + 1)):
    cols.append(price_series.shift(i))

df = pd.concat(cols, axis=1)
cols = ['t+1']
for i in range(1, (n_lags + 1)):
    cols.append('t-' + str(i))
df.columns = cols

fig, axs = plt.subplots(2, 4, figsize=(25,15))
for i in range(1, (n_lags + 1)):
    ax = axs[(i-1)//4, (i-1)%4]
    ax.set_title('t+1 vs t-' + str(i))
    ax.scatter(x=df['t+1'].values, y=df['t-'+str(i)].values)
plt.tight_layout(pad=2)
st.write("Plot Series Lag Harga Minyak Mentah")
st.pyplot(fig)

fig, ax = plt.subplots(figsize=(12, 6))
plot_acf(price_series, lags=60, ax=ax)
plt.title('Autocorrelation Function')
st.write("Autocorrelation Function Harga Minyak Mentah")
st.pyplot(fig)

fig, ax = plt.subplots(figsize=(12, 6))
plot_pacf(price_series, lags=60, ax=ax)
plt.title('Partial Autocorrelation Function')
st.write("Partial Autocorrelation Function Harga Minyak Mentah")
st.pyplot(fig)

# Scaling data untuk analisa bivariate
scdata = MinMaxScaler(feature_range=(0,1))
ds_predict_scaled = scdata.fit_transform(ds_predict)
ds_predict_scaled[:2]

# Membandingkan pergerakan harga minyak mentah dengan variabel lainnya
col_names = ds_predict.columns[:-1]
scaled_features = ds_predict_scaled[:,:-1]
fig = plt.figure(figsize=(24, 24))
plt.style.use('fivethirtyeight')
for i in range(3):
    ax = fig.add_subplot(3,1,i+1)
    ax.plot(ds_predict.index, scaled_features[:,i], label=col_names[i], c='blue')
    ax.plot(ds_predict.index, ds_predict_scaled[:,3], label='crudeoil_price', c='orange')
    ax.set_title('crudeoil_price & ' + col_names[i])
    ax.set_xlabel('Date')
    ax.set_ylabel('Price')
    plt.legend()
fig.tight_layout(pad=3.0)
st.write("Perbandingan Pergerakan Harga Minyak Mentah dengan Variabel Lainnya")
st.pyplot(fig)

cop1= ds_predict_scaled.copy()
fig, ax = plt.subplots(figsize = (20,6))

# Plot harga emas, snp500, us dollar, dan minyak mentah
ax.plot(ds_predict.index, cop1[:, 0], label='Gold Price', linewidth=1)
ax.plot(ds_predict.index, cop1[:, 1], label='S&P 500 Price', linewidth=1)
ax.plot(ds_predict.index, cop1[:, 2], label='US Dollar Index Price', linewidth=1)
ax.plot(ds_predict.index, cop1[:, 3], label='Crude Oil Price', linewidth=1)

# Highlight Era Covid dimulai pada tahun 2020
ax.axvline(pd.Timestamp('2020-01-01'), color='r', linestyle='--', linewidth=2, label='Era Covid')

ax.legend()
ax.grid()
ax.set_title('Harga Emas, S&P 500, US Dollar Index, dan Minyak Mentah', fontweight='bold')
ax.set_xlabel('Date', fontweight='bold')
ax.set_ylabel('Price (USD)', fontweight='bold')

# Rotate date labels for better readability
plt.xticks(rotation=45)

st.write("Perbandingan Harga Emas, S&P 500, US Dollar Index, dan Minyak Mentah")
st.pyplot(fig)

after_fs = ds_predict[['crudeoil_price']]
zscore = np.abs(zs(after_fs['crudeoil_price']))
ds_predict_zscore = after_fs.copy()
ds_predict_zscore['zscore'] = zscore
ds_predict_zscore = ds_predict_zscore.loc[ds_predict_zscore['zscore']<2.5, ['crudeoil_price']]
ds_predict_zscore = ds_predict_zscore.sort_index()

# Cek Keakuratan data : Memastikan bahwa data tersebut akurat dan sesuai dengan realita
ds_predict_list = ['crudeoil_price']
nilai_negatif = ds_predict_zscore[ds_predict_list] < 0
total_nn = nilai_negatif.sum().sum()

if total_nn > 0:
    for index, row in ds_predict_zscore.iterrows():
        for col_index, value in enumerate(row[ds_predict_list]):
            if nilai_negatif.loc[index, ds_predict_list[col_index]]:
                st.write(f"Nilai negatif ditemukan di kolom '{ds_predict_list[col_index]}': {value}, pada tanggal '{index}'")

dataset = ds_predict_zscore
tscv = TimeSeriesSplit(n_splits = 5, test_size = None, gap=0)
dataset = dataset.sort_index()
fig, axs = plt.subplots(5,1, figsize=(15,15), sharex=True)
fold = 0
for train_index, val_index in tscv.split(dataset):
    train = dataset.iloc[train_index]
    test = dataset.iloc[val_index]
    train['crudeoil_price'].plot(ax=axs[fold], label='Training Set', title=f'Data Train/Test Split Fold {fold}')
    test['crudeoil_price'].plot(ax=axs[fold], label='Train')
    axs[fold].axvline(test.index.min(), color='black', ls='--')
    fold +=1
st.write("Data Train/Test Split pada Setiap Fold")
st.pyplot(fig)

st.write("Data Train/Test Split pada Setiap Fold")
