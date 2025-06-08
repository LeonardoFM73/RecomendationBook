
# Laporan Proyek Machine Learning - [Leonardo Fajar Mardika]

## Domain Proyek

Kualitas udara merupakan salah satu aspek penting dalam kesehatan lingkungan dan manusia. Dalam beberapa dekade terakhir, peningkatan urbanisasi dan aktivitas industri menyebabkan peningkatan emisi polutan seperti CO, dan NO2 yang berdampak buruk terhadap kesehatan. Menurut WHO, polusi udara menyebabkan lebih dari 7 juta kematian setiap tahun secara global [World Health Organization, 2021].

Prediksi kualitas udara sangat penting untuk mendukung pengambilan keputusan pemerintah dan masyarakat, seperti penjadwalan aktivitas luar ruangan, kebijakan pembatasan kendaraan, dan peringatan dini terhadap kelompok rentan. Oleh karena itu, diperlukan solusi berbasis machine learning yang mampu memprediksi nilai polutan udara berdasarkan data historis sensor lingkungan.

## Business Understanding

### Problem Statements

- Bagaimana memprediksi konsentrasi polutan udara di masa depan secara akurat berdasarkan data sensor historis?
- Algoritma apa yang paling efektif untuk melakukan prediksi multivariat kualitas udara pada data time series?

### Goals

- Membangun model machine learning untuk memprediksi nilai polutan udara (seperti PM2.5, PM10) di masa depan.
- Menentukan metrik evaluasi yang sesuai dan mengukur performa model terhadap data uji.

### Solution statements

- **Solusi 1:** Membangun model LSTM karena cocok untuk data time series dan dapat menangani dependensi jangka panjang antar variabel.
- **Solusi 2:** Melakukan tuning hyperparameter pada model LSTM (jumlah neuron, epochs, batch size) untuk meningkatkan akurasi model.
- **Solusi 3:** Mengevaluasi model dengan metrik RMSE, MAE, MAPE, dan SMAPE untuk mendapatkan gambaran menyeluruh tentang error prediksi.

## Data Understanding

Dataset diambil dari sumber kualitas udara lokal dengan total data sebanyak sekitar 4000+ entri (jumlah disesuaikan dengan data asli pengguna) yang mencakup pengukuran harian selama lebih dari satu tahun.

Tautan sumber dataset: [Kaggle](https://www.kaggle.com/datasets/arashnic/book-recommendation-dataset)


Jumlah Baris : 9471
Jumlah Kolom : 17

### Fitur-fitur pada dataset:

- Date : Menunjukkan tanggal pengambilan data
- Time : Menunjukkan waktu pengambilan data
- CO(GT) : Menunjukkan besaran CO
- PT08.S1(CO) : Menunjukkan besaran PT08.S1
- NMHC(GT) : Menunjukkan besaran NMHC
- C6H6(GT) : Menunjukkan besaran C6H6
- PT08.S2(NMHC) : Menunjukkan besaran PT08.S2
- NOx(GT) : Menunjukkan besaran NOx
- PT08.S3(NOx) : Menunjukkan besaran PT08.S3
- NO2(GT) : Menunjukkan besaran NO2
- PT08.S4(NO2) : Menunjukkan besaran PT08.S4
- PT08.S5(O3) : Menunjukkan besaran PT08.S5
- T : Menunjukkan temperatur suhu
- RH : Menunjukkan besaran Relative Humidity
- AH : Menunjukkan besaran Absolute Humidity
- Unnamed: 15 : Data yang tidak berguna
- Unnamed: 16 : Data yang tidak berguna   

Dari semua fitur diatas kondisi data dapat dijelaskan:
- Missing data

| Kolom             | Jumlah Data Kosong |
|-------------------|--------------------|
| Date              | 114                |
| Time              | 114                |
| CO(GT)            | 114                |
| PT08.S1(CO)       | 114                |
| NMHC(GT)          | 114                |
| C6H6(GT)          | 114                |
| PT08.S2(NMHC)     | 114                |
| NOx(GT)           | 114                |
| PT08.S3(NOx)      | 114                |
| NO2(GT)           | 114                |
| PT08.S4(NO2)      | 114                |
| PT08.S5(O3)       | 114                |
| T                 | 114                |
| RH                | 114                |
| AH                | 114                |
| Unnamed: 15       | 9471               |
| Unnamed: 16       | 9471               |
- Outlier

| Kolom              | Jumlah Data Outlier |
|--------------------|---------------------|
| CO(GT)             | 1898                |
| PT08.S1(CO)        | 484                 |
| NMHC(GT)           | 914                 |
| C6H6(GT)           | 606                 |
| PT08.S2(NMHC)      | 426                 |
| NOx(GT)            | 509                 |
| PT08.S3(NOx)       | 602                 |
| NO2(GT)            | 1696                |
| PT08.S4(NO2)       | 450                 |
| PT08.S5(O3)        | 458                 |
| T                  | 368                 |
| RH                 | 366                 |
| AH                 | 367                 |
| Unnamed: 15        | 0                   |
| Unnamed: 16        | 0                   |
### Exploratory Data Analysis (EDA)

- Visualisasi distribusi polutan menunjukkan adanya outlier .

## Data Preparation

Langkah-langkah data preparation yang dilakukan:

1. **Missing value handling:** Menghapus atau menginterpolasi data hilang.
2. **Handling Outlier:** Menggantikan data diluar outlier dengan batas atas/bawah.
3. **Feature engineering:** Mengubah timestamp menjadi fitur waktu (misal: jam, hari, bulan).
4. **Scaling:** Menggunakan MinMaxScaler untuk menormalkan nilai antar fitur.
4. **Sequence generation:** Mengubah data menjadi format time series multivariat (X timesteps, Y prediksi).
5. **Train-test split:** 80% data untuk pelatihan, 20% untuk pengujian.

Alasan dilakukan normalisasi adalah karena LSTM sensitif terhadap skala data.

## Modeling

Model utama yang digunakan adalah LSTM dan GRU sebagai berikut:
## **1. Long Short-Term Memory (LSTM)**

### 🔧 Cara Kerja:
Long Short-Term Memory (LSTM) merupakan jenis dari Recurrent Neural Network (RNN) yang dirancang untuk menangani data berurutan dan mengatasi permasalahan *vanishing gradient*. LSTM bekerja dengan memanfaatkan mekanisme memori jangka panjang melalui tiga gerbang utama:

- **Forget Gate**: Memutuskan informasi mana yang harus dilupakan dari sel memori sebelumnya.
- **Input Gate**: Memilih informasi baru yang akan disimpan di memori.
- **Output Gate**: Menentukan informasi dari sel memori yang akan dikeluarkan sebagai output.

LSTM efektif digunakan dalam permasalahan time series karena mampu mengingat pola data historis secara berurutan.

### ⚙️ Parameter yang Digunakan:
```python
LSTM(64, input_shape=(timesteps, X.shape[2]))
```

- `64`: Jumlah unit LSTM yang digunakan. Ini menentukan kapasitas jaringan dalam mengingat informasi.
- `input_shape=(timesteps, X.shape[2])`: Dimensi input dari data sekuensial multivariat (jumlah langkah waktu dan fitur).

Layer output:
```python
Dense(y.shape[2])
```

- Layer `Dense` digunakan untuk menghasilkan output akhir sebesar dimensi target.

**Parameter default lainnya:**
- `activation='tanh'`
- `recurrent_activation='sigmoid'`
- `return_sequences=False` (default)

---

## **2. Gated Recurrent Unit (GRU)**

### 🔧 Cara Kerja:
Gated Recurrent Unit (GRU) adalah varian dari LSTM yang menyederhanakan struktur internal namun tetap mempertahankan kemampuan untuk belajar dari data sekuensial. GRU memiliki dua gerbang utama:

- **Update Gate**: Mengontrol seberapa banyak informasi masa lalu yang dibawa ke waktu sekarang.
- **Reset Gate**: Mengatur seberapa banyak informasi lama yang akan dilupakan.

GRU tidak memiliki sel memori terpisah seperti LSTM, sehingga lebih efisien secara komputasi dan lebih cepat dilatih.

### ⚙️ Parameter yang Digunakan:
```python
GRU(64, input_shape=(timesteps, X.shape[2]))
```

- `64`: Jumlah unit GRU.
- `input_shape`: Sama seperti pada LSTM, menyesuaikan dimensi data input.

Layer output:
```python
Dense(y.shape[2])
```

- Digunakan untuk menghasilkan output prediksi.


## Evaluation
**Problem 1: Bagaimana memprediksi konsentrasi polutan udara di masa depan secara akurat berdasarkan data sensor historis?**  
Ya. Model LSTM dan GRU memanfaatkan data sensor historis yang berbentuk time series untuk mempelajari pola perubahan polusi udara. Hasil evaluasi menunjukkan bahwa kedua model mampu melakukan prediksi dengan cukup baik.

**Problem 2: Algoritma apa yang paling efektif untuk melakukan prediksi multivariat kualitas udara pada data time series?**  
Ya. Dengan membandingkan performa model LSTM dan GRU melalui metrik evaluasi seperti RMSE, MAE, MAPE, dan SMAPE, dapat disimpulkan bahwa GRU adalah model yang lebih efektif untuk kasus ini karena memberikan nilai error yang lebih rendah pada semua metrik.

### ✅ Apakah model berhasil mencapai setiap goals?

1. **Membangun model machine learning untuk memprediksi nilai polutan udara di masa depan**  
   ✔️ Sudah tercapai. Dua model deep learning, yaitu LSTM dan GRU, telah berhasil dibangun dan dilatih menggunakan data sensor kualitas udara multivariat.

2. **Menentukan metrik evaluasi yang sesuai dan mengukur performa model terhadap data uji**  
   ✔️ Sudah tercapai. Model dievaluasi menggunakan empat metrik yang relevan untuk prediksi time series: RMSE, MAE, MAPE, dan SMAPE.

### ✅ Apakah setiap solusi statement berdampak?

- **Solusi 1: Membangun model LSTM dan GRU**  
  ✔️ Berdampak langsung. Kedua model mampu menangani karakteristik data time series dengan banyak variabel, dan memberikan hasil prediksi yang dapat diandalkan.

- **Solusi 2: Melakukan tuning hyperparameter pada model**  
  ✔️ Berdampak signifikan. Proses tuning meningkatkan akurasi model dan mempercepat konvergensi selama pelatihan.

- **Solusi 3: Mengevaluasi model dengan berbagai metrik**  
  ✔️ Berdampak besar. Evaluasi dengan metrik yang berbeda memberikan pemahaman menyeluruh tentang kualitas prediksi model dan membantu dalam pemilihan model terbaik.

---

## Kesimpulan

Model GRU memberikan hasil evaluasi yang lebih baik dibandingkan LSTM:
- **GRU RMSE:** 96.55 (lebih rendah dari LSTM: 107.55)
- **GRU MAE:** 51.58 (lebih rendah dari LSTM: 57.22)
- **GRU MAPE:** 17.58% (lebih baik dari LSTM: 19.53%)
- **GRU SMAPE:** 15.28% (lebih baik dari LSTM: 17.07%)

Dengan demikian, GRU direkomendasikan sebagai model yang lebih efektif untuk menjawab kebutuhan bisnis dalam memprediksi kualitas udara secara akurat dan efisien.

### Formula Evaluasi:
- **RMSE**  
  $$\mathrm{RMSE} = \sqrt{\frac{1}{n} \sum_{i=1}^n (y_i - \hat{y}_i)^2}$$

- **MAE**  
  $$\mathrm{MAE} = \frac{1}{n} \sum_{i=1}^n |y_i - \hat{y}_i|$$

- **MAPE**  
  $$\mathrm{MAPE} = \frac{100\%}{n} \sum_{i=1}^n \left| \frac{y_i - \hat{y}_i}{y_i} \right|$$

- **SMAPE**  
  $$\mathrm{SMAPE} = \frac{100\%}{n} \sum_{i=1}^n \frac{|y_i - \hat{y}_i|}{\frac{|y_i| + |\hat{y}_i|}{2}}$$

### Hasil:
##### LSTM
| Metrik | Nilai |
|--------|-------|
| RMSE   | 107.5519427988917 |
| MAE    | 57.22130302720162 |
| MAPE   | 0.19532512224129206 |
| SMAPE  | 17.06941696515311 |
##### GRU
| Metrik | Nilai |
|--------|-------|
| RMSE   | 96.5510672970197|
| MAE    | 51.58401944322152 |
| MAPE   | 0.17580061082491324 |
| SMAPE  | 15.279475230172848 |
**Interpretasi:**
- Nilai RMSE dan MAE menunjukkan adanya deviasi prediksi yang masih bisa diperbaiki.
- MAPE tinggi disebabkan oleh nilai aktual yang mendekati nol (mempengaruhi denominator).
- SMAPE menunjukkan hasil yang lebih stabil dan layak digunakan untuk evaluasi akhir.

---

_Peningkatan lebih lanjut dapat dilakukan dengan menambahkan fitur eksternal (cuaca, waktu), serta mempertimbangkan model hybrid seperti LSTM-GRU atau ensembel._

