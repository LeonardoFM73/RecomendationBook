# Laporan Proyek Machine Learning - Leonardo Fajar Mardika

## Project Overview
Minat baca masyarakat dunia terus meningkat seiring dengan semakin mudahnya akses terhadap buku digital melalui berbagai platform online. Namun, membanjirnya jumlah buku baru yang terbit setiap tahun juga menyebabkan permasalahan baru: pengguna kerap mengalami kesulitan dalam menemukan buku yang sesuai dengan preferensi mereka. Dalam konteks ini, sistem rekomendasi menjadi solusi krusial untuk menyaring dan menyajikan konten yang relevan bagi pengguna secara personalisasi.

Sistem rekomendasi buku adalah teknologi yang mampu menyarankan buku kepada pengguna berdasarkan riwayat interaksi mereka, seperti penilaian, pencarian, atau pembelian sebelumnya. Teknologi ini tidak hanya meningkatkan pengalaman pengguna, tetapi juga dapat mendorong tingkat keterlibatan dan penjualan buku. Dalam platform seperti Amazon dan Goodreads, sistem rekomendasi telah terbukti berkontribusi signifikan terhadap keberhasilan bisnis dengan memberikan konten yang relevan dan meningkatkan kepuasan pengguna [[1]](https://doi.org/10.1016/j.is.2015.08.006 ).

Masalah ini penting untuk diselesaikan karena pengguna seringkali tidak tahu harus mulai dari mana untuk mencari buku yang sesuai dengan minat mereka. Tanpa sistem rekomendasi, pencarian buku menjadi tidak efisien, memakan waktu, dan membuat pengguna kehilangan minat membaca. Oleh karena itu, proyek ini bertujuan untuk membangun sistem rekomendasi buku menggunakan pendekatan collaborative filtering berbasis algoritma k-Nearest Neighbors (k-NN), yang fokus pada kemiripan antar buku dari segi rating pengguna.

Studi sebelumnya menunjukkan bahwa collaborative filtering menjadi metode yang populer dalam sistem rekomendasi karena kemampuannya mempelajari pola perilaku pengguna tanpa membutuhkan informasi eksplisit dari item [[2]](https://doi.org/10.1007/978-1-4899-7637-6 ). Model ini akan membantu pengguna menemukan buku-buku yang kemungkinan besar mereka sukai, berdasarkan kesamaan preferensi pengguna lain atau item serupa yang pernah mereka beri rating tinggi.

Referensi:

[1] Jannach, D., & Adomavicius, G. (2016). Recommendation systems: Challenges, insights and research opportunities. Information Systems, 56, 1–9. https://doi.org/10.1016/j.is.2015.08.006 

[2] Ricci, F., Rokach, L., & Shapira, B. (2015). Recommender Systems Handbook (2nd ed.). Springer. https://doi.org/10.1007/978-1-4899-7637-6
Pada bagian ini, Kamu perlu menuliskan latar belakang yang relevan dengan proyek yang diangkat.

**Rubrik/Kriteria Tambahan (Opsional)**:
- Jelaskan mengapa dan bagaimana masalah tersebut harus diselesaikan
- Menyertakan hasil riset terkait atau referensi. Referensi yang diberikan harus berasal dari sumber yang kredibel dan author yang jelas.
- Format Referensi dapat mengacu pada penulisan sitasi [IEEE](https://journals.ieeeauthorcenter.ieee.org/wp-content/uploads/sites/7/IEEE_Reference_Guide.pdf), [APA](https://www.mendeley.com/guides/apa-citation-guide/) atau secara umum seperti [di sini](https://penerbitdeepublish.com/menulis-buku-membuat-sitasi-dengan-mudah/)
- Sumber yang bisa digunakan [Scholar](https://scholar.google.com/)

## Business Understanding
### Problem Statements

Menjelaskan pernyataan masalah:
- Bagaimana memberikan rekomendasi buku yang relevan kepada pengguna berdasarkan riwayat interaksi mereka?
- Algoritma rekomendasi apa yang efektif dan ringan untuk diimplementasikan?

### Goals

Menjelaskan tujuan proyek yang menjawab pernyataan masalah:
- Membangun sistem rekomendasi buku berbasis content base filtering dan collaborative filtering.
- Menghasilkan top-N rekomendasi buku untuk setiap pengguna berdasarkan histori pengguna.

Semua poin di atas harus diuraikan dengan jelas. Anda bebas menuliskan berapa pernyataan masalah dan juga goals yang diinginkan.

## Data Understanding
Sumber dataset didapat dari sumber [Kaggle](https://www.kaggle.com/datasets/arashnic/book-recommendation-dataset).

Variabel-variabel pada Books.csv dataset adalah sebagai berikut:
- ISBN : merupakan nomor identifikasi unik buku.
- Book-Title : merupakan judul buku.
- Book-Author : merupakan nama penulis buku.
- Year-Of-Publication : merupakan tahun penerbitan.
- Publisher : merupakan organisasi/instansi yang menerbitkan buku tersebut.
- Image-URL-S : merupakan gambar sampul buku berukuran kecil.
- Image-URL-M	 : merupakan gambar sampul buku berukuran sedang.
- Image-URL-L : merupakan gambar sampul buku berukuran besar.
Books.csv memiliki 
Jumlah Baris 271360
Jumlah Kolom 8

Data kosong  Ratings.csv
| Kolom             | Jumlah Data Kosong |
|-------------------|--------------------|
| ISBN           | 0                  |
| Book-Title           | 0                |
| Book-Author               | 2             |
| Year-Of-Publication               | 0             |
| Publisher                             | 2             |
| Image-URL-S              | 0             |
| Image-URL-M              | 0             |
| Image-URL-M              | 3             |

Variabel-variabel pada Ratings.csv dataset adalah sebagai berikut:
- User-ID : merupakan nomor identitas pembaca.
- ISBN : merupakan nomor identifikasi unik buku.
- Book-Rating : merupakan nilai yang diberikan pembaca kepada buku.
Ratings.csv memiliki 
Jumlah Baris 1149780
Jumlah Kolom 3

Data kosong  Ratings.csv
| Kolom             | Jumlah Data Kosong |
|-------------------|--------------------|
| User-ID           | 0                  |
| ISBN          | 0                |
| Book-Rating               | 0             |


Variabel-variabel pada Ratings.csv dataset adalah sebagai berikut:
- User-ID : merupakan nomor identitas pembaca.
- Location : merupakan lokasi pembaca buku.
- Age : merupakan umur pembaca buku.
Users.csv memiliki 
Jumlah Baris 278858
Jumlah Kolom 3

Data kosong  Ratings.csv
| Kolom             | Jumlah Data Kosong |
|-------------------|--------------------|
| User-ID           | 0                  |
| Location          | 0                |
| Age               | 110762             |

**EDA (Exploratory Data Analysis)**:
- Melakukan pengurutan 10 buku rating terbanyak
- Mengetahui distribusi usia pembaca
- Mengetahui publisher dengan buku terbitan terbanyak

Top 10 Buku dengan Rating Terbanyak:
| Book-Title |    Book-Author| Year-Of-Publication|  
|-------------------|--------------------|--------------------|
| Wild Animus|    Rich Shapero|2004
|The Joy Luck Club|         Amy Tan   |1994
|The Secret Life of Bees|   Sue Monk Kidd   |2003
|  The Lovely Bones: A Novel|    Alice Sebold   |2002
 The Red Tent (Bestselling Backlist)|   Anita Diamant   |1998
|Where the Heart Is (Oprah's Book Club (Paperba...|    Billie Letts |1998  
| The Da Vinci Code|       Dan Brown|   2003
|Divine Secrets of the Ya-Ya Sisterhood: A Novel |  Rebecca Wells   |1997
|Snow Falling on Cedars|  David Guterson|  1995 

##### Distribusi usia pembaca
![usia](https://raw.githubusercontent.com/LeonardoFM73/GambarLaskarAI/main/download.png)
##### Distribusi publisher
![usia](https://raw.githubusercontent.com/LeonardoFM73/GambarLaskarAI/refs/heads/main/download%20(2).png)

## Data Preparation
Pada bagian ini Anda menerapkan dan menyebutkan teknik data preparation yang dilakukan. Teknik yang digunakan pada notebook dan laporan harus berurutan.
Langkah-langkah preprocessing yang dilakukan:
1. Mengambil 50000 data ratings_cv karena keterbatasan hardware 
2. Menggabungkan data rating dengan data buku berdasarkan ISBN.
3. Melakukan cleaning dengan drop data "Image-URL-S","Image-URL-M","Image-URL-L" yang berupa link
3. Menghapus kolom 'Age' 
4. Melakukan ekstraksi informasi negara dari kolom 'Location' pada dataset 'users.csv'
5. Melakukan filter data dengan mendapatkan pengguna aktif (pengguna yang telah memberi rating lebih dari satu buku) 
6. Melakukan filter buku-buku populer (buku yang telah diberi rating oleh setidaknya satu dari pengguna aktif tersebut)
6. Memfilter hanya data yang relevan (rating > 0).
7. Membuat kolom 'content' yang menggabungkan kolom fitur 'Book-Title', 'Book-Author', 'Publisher', dan 'Year-Of-Publication'
8. Menggantu nilai NaN dengan data kosong dengan fillna('')
9. Pembuatan df_unique_books yang berisi data-data hanya satu buku yang memiliki ISBN yang unik
10. Melakukan Vektorisasi dengan TF-IDF
11. Menghilangkan baris dengan judul buku yang tidak valid atau duplikat.
Menyamakan data rating dengan data buku yang sudah unik dan bersih di df_unique_books.




## Modeling
##### Metode: Content-Based Filtering dengan TF-IDF + Cosine Similarity

**1. Perhitungan Kemiripan: Cosine Similarity**
Cosine similarity digunakan untuk mengukur tingkat kemiripan antar buku berdasarkan sudut antar vektor dalam ruang fitur. Menggunakan cosine similarity, dihitung seberapa mirip satu buku dengan yang lainnya berdasarkan vektor TF-IDF-nya. Nilai cosine similarity berkisar dari 0 (tidak mirip) hingga 1 (sangat mirip).
```python
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
```
**2. Fungsi Rekomendasi**
Dibuat fungsi recommend_books(title, num_recommendations) yang menerima judul buku dan menghasilkan daftar buku serupa berdasarkan skor similarity tertinggi (selain dirinya sendiri).
```python
def recommend_books(title, num_recommendations=10):
    ...
    return df_unique_books.iloc[book_indices][['Book-Title', 'Book-Author', 'Publisher', 'Year-Of-Publication']]
```
✅ Hasil Top-N Recommendation
Contoh rekomendasi untuk buku "All Smiles":
```python
recommend_books("All Smiles")
```
| Judul Rekomendasi                                    | Penulis                   | Penerbit                      | Tahun |
| ---------------------------------------------------- | ------------------------- | ----------------------------- | ----- |
| Be Mine                                              | Cait Logan                | Dell                          | 1997  |
| Gilgamesh: A Novel                                   | Joan London               | Grove Press                   | 2003  |
| Dracula in London                                    | P. N. Elrod               | Ace Books                     | 2001  |
| The Call of the Wild (Apple Classics)                | Jack London               | Scholastic                    | 1993  |
| The Pursuit (Avon Historical Romance)                | Johanna Lindsey           | Avon                          | 2003  |
| London Holiday                                       | Richard Peck              | Penguin USA                   | 1998  |
| Call of the Wild and Selected Stories                | Jack London               | Signet Classics               | 1993  |
| Mouthful Of Breath Mints and No One to Kiss          | Cathy Guisewite           | Andrews McMeel Publishing     | 1983  |
| Breathing Room (Avon Romance)                        | Susan Elizabeth Phillips  | Avon Books                    | 2003  |
| Down and Out in Paris and London                     | George Orwell             | Harvest Books                 | 1972  |


✅ Kelebihan Model Content-Based Filtering
- Personalized tanpa data pengguna lain (no cold-start for items)
- Rekomendasi didasarkan pada kesamaan konten item, sehingga bisa memberikan saran meskipun pengguna baru belum memberi rating.
- Tidak tergantung pada interaksi pengguna lain
- Tidak perlu data rating dari banyak pengguna. Cukup satu pengguna dan informasi konten buku.
- Buku baru dapat langsung dimasukkan ke sistem selama memiliki metadata (judul, penulis, dsb.).

❌ Kekurangan Model Content-Based Filtering
- Over-specialization (rekomendasi sempit)
- Hanya merekomendasikan item yang sangat mirip. Tidak ada eksplorasi item baru di luar preferensi pengguna.
- Bergantung pada kualitas metadata


### Metode Collaborative Filtering
Pada tahap ini, digunakan pendekatan Collaborative Filtering untuk membangun sistem rekomendasi berdasarkan pola interaksi pengguna terhadap item. Model yang digunakan adalah SVD (Singular Value Decomposition) dari pustaka Surprise.

🔧 Metode: Collaborative Filtering dengan SVD
###### 1. Persiapan Data untuk Surprise
Data diubah ke format yang kompatibel dengan library Surprise, yaitu terdiri dari:
```python
reader = Reader(rating_scale=(0, 10))
data = Dataset.load_from_df(final_ratings[['User-ID', 'ISBN', 'Book-Rating']], reader)
```
###### 2. Pembagian Data Training dan Testing
Data dibagi menjadi:
80% untuk pelatihan
20% untuk pengujian
```python
trainset, testset = train_test_split(data, test_size=0.2)
```
###### 3. Model SVD
Model Singular Value Decomposition (SVD) digunakan karena kemampuannya dalam menangkap dimensi laten dari pengguna dan item berdasarkan pola rating.
```python
model = SVD()
model.fit(trainset)
```
###### 4. Evaluasi Model
Evaluasi dilakukan pada data uji dengan menghitung RMSE (Root Mean Square Error) terhadap prediksi model.
```python
predictions = model.test(testset)
rmse = accuracy.rmse(predictions)
```
###### ✅ Hasil Top-N Recommendation
Rekomendasi buku untuk pengguna tertentu, misalnya User-ID: 276747, dilakukan dengan:
- Mengidentifikasi buku yang belum pernah dirating oleh pengguna.
- Melakukan prediksi rating terhadap buku-buku tersebut.
- Mengurutkan prediksi dari yang tertinggi.

| Judul Buku                                                                 | ISBN         | Estimasi Rating |
|---------------------------------------------------------------------------|--------------|-----------------|
| Harry Potter and the Sorcerer's Stone (Book 1)                            | 0590353403   | 8.53            |
| The Secret Life of Bees                                                   | 0142001740   | 8.32            |
| Harry Potter and the Goblet of Fire (Book 4)                              | 0439139597   | 8.32            |
| Harry Potter and the Chamber of Secrets (Book 2)                          | 0439064872   | 8.26            |
| The Brethren                                                              | 0440236673   | 8.26            |
| Harry Potter and the Order of the Phoenix (Book 5)                        | 043935806X   | 8.24            |
| Harry Potter and the Goblet of Fire (Book 4)                              | 0439139600   | 8.24            |
| The Handmaid's Tale                                                       | 0449212602   | 8.23            |
| Their Eyes Were Watching God: A Novel                                     | 0060916508   | 8.23            |
| The Fellowship of the Ring (The Lord of the Rings, Part 1)                | 0345339703   | 8.21            |




## Evaluation
###### Problem 1: Apakah sistem dapat memberikan rekomendasi berdasarkan buku yang pernah dibaca pengguna?
✔️ Ya, dengan pendekatan metode content-based filtering sistem dan collaborative merekomendasikan buku-buku yang mirip dengan buku yang pernah diberi rating tinggi oleh pengguna.

###### Problem 2: Algoritma rekomendasi apa yang efektif dan ringan untuk diimplementasikan?
✔️ Ya, model TF-IDF + Cosine Similarity dan SVD menghasilkan rekomendasi yang intuitif dan dapat dijelaskan berdasarkan kemiripan item. Model SVD (Collaborative Filtering) menghasilkan nilai matriks evalusi yang bagus sedangkan yang content based filtering kurang baik. Hal ini disebabkan ileh fitur parameter yang dipakai dalam pelatihan

✅ Apakah model berhasil mencapai setiap goals?
###### Membangun sistem rekomendasi buku content base filtering dan collaborative filtering.
✔️ Tercapai melalui pendekatan item-item similarity.
###### Menghasilkan rekomendasi buku personalisasi
✔️ Tercapai melalui input judul buku dan menghasilkan  minimal 5 rekomendasi terdekat.
##### Untuk content based filtering didapat besar matriks evaluasi 
Precision@10: 0.034990791896869246
Recall@10: 0.07983193277310924
Coverage: 0.026003452740612863
Diversity: 1.0

##### Untuk collaborative filtering didapat besar matriks evaluasi 
**Precision@10**: 0.9481
**Recall@10**: 0.9982
**F1-score@10**: 0.9725
**RMSE**  = 1.7629414134994954

**Formula Precision@K**  
$$\mathrm{Precision@K} = \frac{Jumlah\ Rekomendasi\ yang\ Relevan\ di\ Top-K}{Jumlah\ Total\ Rekomendasi\ di\ Top-K}$$
**Formula Recall@K**  
$$\mathrm{Recall@K} = \frac{Jumlah\ Rekomendasi\ yang\ Relevan\ di\ Top-K}{Jumlah\ Item\ Relevan}$$
**Formula F1@K**  
$$\mathrm{F1@K} = \frac{Precision@K * Recall@K }{Precision@K + Recall@K }$$
**F1@K** = 2 * (Precision@K * Recall@K) / (Precision@K + Recall@K)
**Formula RMSE**  
  $$\mathrm{RMSE} = \sqrt{\frac{1}{n} \sum_{i=1}^n (y_i - \hat{y}_i)^2}$$

