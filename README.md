# Penilaian Proyek
Proyek ini berhasil mendapatkan bintang 5/5 pada submission dicoding course Machine Learning Operations (MLOps).

<img src="https://raw.githubusercontent.com/AbiyaMakruf/Dicoding-PengembanganDanPengoperasianSistemMachineLearning/main/images/nilai.png" width="500">

Kriteria tambahan yang saya kerjakan sehingga mendapat nilai terbaik:
1. Memanfaatkan komponen Tuner untuk menjalankan proses hyperparameter tuning secara otomatis.
2. Menerapkan prinsip clean code dalam membuat machine learning pipeline.  
3. Menambahkan sebuah berkas notebook untuk menguji dan melakukan prediction request ke sistem machine learning yang telah dijalankan di cloud.
4. Menyinkronkan Prometheus dengan Grafana untuk membuat dashboard monitoring yang lebih menarik.

# Laporan Proyek
Nama: Muhammad Abiya Makruf

Username dicoding: abiyamf

| | Deskripsi |
| ----------- | ----------- |
| Dataset | Dataset yang digunakan adalah [Predicting Hiring Decisions in Recruitment Data](https://www.kaggle.com/datasets/rabieelkharoua/predicting-hiring-decisions-in-recruitment-data) data ini memberikan wawasan tentang faktor-faktor yang memengaruhi keputusan perekrutan.|
| Masalah | Masalah yang diangkat dalam proyek ini adalah prediksi keputusan perekrutan berdasarkan data kandidat yang meliputi usia, jenis kelamin, tingkat pendidikan, tahun pengalaman, jumlah perusahaan sebelumnya, dll. Mengingat proses perekrutan sangat penting untuk menentukan kandidat yang tepat bagi sebuah perusahaan, memiliki model yang dapat memprediksi keputusan perekrutan berdasarkan data kandidat historis sangatlah bermanfaat. Model ini dapat membantu perusahaan dalam mengotomatisasi dan mempercepat proses seleksi, memastikan objektivitas, dan meningkatkan efisiensi dalam pengambilan keputusan perekrutan. |
| Solusi machine learning | Solusi machine learning yang diusulkan adalah membangun pipeline machine learning yang mencakup preprocessing data, pelatihan model, evaluasi model, dan deployment model menggunakan TensorFlow Serving. Pipeline ini akan menggunakan TFX (TensorFlow Extended) untuk mengelola alur kerja. |
| Metode pengolahan | Preprocessing Data <br> Metode pengolahan data yang digunakan meliput: <br> 1. Normalisasi: Fitur numerik dinormalisasi menggunakan `tft.scale_to_0_1`|
| Arsitektur model | Arsitektur model yang digunakan adalah model neural network dengan beberapa lapisan fully connected. Berikut adalah arsitektur model yang digunakan: <br><br> 1. Input layer: Mengambil fitur-fitur yang telah ditransformasikan. <br> 2. Hidden Layer: tiga lapisan dense menggunakan aktivasi ReLU. <br> 3. Output Layer: Lapisan dense dengan 1 unit dan fungsi aktivasi sigmoid. <br> 4. Optimizer: Adam. <br> 5. Loss: binary_crossentropy. <br><br> Terdapat juga [tuner](https://github.com/AbiyaMakruf/Dicoding-PengembanganDanPengoperasianSistemMachineLearning/blob/main/modules/tuner.py)  yang memiliki urutan arsitektur yang serupa dengan tambahan layer dropout.  |
| Metrik evaluasi | Metrik yang digunakan untuk mengevaluasi performa model adalah: <br><br> 1. Precision: Metrik yang digunakan untuk mengukur akurasi dari prediksi positif model.<br> 2. Recall: Metrik yang digunakan untuk mengukur kemampuan model dalam menemukan semua instance positif yang sebenarnya. <br> 3. BinaryAccuracy: Akurasi khusus untuk masalah klasifikasi dengan jumlah kelas adalah dua. <br><br> Model yang dibuat dinyatakan 'blessed' jika BinaryAccuracy > 0.8.|
| Performa model | Performa model dievaluasi menggunakan dataset evaluasi. Hasil evaluasi menunjukkan bahwa model memiliki akurasi yang tinggi dalam mengklasifikasikan recruitment data dengan menggunakan fitur-fitur yang tersedia. Hasil hyperparameter tuning menunjukkan bahwa model dengan arsitektur tertentu memberikan performa terbaik dengan akurasi mencapai lebih dari 87%. |
| Opsi deployment | Model yang telah dibuat dapat dideploy menggunakan beberapa opsi seperti [heroku](https://www.heroku.com/) dan [railway](https://railway.app/). Proyek ini melakukan deploy model ke platform railway.|
| Web app | Tautan web app yang digunakan untuk mengakses model serving. [cc-model](https://proyek-akhir-mlops-production.up.railway.app/v1/models/cc-model/metadata)|
| Monitoring | Memonitor model yang sudah dideploy menggunakan Prometheus dan gafana.  Hasil monitoring menggunakan `request_count` ataupun `session_created` memberikan hasil yang bagus, setiap request prediksi yang masuk ke model tercatat dengan baik dan semenjak sessi dibentuk model tidak pernah down.|

# How to reproduce

## Membuat virtual environtments
- Buat virtual environtment dengan menjalankan perintah berikut
    ```
    conda create --name proyek-akhir-mlops python==3.9.15
    ```

- Aktifkan environtment 
    ```
    conda activate proyek-akhir-mlops
    ```


## Menginstall requirements
- Pastikan sudah pada environtment proyek-akhir-mlops kemudian jalankan
    ```
    pip install -r requirements.txt
    ```

- Jika ingin menerapkan python clean code maka jalankan
    ```
    pip install autopep8 pylint
    ```

## Menjalankan pipeline
- Clone repository dengan menjalankan
    ```
    git clone https://github.com/AbiyaMakruf/Dicoding-PengembanganDanPengoperasianSistemMachineLearning
    ```

- Buka berkas notebook.ipynb

- Jika proses berhasil maka akan muncul folder output/serving model

## Menjalankan model machine learning di railway

- Lakukan pembuatan akun railway, install railway cli, login ke akun railway, membuat project baru di railway, dan menghubungkan project ke cli menggunakan cara [berikut](https://www.dicoding.com/academies/443/discussions/225535). 

- Setelah itu lakukan perintah `railway up` untuk melakukan push dan build docker ke railway.

- Jika deploy berhasil maka dilanjutkan dengan membuat domain seperti pada diskusi.

- Akses domain dengan menambahkan `/v1/models/cc-model/metadata`

## Monitoring model
### Menggunakan Prometheus
- Jalankan kedua perintah berikut

    ```
    docker build -t cc-monitoring .\monitoring\
    docker run -p 9090:9090 cc-monitoring 
    ```

- Masuk ke dalam dashboard yang telah disediakan oleh Prometheus melalui tautan 
    ```
    http://localhost:9090/
    ```

- Gunakan salah satu query yaitu 
    ```
    :tensorflow:serving:request_count
    ```

### Menggunakan Grafana (windows tutorial)
- Lakukan instalasi [grafana](https://grafana.com/grafana/download?platform=windows).

- Start Grafana by executing grafana-server.exe, located in the bin directory, preferably from the command line.

- To run Grafana, open your browser and go to the Grafana port http://localhost:3000/

- Masukkan admin dan admin

- Tambahkan data source baru dengan memilih pilihan prometheus

- Pada bagian `connection` masukkan http://localhost:9090

## Melakukan predict
- Buka file test.ipynb
- Jalankan file paling terakhir
