from flask import Flask, render_template, request #untuk membuat aplikasi web
import cv2 #menganalisis gambar dan mendeteksi wajah
from keras.models import load_model #mengimpor fungsi load_model
import numpy as np #mengakses fungsi NumPy 
from PIL import Image #mengonversi gambar

app = Flask(__name__) #membuat aplikasi flask

app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 1 #mengatur waktu pada file yg dikirim ke browser


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/after', methods=['GET', 'POST']) #mengambil data dari server dan mengirim data ke server
def after():
    img = request.files['file1'] #menyimpan request file
    img.save('static/file.jpg')
    # Buka gambar
    img = Image.open('static/file.jpg')

    # Kompresi gambar dengan kualitas 50%
    img.save('static/compressed.jpg', optimize=True, quality=50)

    # Image feature
    img1 = cv2.imread('static/compressed.jpg')

    gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY) #mengubah warnah menjadi gray
    cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml') #memuat classifier untuk deteksi wajah pada gambar
    faces = cascade.detectMultiScale(gray, 1.1, 3) #mendeteksi wajah

    #mendeteksi wajah dengan algoitma diatas, menandai wajah terdeteksi dengan kotak hijau
    for x,y,w,h in faces:
        cv2.rectangle(img1, (x,y), (x+w, y+h), (0,255,0), 2) #gambar, koordinat titik awal kotak, koordinat titik akhir kotak, warna

        cropped = img1[y:y+h, x:x+w]

    cv2.imwrite('static/after.jpg', img1) #wajah yang terdeteksi dipotong

    try:
        cv2.imwrite('static/cropped.jpg', cropped) #menyimpan gambar cropped

    except:
        pass


    try:
        image = cv2.imread('static/cropped.jpg', 0) #membaca gambar dengan mode grayscale
    except:
        image = cv2.imread('static/compressed.jpg', 0)

    
    image = cv2.resize(image, (48, 48)) #untuk mengonversi gambar 
    image = image/255.0 #memudahkan proses ke model

    image = np.reshape(image, (1, 48, 48, 1)) #mengubah dimensi gambar yang dapat diterima model, jumlah gambar, ukuran gambar, warna

    model = load_model('model.h5') #memanggil fungsi 'load_model'
    prediction = model.predict(image) #variabel prediction = gambar yang telah di reshape diinput dan memanggil fungsi predict

    label_map = ['anger', 'neutral', 'fear', 'happy', 'sad', 'surprise'] #label untuk masing2 kelas emosi
    prediction = np.argmax(prediction) #mengambil nilai maksimum untuk prediksi dengan model yang telah di load
    final_prediction = label_map[prediction] #mengambil label sesuai label disave 

    return render_template('after.html', data=final_prediction) #data=parameter nilai final_prediction


if __name__ == "__main__":
    app.run(debug=True)
