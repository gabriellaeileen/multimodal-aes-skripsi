import csv
import requests
import os

# Membuat folder images jika belum ada
os.makedirs('../images', exist_ok=True)

# Membaca file data.csv
with open('data.csv', 'r', encoding='latin1') as file:
    reader = csv.reader(file)
    next(reader)  # Lewati header
    for i, row in enumerate(reader):
        url = row[0]  # Kolom graph adalah kolom pertama
        try:
            response = requests.get(url)
            response.raise_for_status()  # Cek jika request berhasil
            # Simpan sebagai JPG meski asli mungkin PNG
            with open(f'images/{i}.jpg', 'wb') as f:
                f.write(response.content)
            print(f"Downloaded {url} as {i}.jpg")
        except requests.exceptions.RequestException as e:
            print(f"Failed to download {url}: {e}")

print("Download selesai.")