import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import re
import matplotlib.pyplot as plt

# 1. Verileri Yükleme
files = [
    r'C:\Users\yagiz\OneDrive\Masaüstü\Uygulamalar\kodlar\ist_rent\dataset\5_9_2022_sahibinden_ev.csv',
    r'C:\Users\yagiz\OneDrive\Masaüstü\Uygulamalar\kodlar\ist_rent\dataset\22_5_2022_sahibinden_ev.csv',
    r'C:\Users\yagiz\OneDrive\Masaüstü\Uygulamalar\kodlar\ist_rent\dataset\26_5_2022_sahibinden_ev.csv'
]

dataframes = []
for file in files:
    df = pd.read_csv(file)
    # Dosya adından tarihi çıkar
    date_match = re.search(r'(\d{1,2})_(\d{1,2})_(\d{4})', file)
    if date_match:
        day, month, year = map(int, date_match.groups())
        df['date'] = pd.Timestamp(year=year, month=month, day=day)
    
    dataframes.append(df)

data = pd.concat(dataframes, ignore_index=True)
print("Veri yüklendi ve birleştirildi.")

# 2. Veri Temizleme
data = data.dropna()  # Eksik verileri temizle
print("Eksik veriler temizlendi.")

# Sayısal verilere dönüştürme (örneğin fiyatları veya metrekareleri düzenleme)
data['price'] = pd.to_numeric(data['price'].replace('[^0-9]', '', regex=True), errors='coerce')
data['area'] = pd.to_numeric(data['area'].replace('[^0-9]', '', regex=True), errors='coerce')
data['rooms'] = pd.to_numeric(data['numberOfRooms'].replace('[^0-9]', '', regex=True), errors='coerce')

# Geçersiz verileri temizleme
data = data.dropna()
print("Geçersiz veriler temizlendi.")

# Tarih bileşenlerini ayırma
data['year'] = data['date'].dt.year

# 3. Özellik ve Hedef Seçimi
X = data[['area', 'rooms', 'town', 'year']]  # Özellikler
y = data['price']  # Hedef değişken

# Konum ve tarih gibi kategorik değişkenleri kodlama
X = pd.get_dummies(X, columns=['town'], drop_first=True)
print("Kategorik değişkenler kodlandı.")

# 4. Veriyi Eğitim ve Test Setine Bölme
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Veri eğitim ve test setlerine bölündü.")

# 5. Model Eğitimi
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
print("Model eğitildi.")

# 6. Modeli Değerlendirme
predictions = model.predict(X_test)
mse = mean_squared_error(y_test, predictions)
print(f'Model Mean Squared Error: {mse}')

# 7. 2022-2025 Arası Tahminler İçin Veri Hazırlama
future_years = pd.date_range(start='2022', end='2025', freq='Y').year
future_data = pd.DataFrame({
    'area': [140] * len(future_years),
    'rooms': [3] * len(future_years),
    'year': future_years,
    'town_Silivri Alibey Mah': [1] * len(future_years)  # Örneğin Silivri Alibey Mah için dummy değer
})

# Eksik sütunları doldurma
for col in X.columns:
    if col not in future_data.columns:
        future_data[col] = 0

# Sütunları sıralama
future_data = future_data[X.columns]
print("Gelecek yıllar için veri hazırlandı.")

# Tahmin
future_predictions = model.predict(future_data)
print("Gelecek yıllar için tahminler yapıldı.")

# 8. Tahminleri Grafiğe Dökmek
plt.figure(figsize=(10, 6))
plt.plot(future_years, future_predictions, label='Predicted Prices')
plt.xlabel('Year')
plt.ylabel('Price (Million TL)')
plt.title('Predicted Rent Prices from 2022 to 2025')
plt.legend()
plt.grid(True)
plt.show()
print("Tahminler grafiğe döküldü.")