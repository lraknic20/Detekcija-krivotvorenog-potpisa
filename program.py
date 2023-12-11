import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import cv2
import os
import matplotlib.pyplot as plt

def load_data(data_path):
    data = []
    labels = []

    for root, dirs, files in os.walk(data_path):
        for filename in files:
            if filename.lower().endswith(".png"):
                img_path = os.path.join(root, filename)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                img = cv2.resize(img, (150, 150))

                # Provjeri je li "_forg" u putanji
                label = 1 if "_forg" in root else 0  # 1 za krivotvorene, 0 za originalne potpise
                data.append(img.flatten())
                labels.append(label)

                """ # Prikazivanje prvih 5 slika
                if len(data) <= 5:
                    plt.figure()
                    plt.imshow(img, cmap='gray')
                    plt.title(f"Label: {label}")
                    plt.show() """

    return np.array(data), np.array(labels)

# Putanja do skupa podataka
data_path = r"C:\Users\leon3\Documents\Faks\Biometrijski sustavi\Projekt\Baze\archive\sign_data\train"

print("Učitavanje podataka...")
X, y = load_data(data_path)

broj_originala = np.sum(y == 0)
broj_krivotvorenih = np.sum(y == 1)

print(f"Broj originalnih potpisa: {broj_originala}")
print(f"Broj krivotvorenih potpisa: {broj_krivotvorenih}")

# Podjela skupa podataka na trening i test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Inicijalizacija modela...")
model = RandomForestClassifier(n_estimators=100)
#model = RandomForestClassifier(n_estimators=100, random_state=42)

print("Treniranje modela...")
model.fit(X_train, y_train)

print("Predviđanje na testnom skupu...")
y_pred = model.predict(X_test)

# Evaluacija modela
accuracy = accuracy_score(y_test, y_pred)
print(f"Točnost modela: {accuracy:.2f}")

print("Izvještaj o klasifikaciji:")
print(classification_report(y_test, y_pred))