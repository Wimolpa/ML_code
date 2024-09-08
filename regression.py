import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import  LogisticRegression
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
import seaborn as sns
from sklearn.metrics import confusion_matrix

df = pd.read_csv('C:/Users/asus/Downloads/weather_classification_data.csv')

# เลือกคอลัมน์ที่เป็น features และ target
x = df.drop('Weather Type', axis=1)
y = df['Weather Type']

# แปลงข้อมูล categorical ให้เป็นตัวเลข
le = LabelEncoder()

# ทำการแปลงคอลัมน์ที่เป็น string ทั้งหมดใน features
for column in x.columns:
    if x[column].dtype == 'object':  # ตรวจสอบว่าเป็น string หรือไม่
        x[column] = le.fit_transform(x[column])

# ทำการแปลง target (ถ้า target เป็น string)
if y.dtype == 'object':
    y = le.fit_transform(y)

# แบ่งข้อมูลเป็น train และ test sets
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.20)

log_reg = LogisticRegression()
log_reg = log_reg.fit(X_train, y_train)

# ทำนายผลจาก Logistic Regression
y_pred_log_reg = log_reg.predict(X_test)

# คำนวณค่า Accuracy ของ Logistic Regression
accuracy_log_reg = accuracy_score(y_test, y_pred_log_reg)

# แสดงผลลัพธ์ของ Logistic Regression
print(f"Accuracy: {accuracy_log_reg:.2f}")

# Logistic Regression
log_reg = LogisticRegression()
log_reg = log_reg.fit(X_train, y_train)

# ทำนายผลจาก Logistic Regression
y_pred_log_reg = log_reg.predict(X_test)

# แปลงค่าผลลัพธ์กลับไปเป็นชื่อเดิม
y_test_labels = le.inverse_transform(y_test)
y_pred_labels = le.inverse_transform(y_pred_log_reg)

# แสดงผล Confusion Matrix
conf_matrix = confusion_matrix(y_test_labels, y_pred_labels)

# แสดงผล classification report
target_names = ['Cloudy', 'Rainy', 'Sunny', 'Snowy']
clr = classification_report(y_test, y_pred_log_reg, target_names=target_names)
print(clr)

plt.figure(figsize=(7, 5))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=target_names, yticklabels=target_names)
plt.title('Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()