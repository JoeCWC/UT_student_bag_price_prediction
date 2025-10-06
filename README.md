# UT_student_bag_price_prediction

## ⚙️ 安裝環境
```bash
# 建立 conda 環境
conda create -n ai_cup_env python=3.13.5 -y
conda activate ai_cup_env

# 安裝必要套件
pip install -r requirements.txt
```
---
## 🚀 操作步驟
1. 準備資料
於官網下載資料  
train.csv [下載連結](https://www.kaggle.com/code/himanshukumar7079/backpack-pred/input?select=train.csv)  
test.csv [下載連結](https://www.kaggle.com/code/himanshukumar7079/backpack-pred/input?select=test.csv)  
Noisy_Student_Bag_Price_Prediction_Dataset.csv [下載連結](https://www.kaggle.com/code/himanshukumar7079/backpack-pred/input?select=Noisy_Student_Bag_Price_Prediction_Dataset.csv)

2. 執行訓練
```bash
python baseline.py
```
---