# 🌱 AyurDrishti – AI-Powered Medicinal Plant Identifier + Ayurvedic Chatbot

AyurDrishti is an AI-based medicinal plant recognition system that identifies **80 medicinal plants from leaf images**, fetches Ayurvedic medicinal uses, stores them in MongoDB, and features an **Ayurveda expert chatbot** powered by Gemini AI.

---

## 🚀 Features

### 🔍 1. Plant Image Identification (AI Model)
- Custom-trained **MobileNetV2 model**
- Identifies **80+ medicinal plants**
- Returns:
  - Scientific name  
  - Common name  
  - Medicinal uses  

---

### 📚 2. Medicinal Uses Fetching (DB + Gemini)
- Checks uses in **MongoDB**
- If not found → fetches from **Gemini 2.5 Flash**
- Stores results for future use

---

### 🤖 3. Ayurveda Chatbot
- Replies in **English / Hindi / Hinglish**
- Max **60 words**
- Uses Ayurvedic principles
- Suggests:
  - One herb  
  - Its use  
  - Dosage  
- Rejects non-health queries

---

### 🎨 4. Modern UI (React + Vite + Tailwind)
- Clean, responsive design
- Smooth Plant Identification UI
- Dedicated Chatbot UI

---

### 🗄️ 5. MongoDB Atlas Integration
Stores:
- Plant uses  

---

## 🛠️ Tech Stack

| Component | Technology |
|----------|------------|
| AI Model | TensorFlow + MobileNetV2 |
| Backend | Flask |
| Frontend | React (Vite) |
| Database | MongoDB Atlas |
| AI API | Gemini 2.5 Flash |
| Storage | labels.txt + leaf_model.h5 |

---

## 📂 Project Structure

```
AyurDrishti/
│
├── backend/
│   ├── model/
│   │   ├── leaf_model.h5
│   │   └── labels.txt
│   ├── routes/
│   │   ├── predict.py
│   │   └── chatbot.py
│   ├── db.py
│   ├── gemini_fetch.py
│   ├── app.py
│   ├── requirements.txt
│   └── .env
│
├── frontend/
│   ├── src/
│   │   ├── api/
│   │   ├── components/
│   │   ├── pages/
│   │   └── App.jsx
│   └── package.json
│
└── README.md
```

---

## ⚙️ Backend Setup

### 1️⃣ Create virtual environment    
```bash
cd backend
python -m venv venv
venv\Scripts\activate
```

### 2️⃣ Install dependencies
```bash
pip install -r requirements.txt
```

### 3️⃣ Add `.env`
```env
MONGO_URI=your_mongodb_uri
GEMINI_API_KEY=your_gemini_api_key
```

### 4️⃣ Run backend
```bash
python app.py
```

Runs at: **http://127.0.0.1:5000**

---

## 🖥️ Frontend Setup

### Install & run
```bash
cd frontend
npm install
npm run dev
```
Runs at: **http://localhost:5173**

---

## 🔮 API Endpoints

### 🔹 POST /predict
Upload image → returns plant info.

### 🔹 POST /api/get-plant-uses
Fetch uses from DB / Gemini.

### 🔹 POST /api/chatbot
Chatbot Ayurvedic response.

---

## 🧠 AI Model Details
- Base: **MobileNetV2**
- Image size: **224×224**
- 80 plant classes
- labels.txt format:
  ```
  0 Allium_cepa_(Onion)
  1 Azadirachta_indica_(Neem)
  2 Carica_papaya_(Papaya)
  ```

---

## ⭐ Example Output

### 🌿 Plant Identification
```json
{
  "scientific_name": "Ficus_religiosa",
  "common_name": "Peepal",
  "uses": "Helps in skin issues, diabetes, cough, and inflammation"
}
```

### 🤖 Chatbot Response
```
Fever often signals pitta imbalance. Try Guduchi 3g twice daily.
Consult a doctor for persistent symptoms.
```

---

## 💡 Made with AI + Ayurveda = Smarter healing 🌿