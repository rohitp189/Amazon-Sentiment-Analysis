# 🛒 Amazon Sentiment Analysis & Fake Review Detection  

## 📌 About the Project  
This application is designed to **analyze Amazon product reviews, detect fake reviews, and provide valuable customer sentiment insights**.  

By leveraging **AI-powered sentiment analysis** and **fake review detection**, this system ensures **authentic product feedback**, helping businesses and customers make **informed purchasing decisions**.  

---

## 🌟 Key Features  
✔ **Sentiment Analysis** – Identifies **Positive, Neutral, and Negative** reviews.  
✔ **Fake Review Detection** – Spots **unverified or misleading** reviews.  
✔ **Feature Analysis** – Highlights **key pros and cons** of a product.  
✔ **Trust Score Calculation** – Generates an **overall trust score** based on review authenticity and sentiment.  
✔ **Comprehensive Review Breakdown** – Provides **detailed sentiment distribution and authenticity analysis**.  

---

## 🔍 How Does It Work?  
📂 **Upload Amazon Reviews (CSV)** – Select and upload a CSV file containing product reviews. The system will automatically process the data.  

🧠 **AI-Powered Analysis** – The model classifies sentiment (**Positive, Neutral, Negative**) and detects **fake or unreliable reviews**.  

📊 **Detailed Review Breakdown** – The system presents a summary of **sentiment scores, authenticity checks, and key review insights**.  

🔎 **Final Trust Score** – A **comprehensive trust score** is computed, combining sentiment analysis and fake review detection.  

---

## 🛠️ Technologies Used  
🔹 **Python** – Core programming language  
🔹 **Web Interface** – Streamlit for an interactive and user-friendly UI  
🔹 **Deep Learning** – PyTorch & Transformers (DistilBERT) for sentiment classification and fake review detection  
🔹 **Data Processing** – Pandas & NumPy for handling and structuring data  
🔹 **Data Visualization** – Plotly for interactive graphs and charts  

---

## 🔧 Installation & Setup  

1️⃣ **Clone this repository**  
   - Download or clone the project from GitHub.  

2️⃣ **Set up a Virtual Environment**  
   - On **Windows**:  
     ```bash
     python -m venv venv  
     venv\Scripts\activate  
     ```
   - On **macOS/Linux**:  
     ```bash
     python3 -m venv venv  
     source venv/bin/activate  
     ```

3️⃣ **Install Dependencies**  
   - Install the required libraries using:  
     ```bash
     pip install -r requirements.txt  
     ```

4️⃣ **Run the Streamlit Application**  
   - Execute the app using:  
     ```bash
     streamlit run app.py  
     ```

---

## 📊 Results & Insights  
✅ **85% accuracy** achieved in sentiment classification and fake review detection.  
✅ The system effectively identified **potentially unreliable reviews**, improving the authenticity of product feedback.  
✅ The **Trust Score** feature effectively helped identify reliable products, enhancing users' confidence in their purchasing decisions.  
 

---

## 🚀 Future Improvements  
🔹 **Improve Fake Review Detection** – Integrate advanced models like **LSTM/BERT** for better accuracy.  
🔹 **Enhance UI/UX** – Improve the **Streamlit** interface for a more user-friendly experience.  
🔹 **Real-time Amazon Review Scraping** – Enable dynamic fetching and analysis of Amazon reviews.  
🔹 **Expand Dataset** – Increase dataset diversity for **better model generalization across various product categories**.  

---

## 🤝 Contributing  
We welcome contributions! If you'd like to improve this project, feel free to **fork the repo, raise issues, or submit pull requests**.  

---

## 📜 License  
This project is **open-source** under the **MIT License**.  
