# ATS Keywords & Technologies Extractor

A modern, browser-friendly Streamlit web app that **extracts repeated words, phrases, and ATS-relevant technologies from job descriptions**.  
Easily discover key skills and optimize your resume for applicant tracking systems (ATS).  
Supports AI-powered technology extraction via OpenAI GPT-4o.  
Stores your OpenAI key securely in your browser (never sent to us or any backend).

## 🚀 Features

- Paste any job description: Full-height, responsive text area.
- Repeated words/phrases analysis: Instantly see top keywords, bigrams, trigrams, and more.
- AI-powered technology detection: Uses GPT-4o to extract tools, languages, frameworks, and platforms mentioned in the JD.
- Side-by-side, mobile-friendly layout.
- OpenAI key field is auto-filled via browser storage (using streamlit-local-storage).
- No backend required – everything runs in your browser.

## 🛠️ Requirements

- Python 3.8+
- uv (fast dependency management)
- Streamlit
- nltk
- openai
- python-dotenv
- streamlit-local-storage

## ⚡ Quick Start

1. Clone the repository  
    ```bash
    git clone https://github.com/soumith2105/ats-keywords-extractor.git  
    cd ats-keywords-extractor
    ```

2. Install dependencies using uv  
    ```bash
    uv pip install -r requirements.txt
    ```

3. Run the Streamlit app  
    ```bash
    streamlit run main.py
    ```

4. (First time) Download NLTK data  
   The app will automatically download `punkt` and `stopwords` if not present.

5. Open in your browser  
   Go to http://localhost:8501 if not opened automatically.

## 🔑 OpenAI API Key

- Your OpenAI API key is only stored in your browser’s local storage.
- Get a key at https://platform.openai.com/api-keys
- No key is ever sent to a server except OpenAI for inference.

## 📱 Mobile-Friendly

- Layout is fully responsive.
- Works great on desktop and mobile browsers.

## 📦 Deploy

- The app can be run on Streamlit Community Cloud or any standard Python server.
- Push your code to GitHub, add your dependencies to requirements.txt, and deploy!

## 📝 Example Usage

1. Paste a job description on the left.
2. Enter your OpenAI API key (shown next to the app title).
3. View the extracted keywords, phrases, and detected technologies on the right.
4. Copy/paste results into your resume or for your job search analysis!

## 🙏 Credits

- Built with Streamlit
- Browser storage via streamlit-local-storage
- AI extraction powered by OpenAI GPT-4o

## 🤝 Contributing

Contributions and suggestions welcome!  
Open an issue or PR to help improve the app.