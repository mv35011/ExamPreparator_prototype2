# 📚 Exam Preparator

### AI-powered Study Assistant for Efficient Exam Preparation

## 🚀 Project Overview
Exam Preparator is an AI-driven web application designed to help students efficiently prepare for their exams. By leveraging advanced Natural Language Processing (NLP) and Computer Vision (CV), the system extracts key information from past question papers, syllabus notes, and study materials. It then generates important formulas, derivations, and potential exam questions, allowing students to focus on what matters most.

## 🌟 Features
- **📝 Subject Profiles** – Create and manage subjects for different courses.
- **📄 Upload Study Materials** – Supports PDFs and image-based documents.
- **🔍 AI-powered Extraction** – Extracts key formulas, derivations, and important questions.
- **📊 Topic Summarization** – Provides concise study notes from long-form content.
- **🤖 LLM-Powered Analysis** – Uses finetuned mistral-8b and Deepseek r1 as a fallback
- **🖼️ OCR Support** – Extracts text from images using Computer Vision.

## 🏗️ Tech Stack
- **Backend:** Streamlit
- **Frontend:** Streamlit frontend
- **AI/ML:** Deepseek LLM (via Groq API), LangChain, Hugging Face Transformers, mistral-8b
- **Database:** PostgreSQL / SQLite (for development)
- **Deployment:** Docker, AWS / Vercel (planned)

## 📦 Installation & Setup
### 🔧 Prerequisites
- Python 3.10
- Pip & Virtual Environment
- PostgreSQL (optional for production)
- Groq API Key for LLM integration

### ⚙️ Setup Instructions
1. **Clone the repository:**
   ```sh
   git clone https://github.com/yourusername/exam-preparator.git
   cd ExamPreparator_prototype2
   ```

2. **Create and activate a virtual environment:**
   ```sh
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install dependencies:**
   ```sh
   pip install -r requirements.txt
   ```

4. **Set up environment variables:** (Create a `.env` file and add the required keys)
   ```sh
   GROQ_API_KEY=your_api_key_here
   DEBUG=True
   HUGGINGFACE_TOKEN=huggingface_token
   ```

5.**Run the app**
  ```
  streamlit run main.py
  ```

## 📜 Usage
1. Upload PDFs or images of question papers and notes.
2. The AI extracts key information and generates study material.
3. View summarized notes and important questions.
4. Use the web app to structure your exam preparation effectively.

## 🚧 Roadmap
- [ ] Implement a user-friendly frontend using React/Next.js.
- [ ] Enhance AI accuracy with fine-tuned models.
- [ ] Add export options (PDF summaries, flashcards, etc.).
- [ ] Deploy a cloud-based version for public use.

## 🤝 Contributing
Contributions are welcome! If you’d like to contribute:
1. Fork the repository.
2. Create a feature branch (`git checkout -b feature-name`).
3. Commit changes (`git commit -m 'Add new feature'`).
4. Push to your branch (`git push origin feature-name`).
5. Open a Pull Request.

## 📄 License
This project is licensed under the MIT License.

## 📞 Contact
- **Author:** Manmohan Vishwakarma1
- **Email:** mv350113@gmail.com
- **GitHub:** [mv3501](https://github.com/mv35011)
- **LinkedIn:** ([https://linkedin.com/in/yourprofile](https://www.linkedin.com/in/manmohan-vishwakarma-4baa99270/))

---
✨ *Empowering students with AI-driven exam preparation!*

