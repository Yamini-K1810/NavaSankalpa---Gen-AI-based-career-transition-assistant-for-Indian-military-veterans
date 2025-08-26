# NavaSankalpa---Gen-AI-based-career-transition-assistant-for-Indian-military-veterans

## 📌 Overview  
**Nava Sankalpa** is an AI-powered web application designed to assist **Indian military veterans** in transitioning to civilian careers.  
The platform leverages **Generative AI, NLP, and vector-based search** to help veterans translate their military experience into **civilian-friendly resumes**, receive **job recommendations**, and interact with an **AI-powered resume chatbot**.  

---

## ✨ Features  
- 🔐 **User Authentication** – Secure login/signup with password hashing.  
- 👤 **Profile Management** – Veterans can create structured personal, military, and professional profiles.  
- 📄 **AI-Powered Resume Generation** – Converts military terminology into **ATS-optimized civilian resumes**.  
- 📂 **Resume Upload & Parsing** – Extracts structured information from uploaded resumes.  
- 💬 **Interactive Resume Chatbot** – Context-aware chatbot answers queries about uploaded CVs.  
- 🎯 **Personalized Job Recommendations** – Suggests suitable job roles, salary ranges, and upskilling paths.  
- 📚 **Resource Hub** – Curated guides for resume writing, interviews, and networking.  
- 📑 **PDF Export** – Professionally formatted resumes generated with **WeasyPrint / FPDF**.  

---

## 🛠️ Tools & Frameworks  

| Category                  | Tools / Frameworks                                                                 |
|---------------------------|------------------------------------------------------------------------------------|
| **Frontend / UI**         | Streamlit                                                                          |
| **AI / NLP**              | LangChain, Groq Inference Engine, Meta LLaMA-4 (LLM)                               |
| **Embeddings & Search**   | HuggingFace (MiniLM), FAISS                                                        |
| **Document Processing**   | PyPDFLoader, RecursiveCharacterTextSplitter                                        |
| **PDF Generation**        | WeasyPrint (primary), FPDF (fallback)                                              |
| **Authentication & DB**   | hashlib (password hashing), JSON DB (local storage)                                |
| **Web Scraping (Future)** | requests, BeautifulSoup                                                            |
| **UI/UX Design**          | Segoe UI / Roboto fonts, Pastel colors, Card-based layout, Subtle Indian Army logo |

---

## ⚙️ Architecture Workflow  

1. **User Authentication** → Secure login using hashed credentials.  
2. **Profile Creation** → Veterans input personal + military details.  
3. **Resume Upload** → Resume PDF parsed into text.  
4. **Resume Information Extraction** → AI extracts key fields.  
5. **Resume Chatbot** → FAISS + embeddings allow contextual Q&A.  
6. **AI Resume Generation** → LLM generates new ATS-friendly CV.  
7. **Job Recommendations** → AI suggests jobs, salaries, and courses.  
8. **Resource Hub** → Access career transition resources.  
9. **PDF Export** → Generate polished PDF resumes.  

---

## 📊 Performance Highlights  
- ⏱️ Resume extraction latency: **<0.5 sec** (Groq accelerated).  
- 🎯 Extraction accuracy: **F1 score up to 1.00** on structured fields.  
- 🧠 Context-aware chatbot powered by **FAISS + LangChain**.  

---

## 📌 Future Enhancements  
- 🔗 Integration with live job portals (via APIs / web scraping).  
- ☁️ Migration from JSON DB → Cloud-based SQL/NoSQL storage.  
- 📊 User dashboards with analytics and career tracking.  
- 🤝 Integration with external career counseling platforms.  
- 🚀 Cloud deployment (AWS/GCP/Azure).  

---
