import os
import fitz  # PyMuPDF
import streamlit as st
from dotenv import load_dotenv
import google.generativeai as genai
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load API key
load_dotenv()
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-1.5-flash')

# Job Description (fixed for demo, can be changed dynamically if needed)
job_description = """
We are hiring a Software Engineer with experience in Python, APIs, data structures, algorithms, databases, and cloud platforms like AWS or GCP.
Candidates should have a Bachelor’s degree in Computer Science or a related field, and at least 1 year of industry experience.
"""

def extract_text_from_pdf(file):
    try:
        text = ""
        doc = fitz.open(stream=file.read(), filetype="pdf")
        for page in doc:
            text += page.get_text()
        return text
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
        return ""

def extract_details_with_gemini(resume_text):
    prompt = f"""
Extract the following details from this resume:
1. Key skills and technologies
2. Total years of experience
3. Educational qualifications
4. Notable projects or achievements

Resume Text:
{resume_text[:4000]}  # truncated to model limit
"""
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        st.error(f"Gemini error: {e}")
        return "Gemini extraction failed."

def compute_similarity(resume_text, job_desc):
    try:
        vectorizer = TfidfVectorizer(stop_words='english')
        vectors = vectorizer.fit_transform([resume_text, job_desc])
        sim_score = cosine_similarity(vectors[0:1], vectors[1:2])
        return sim_score[0][0]
    except Exception as e:
        st.error(f"Similarity error: {e}")
        return 0.0

st.title("📄 Resume Screening and Ranking with Gemini AI")

st.write("""
Upload one or more PDF resumes, and this app will:
- Extract text from the resumes
- Compute similarity score with the job description
- Extract detailed info using Gemini AI
- Rank resumes based on similarity score
""")

uploaded_files = st.file_uploader("Upload PDF resumes", type=["pdf"], accept_multiple_files=True)

if uploaded_files:
    results = []
    with st.spinner("Processing resumes... this may take a while for multiple files."):
        for uploaded_file in uploaded_files:
            text = extract_text_from_pdf(uploaded_file)
            if text:
                score = compute_similarity(text, job_description)
                summary = extract_details_with_gemini(text)
                results.append({
                    "filename": uploaded_file.name,
                    "score": round(score, 2),
                    "summary": summary
                })

    if results:
        results.sort(key=lambda x: x["score"], reverse=True)
        st.header("📊 Resume Ranking Results:")
        for i, res in enumerate(results, 1):
            st.subheader(f"{i}. {res['filename']} — Similarity Score: {res['score']}")
            st.markdown(res["summary"])
else:
    st.info("Please upload PDF resumes to start screening.")
