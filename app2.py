import streamlit as st
import random
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.embeddings import HuggingFaceEmbeddings
import os
import tempfile
import requests
from bs4 import BeautifulSoup
from datetime import datetime
from fpdf import FPDF
import base64
import json
from pathlib import Path
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import time
from threading import Thread
from queue import Queue

# Configuration
st.set_page_config(page_title="Veteran Career Assistant", layout="wide")
st.title("Military-to-Civilian Career Transition Assistant")

# Notification Settings
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587
SMTP_USERNAME = "revanthkamalakar290@gmail.com"
SMTP_PASSWORD = "Revanthk@2001"  # In production, use environment variables
SENDER_EMAIL = "revanthkamalakar290@gmail.com"

# Job scraping APIs (placeholder - replace with actual APIs in production)
JOB_SCRAPING_APIS = {
    "indeed": "https://www.indeed.com/jobs?q={query}&l={location}",
    "naukri": "https://www.naukri.com/{query}-jobs",
    "shine": "https://www.shine.com/job-search/{query}-jobs"
}

USER_DB_FILE = "user_db.json"
JOB_DB_FILE = "job_db.json"

# Session State Init
if "conversation_stage" not in st.session_state:
    st.session_state.conversation_stage = "login"
    st.session_state.authenticated = False
    st.session_state.current_user = None
    st.session_state.user_profile = {
        "name": "", "rank": "", "branch": "",
        "service_years": "", "skills": "",
        "certifications": "", "desired_role": "", 
        "location_preference": "", "salary_expectation": ""
    }
    st.session_state.generated_cv = None
    st.session_state.job_matches = []
    st.session_state.chat_history = []
    st.session_state.last_notification_check = 0
    st.session_state.recommendation_thread = None
    st.session_state.new_jobs_available = False

# LLM Init
groq_api_key = "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"
llm = ChatGroq(model="meta-llama/llama-4-scout-17b-16e-instruct", groq_api_key=groq_api_key, temperature=0.3)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Real-time Job Queue
job_queue = Queue()

# ------------------------- Notification System -------------------------

def send_email_notification(recipient_email, subject, message):
    try:
        msg = MIMEMultipart()
        msg['From'] = SENDER_EMAIL
        msg['To'] = recipient_email
        msg['Subject'] = subject
        
        # HTML email template
        html = f"""
        <html>
            <body>
                <div style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto;">
                    <div style="background-color: #f8f9fa; padding: 20px; border-radius: 5px;">
                        <h2 style="color: #0056b3;">Veteran Career Assistant</h2>
                        <p>{message}</p>
                        <p style="font-size: 0.9em; color: #6c757d;">
                            This is an automated message. Please do not reply directly to this email.
                        </p>
                    </div>
                </div>
            </body>
        </html>
        """
        
        msg.attach(MIMEText(html, 'html'))
        server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
        server.starttls()
        server.login(SMTP_USERNAME, SMTP_PASSWORD)
        server.sendmail(SENDER_EMAIL, recipient_email, msg.as_string())
        server.quit()
        return True
    except Exception as e:
        st.error(f"Failed to send notification: {str(e)}")
        return False

def simulate_sms_notification(mobile_number, message):
    """In production, replace with actual SMS gateway integration"""
    st.sidebar.info(f"Simulated SMS to {mobile_number}: {message}")
    return True

def notify_user(step, user_email=None, mobile_number=None):
    user_name = st.session_state.user_profile.get("name", "Valued Veteran")
    notifications = {
        "profile_complete": {
            "subject": "Profile Completed - Next Steps",
            "message": f"Dear {user_name},\n\nYour military profile has been successfully saved. We're now generating your personalized CV to highlight your unique skills and experience for civilian employers.\n\nYou'll receive another notification when your CV is ready for review.",
            "sms": f"Hi {user_name}, your profile is complete! CV generation in progress."
        },
        "cv_generated": {
            "subject": "Your CV Is Ready for Download",
            "message": f"Dear {user_name},\n\nYour military-to-civilian CV has been generated and is now available for download. We've tailored it to emphasize transferable skills that civilian employers value.\n\nLog in to review your CV and explore job matches we've identified based on your profile.",
            "sms": f"{user_name}, your CV is ready! Log in to download and view job matches."
        },
        "job_matched": {
            "subject": "New Job Matches Available",
            "message": f"Dear {user_name},\n\nWe've found new job opportunities that match your military experience and desired civilian role. These positions value your unique skills and background.\n\nLog in to view the matches and apply directly through our platform.",
            "sms": f"New job matches found for {user_name}! Check your dashboard."
        },
        "new_jobs_available": {
            "subject": "New Opportunities Matching Your Profile",
            "message": f"Dear {user_name},\n\nNew job postings matching your skills and preferences have been added to our system. We've highlighted the most relevant opportunities for you.\n\nLog in to explore these new options and take the next step in your civilian career.",
            "sms": f"New jobs matching your profile, {user_name}! Check your dashboard."
        }
    }
    
    if step in notifications:
        note = notifications[step]
        
        # Send email if available
        if user_email:
            if send_email_notification(user_email, note["subject"], note["message"]):
                st.sidebar.success(f"Email notification sent to {user_email}")
            else:
                st.sidebar.warning("Couldn't send email notification")
        
        # Send SMS if mobile available
        if mobile_number and "sms" in note:
            if simulate_sms_notification(mobile_number, note["sms"]):
                st.sidebar.success(f"SMS notification sent to {mobile_number}")

# ------------------------- User Authentication -------------------------

def generate_otp():
    return str(random.randint(100000, 999999))

def send_otp_to_mobile(mobile_number, otp):
    """In production, integrate with SMS gateway"""
    st.success(f"Simulated OTP to {mobile_number}: {otp}")

def load_user_db():
    try:
        if not Path(USER_DB_FILE).exists():
            return {}
        with open(USER_DB_FILE, "r") as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Error loading user DB: {str(e)}")
        return {}

def save_user_db(db):
    try:
        with open(USER_DB_FILE, "w") as f:
            json.dump(db, f, indent=2)
    except Exception as e:
        st.error(f"Error saving user DB: {str(e)}")

def register_user_mobile(mobile_number, email=""):
    db = load_user_db()
    if mobile_number in db:
        return False, "Mobile number already registered"
    db[mobile_number] = {
        "email": email, 
        "profile": None,
        "notification_prefs": {
            "email_alerts": True,
            "sms_alerts": True,
            "job_updates": True
        },
        "last_active": datetime.now().isoformat()
    }
    save_user_db(db)
    return True, "Registration successful"

def save_user_profile(mobile_number, profile_data):
    db = load_user_db()
    if mobile_number not in db:
        return False
    db[mobile_number]["profile"] = profile_data
    db[mobile_number]["last_updated"] = datetime.now().isoformat()
    save_user_db(db)
    return True

def get_user_profile(mobile_number):
    db = load_user_db()
    return db.get(mobile_number, {}).get("profile")

def update_user_activity(mobile_number):
    db = load_user_db()
    if mobile_number in db:
        db[mobile_number]["last_active"] = datetime.now().isoformat()
        save_user_db(db)

def handle_login():
    st.subheader("Login or Register with Mobile Number")
    login_tab, register_tab = st.tabs(["Login", "Register"])

    with login_tab:
        mobile = st.text_input("Mobile Number", max_chars=10, key="login_mobile", placeholder="10-digit mobile number")
        if st.button("Send OTP", key="login_send_otp"):
            if not mobile or not mobile.isdigit() or len(mobile) != 10:
                st.error("Please enter a valid 10-digit mobile number")
            else:
                otp = generate_otp()
                st.session_state.sent_otp = otp
                st.session_state.otp_mobile = mobile
                send_otp_to_mobile(mobile, otp)
                st.success("OTP sent successfully!")

        if "sent_otp" in st.session_state and st.session_state.get("otp_mobile") == mobile:
            entered_otp = st.text_input("Enter OTP", key="login_otp", placeholder="6-digit OTP")
            if st.button("Verify OTP", key="login_verify"):
                if entered_otp == st.session_state.sent_otp:
                    db = load_user_db()
                    st.session_state.authenticated = True
                    st.session_state.current_user = st.session_state.otp_mobile
                    st.session_state.user_email = db.get(mobile, {}).get("email", "")
                    profile = get_user_profile(mobile)
                    st.session_state.user_profile = profile if profile else st.session_state.user_profile
                    st.session_state.conversation_stage = "cv_generation" if profile else "profile_input"
                    update_user_activity(mobile)
                    st.rerun()
                else:
                    st.error("Invalid OTP. Please try again.")

    with register_tab:
        new_mobile = st.text_input("Mobile Number", max_chars=10, key="reg_mobile", placeholder="10-digit mobile number")
        email = st.text_input("Email Address (recommended)", key="reg_email", placeholder="For important updates")
        if st.button("Send OTP", key="reg_send_otp"):
            if not new_mobile or not new_mobile.isdigit() or len(new_mobile) != 10:
                st.error("Please enter a valid 10-digit mobile number")
            else:
                otp = generate_otp()
                st.session_state.sent_otp = otp
                st.session_state.otp_mobile = new_mobile
                send_otp_to_mobile(new_mobile, otp)
                st.success("OTP sent successfully!")

        if "sent_otp" in st.session_state and st.session_state.get("otp_mobile") == new_mobile:
            entered_otp = st.text_input("Enter OTP", key="reg_otp", placeholder="6-digit OTP")
            if st.button("Verify & Register", key="reg_verify"):
                if entered_otp == st.session_state.sent_otp:
                    success, msg = register_user_mobile(new_mobile, email)
                    if success:
                        st.success(msg)
                        st.session_state.authenticated = True
                        st.session_state.current_user = new_mobile
                        st.session_state.user_email = email
                        st.session_state.conversation_stage = "profile_input"
                        update_user_activity(new_mobile)
                        st.rerun()
                    else:
                        st.error(msg)
                else:
                    st.error("Invalid OTP. Please try again.")

# ------------------------- Profile Management -------------------------

def handle_profile_input():
    st.subheader("Step 1: Tell Us About Your Military Service")
    with st.form("profile_form"):
        cols = st.columns(2)
        with cols[0]:
            st.session_state.user_profile["name"] = st.text_input("Full Name*", help="As it should appear on your CV")
            st.session_state.user_profile["rank"] = st.text_input("Rank* (e.g., Subedar, Naik)", help="Your highest achieved rank")
            st.session_state.user_profile["branch"] = st.selectbox(
                "Service Branch*", 
                ["Indian Army", "Indian Navy", "Indian Air Force", "Paramilitary Forces"],
                help="Which branch of service you were in"
            )
            st.session_state.user_profile["service_years"] = st.number_input(
                "Years of Service*", 
                min_value=1, max_value=30,
                help="Total years served"
            )
        with cols[1]:
            st.session_state.user_profile["skills"] = st.text_area(
                "Key Skills* (comma separated)", 
                help="Technical, leadership, and transferable skills"
            )
            st.session_state.user_profile["certifications"] = st.text_input(
                "Certifications", 
                help="Any professional certifications earned"
            )
            st.session_state.user_profile["desired_role"] = st.text_input(
                "Desired Civilian Role*", 
                help="Target job title or field"
            )
            st.session_state.user_profile["location_preference"] = st.text_input(
                "Location Preference", 
                help="Preferred cities or regions for work"
            )
            st.session_state.user_profile["salary_expectation"] = st.text_input(
                "Salary Expectation", 
                help="Expected or desired salary range"
            )
        
        if st.form_submit_button("Save Profile & Continue"):
            if not st.session_state.user_profile["name"] or not st.session_state.user_profile["rank"]:
                st.error("Please fill in all required fields (marked with *)")
            else:
                save_user_profile(st.session_state.current_user, st.session_state.user_profile)
                notify_user(
                    "profile_complete", 
                    st.session_state.user_email,
                    st.session_state.current_user
                )
                st.session_state.conversation_stage = "cv_generation"
                st.rerun()

# ------------------------- CV Generation -------------------------

def generate_pdf_cv(user_email, mobile_number):
    profile = st.session_state.user_profile
    prompt = f"""
    Create a professional CV for an Indian servicemember transitioning to civilian life with these details:
    
    Personal Information:
    - Name: {profile['name']}
    - Contact: {mobile_number} | {user_email if user_email else 'Email not provided'}
    
    Military Service:
    - Rank: {profile['rank']} ({profile['branch']})
    - Years of Service: {profile['service_years']} years
    - Key Achievements: [Generate 3-5 bullet points highlighting leadership, operations, and special projects]
    
    Skills:
    - Technical: {profile['skills']}
    - Transferable: Leadership, Teamwork, Problem-solving, Adaptability
    - Certifications: {profile['certifications'] if profile['certifications'] else 'None listed'}
    
    Career Objective:
    - Seeking {profile['desired_role']} position where military experience in [relevant skills] can contribute to [industry/organization type].
    
    Additional Sections:
    - Education: [Assume standard military training if not provided]
    - Languages: [Assume Hindi and English proficiency]
    - References: Available upon request
    
    Formatting Guidelines:
    - Use professional, clean layout
    - Highlight transferable skills
    - Use action verbs (led, managed, implemented)
    - Keep to 1-2 pages
    - Include a skills matrix matching military to civilian skills
    """
    
    with st.spinner("Generating your professional CV..."):
        response = llm.invoke(prompt)
        
        # Create PDF
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        
        # Header
        pdf.set_font("Arial", 'B', 16)
        pdf.cell(200, 10, txt=profile['name'], ln=True, align='C')
        pdf.set_font("Arial", size=12)
        contact_info = f"{mobile_number} | {user_email}" if user_email else mobile_number
        pdf.cell(200, 10, txt=contact_info, ln=True, align='C')
        pdf.ln(10)
        
        # Add content
        for line in response.content.split('\n'):
            if line.strip().endswith(':'):
                pdf.set_font("Arial", 'B', 14)
                pdf.cell(200, 10, txt=line, ln=True)
                pdf.set_font("Arial", size=12)
            else:
                pdf.multi_cell(0, 10, txt=line)
        
        # Save PDF
        pdf_output = f"temp_cv_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        pdf.output(pdf_output)
        
        notify_user("cv_generated", user_email, mobile_number)
        return pdf_output

def handle_cv_generation():
    st.subheader("Step 2: Your Personalized CV")
    if not st.session_state.generated_cv:
        with st.spinner("Generating your military-to-civilian CV..."):
            st.session_state.generated_cv = generate_pdf_cv(
                st.session_state.user_email,
                st.session_state.current_user
            )
    
    # Display CV preview
    with open(st.session_state.generated_cv, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')
        pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600" type="application/pdf"></iframe>'
        st.markdown(pdf_display, unsafe_allow_html=True)
    
    # Download button
    with open(st.session_state.generated_cv, "rb") as f:
        st.download_button(
            label="Download Your CV",
            data=f,
            file_name=f"Military_CV_{st.session_state.user_profile['name'].replace(' ', '_')}.pdf",
            mime="application/pdf"
        )
    
    if st.button("Find Job Matches"):
        st.session_state.conversation_stage = "job_matching"
        st.rerun()

# ------------------------- Job Matching System -------------------------

def scrape_indian_job_sites(query=None, location=None):
    """Enhanced job scraping function with simulated real-time updates"""
    try:
        # In production, replace with actual API calls to job sites
        simulated_jobs = [
            {
                "title": "Operations Manager",
                "company": "Tata Group",
                "location": "Delhi, Mumbai, Bangalore",
                "description": "Seeking candidates with leadership experience and operational expertise. Military veterans encouraged to apply.",
                "posted_date": datetime.now().strftime("%Y-%m-%d"),
                "salary": "₹8L - ₹12L PA",
                "skills": ["Leadership", "Operations Management", "Team Coordination"],
                "source": "Simulated Portal",
                "url": "#"
            },
            {
                "title": "Security Consultant",
                "company": "SecureTech Solutions",
                "location": "Hyderabad, Pune",
                "description": "Security consulting position ideal for veterans with field experience. Certifications preferred but not required.",
                "posted_date": (datetime.now()),
                "salary": "₹6L - ₹10L PA",
                "skills": ["Security", "Risk Assessment", "Crisis Management"],
                "source": "Simulated Portal",
                "url": "#"
            },
            {
                "title": "Logistics Coordinator",
                "company": "Reliance Industries",
                "location": "Mumbai, Chennai",
                "description": "Coordination role requiring organizational skills and supply chain knowledge. Veterans with logistics experience preferred.",
                "posted_date": (datetime.now()),
                "salary": "₹5L - ₹9L PA",
                "skills": ["Logistics", "Supply Chain", "Inventory Management"],
                "source": "Simulated Portal",
                "url": "#"
            },
            {
                "title": "Technical Trainer",
                "company": "TechSkill Academy",
                "location": "Bangalore, Remote",
                "description": "Training position for candidates with technical expertise and teaching ability. Excellent for veterans with instructional experience.",
                "posted_date": datetime.now().strftime("%Y-%m-%d"),
                "salary": "₹7L - ₹11L PA",
                "skills": ["Technical Training", "Curriculum Development", "Public Speaking"],
                "source": "Simulated Portal",
                "url": "#"
            }
        ]
        
        # Filter based on query if provided
        if query:
            query = query.lower()
            simulated_jobs = [job for job in simulated_jobs 
                             if query in job['title'].lower() 
                             or query in job['description'].lower()]
        
        # Filter based on location if provided
        if location:
            location = location.lower()
            simulated_jobs = [job for job in simulated_jobs 
                             if location in job['location'].lower()]
        
        return simulated_jobs
    except Exception as e:
        st.error(f"Job scraping error: {str(e)}")
        return []

def background_job_scraper(user_profile):
    """Run in background to continuously look for new jobs"""
    try:
        while True:
            # Get fresh job matches
            query = user_profile.get("desired_role", "")
            location = user_profile.get("location_preference", "")
            new_jobs = scrape_indian_job_sites(query, location)
            
            # Compare with existing jobs
            existing_jobs = []
            if Path(JOB_DB_FILE).exists():
                with open(JOB_DB_FILE, "r") as f:
                    existing_jobs = json.load(f).get(st.session_state.current_user, [])
            
            # Find truly new jobs
            new_job_ids = {job['title']+job['company'] for job in new_jobs}
            existing_job_ids = {job['title']+job['company'] for job in existing_jobs}
            truly_new_jobs = [job for job in new_jobs 
                            if job['title']+job['company'] not in existing_job_ids]
            
            if truly_new_jobs:
                # Update job DB
                all_jobs = existing_jobs + truly_new_jobs
                job_db = load_job_db()
                job_db[st.session_state.current_user] = all_jobs
                save_job_db(job_db)
                
                # Notify user
                st.session_state.new_jobs_available = True
                notify_user(
                    "new_jobs_available",
                    st.session_state.user_email,
                    st.session_state.current_user
                )
            
            # Sleep for a while before checking again
            time.sleep(3600)  # Check every hour
    except Exception as e:
        st.error(f"Background job scraper error: {str(e)}")

def load_job_db():
    try:
        if not Path(JOB_DB_FILE).exists():
            return {}
        with open(JOB_DB_FILE, "r") as f:
            return json.load(f)
    except Exception:
        return {}

def save_job_db(db):
    with open(JOB_DB_FILE, "w") as f:
        json.dump(db, f, indent=2)

def match_jobs_to_profile(user_email, mobile_number):
    profile = st.session_state.user_profile
    query = profile.get("desired_role", "")
    location = profile.get("location_preference", "")
    
    # Get jobs from DB or scrape fresh
    job_db = load_job_db()
    all_jobs = job_db.get(st.session_state.current_user, [])
    
    if not all_jobs:
        all_jobs = scrape_indian_job_sites(query, location)
        job_db[st.session_state.current_user] = all_jobs
        save_job_db(job_db)
    
    # Start background job checker if not already running
    if not st.session_state.recommendation_thread:
        st.session_state.recommendation_thread = Thread(
            target=background_job_scraper,
            args=(st.session_state.user_profile,),
            daemon=True
        )
        st.session_state.recommendation_thread.start()
    
    # Prepare prompt for LLM to match jobs
    prompt = f"""
    Match this veteran's profile to available jobs and return the top 3-5 matches with detailed explanations:
    
    Veteran Profile:
    - Name: {profile['name']}
    - Military Background: {profile['rank']} in {profile['branch']} for {profile['service_years']} years
    - Skills: {profile['skills']}
    - Certifications: {profile['certifications']}
    - Desired Role: {profile['desired_role']}
    - Location Preference: {profile.get('location_preference', 'Not specified')}
    
    Available Jobs:
    {json.dumps(all_jobs, indent=2)}
    
    Analysis Guidelines:
    1. Focus on transferable skills from military to civilian roles
    2. Highlight how their specific experience matches each role
    3. Consider location preferences if specified
    4. Provide confidence score (1-5) for each match
    5. Include direct comparison of required vs. possessed skills
    6. Suggest any skill gaps and how to address them
    
    Output Format:
    - For each match: Job Title, Company, Match Reasons, Confidence Score
    - Overall analysis of fit
    - Recommended next steps
    """
    
    with st.spinner("Analyzing job matches..."):
        response = llm.invoke(prompt)
        notify_user("job_matched", user_email, mobile_number)
        return response.content

def display_job_matches(job_matches):
    """Parse and display job matches in a user-friendly format"""
    st.subheader("Your Best Job Matches")
    
    # Parse the LLM response (this is simplified - in production you'd want more robust parsing)
    sections = job_matches.split("\n\n")
    
    for section in sections:
        if "Job Title:" in section:
            # This is a job match entry
            st.markdown("---")
            lines = section.split("\n")
            title = lines[0].replace("Job Title:", "").strip()
            company = lines[1].replace("Company:", "").strip()
            
            st.markdown(f"### {title}")
            st.markdown(f"**Company:** {company}")
            
            # Display the rest of the details
            for line in lines[2:]:
                if "Confidence Score:" in line:
                    score = int(line.replace("Confidence Score:", "").strip())
                    st.progress(score/5)
                else:
                    st.markdown(line)
        else:
            # General information or analysis
            st.markdown(section)

def handle_job_matching():
    if not st.session_state.job_matches:
        with st.spinner("Finding jobs matching your profile..."):
            st.session_state.job_matches = match_jobs_to_profile(
                st.session_state.user_email,
                st.session_state.current_user
            )
    
    # Check for new jobs notification
    if st.session_state.new_jobs_available:
        st.success("New job matches available! Refresh to view.")
        st.session_state.new_jobs_available = False
    
    display_job_matches(st.session_state.job_matches)
    
    if st.button("Refresh Job Matches"):
        st.session_state.job_matches = []
        st.rerun()
    
    if st.button("Start Over"):
        st.session_state.conversation_stage = "profile_input"
        st.session_state.generated_cv = None
        st.session_state.job_matches = []
        st.rerun()

# ------------------------- CV Q&A Assistant -------------------------

def handle_cv_qa():
    st.subheader("CV Q&A Assistant")
    st.info("Ask questions about your CV content or how to improve it for civilian roles.")
    
    # Load CV content
    loader = PyPDFLoader(st.session_state.generated_cv)
    pages = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = text_splitter.split_documents(pages)
    db = FAISS.from_documents(docs, embeddings)
    
    if user_query := st.chat_input("Ask about your CV..."):
        # Add user question to chat history
        st.session_state.chat_history.append({"role": "user", "content": user_query})
        st.chat_message("user").write(user_query)
        
        # Get relevant context from CV
        docs = db.similarity_search(user_query)
        context = "\n".join([d.page_content for d in docs])
        
        # Generate response
        prompt = f"""
        You're a career advisor helping a military veteran transition to civilian work. 
        Answer their question about their CV with helpful, specific advice.
        
        Question: {user_query}
        
        Relevant CV Context:
        {context}
        
        Guidelines:
        - Focus on translating military experience to civilian terms
        - Suggest specific improvements if asked
        - Be encouraging and professional
        - Keep responses concise but thorough
        """
        
        with st.spinner("Thinking..."):
            response = llm.invoke(prompt)
            st.session_state.chat_history.append({"role": "assistant", "content": response.content})
            st.chat_message("assistant").write(response.content)

# ------------------------- Main App Flow -------------------------

if not st.session_state.authenticated:
    handle_login()
else:
    # Sidebar with user info and navigation
    st.sidebar.header(f"Welcome, {st.session_state.user_profile.get('name', 'Veteran')}")
    st.sidebar.write(f"Service: {st.session_state.user_profile.get('rank', '')} ({st.session_state.user_profile.get('branch', '')})")
    
    if st.sidebar.button("Logout"):
        st.session_state.authenticated = False
        st.session_state.current_user = None
        st.session_state.conversation_stage = "login"
        st.rerun()
    
    # Progress tracker
    st.sidebar.header("Your Progress")
    stages = {
        "profile_input": "1. Profile Setup",
        "cv_generation": "2. CV Generated",
        "job_matching": "3. Job Matches"
    }
    current_stage = st.session_state.conversation_stage
    for stage, label in stages.items():
        if stage == current_stage:
            st.sidebar.success(f"✓ {label}")
        elif list(stages.keys()).index(stage) < list(stages.keys()).index(current_stage):
            st.sidebar.info(f"✓ {label}")
        else:
            st.sidebar.write(f"◻ {label}")
    
    # Main content area
    if st.session_state.conversation_stage == "profile_input":
        handle_profile_input()
    elif st.session_state.conversation_stage == "cv_generation":
        handle_cv_generation()
    elif st.session_state.conversation_stage == "job_matching":
        handle_job_matching()
    
    # CV Q&A toggle
    if st.session_state.generated_cv and st.checkbox("Ask CV Assistant", key="cv_qa_toggle"):
        handle_cv_qa()
    
    # Display chat history if exists
    if st.session_state.chat_history:
        st.sidebar.header("Chat History")
        for msg in st.session_state.chat_history[-5:]:  # Show last 5 messages
            st.sidebar.write(f"**{msg['role'].title()}**: {msg['content']}")