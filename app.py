import streamlit as st
import base64
import sys
import os
import plotly.graph_objects as go
import pymysql
import pandas as pd

# Add src/ folder to system path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from predict_emotion import predict_emotion
from predict_tone import predict_tone
from sentiment_analysis import analyze_sentiment_vader
from formality_score import calculate_formality_score

# App config
st.set_page_config(page_title="Psychological Tone Analyzer", page_icon="🧠", layout="wide")

# Set background image
def set_background(image_file):
    with open(image_file, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/png;base64,{encoded}");
            background-size: cover;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

set_background("assets/7774747.jpg")  # Replace with your own image path

# Database connection
def connect_db():
    return pymysql.connect(
        host="localhost",
        user="root",
        password="Varsh@24",  # Update your password if needed
        database="tone_analyzer_db",
        charset='utf8mb4',
        cursorclass=pymysql.cursors.Cursor
    )

# Login page
def login():
    st.title("🔐 Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if username == "admin" and password == "admin123":
            st.session_state.logged_in = True
        else:
            st.error("Invalid username or password")

# Main page for analysis
def main():
    st.title("📧 Psychological Tone Analyzer for Emails")
    st.write("This tool helps detect **emotion**, **tone**, **sentiment**, and **formality** in emails.")

    with st.form(key="email_form"):
        email_text = st.text_area("✉️ Enter Email Content", height=300)
        submit_button = st.form_submit_button("🔍 Analyze")

    if submit_button and email_text.strip():
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.subheader("🧠 Emotion")
            emotion = predict_emotion(email_text)
            st.success(emotion)

        with col2:
            st.subheader("🎯 Tone")
            tone = predict_tone(email_text)
            st.success(tone)

        with col3:
            st.subheader("🔍 Sentiment")
            sentiment_dict = analyze_sentiment_vader(email_text)
            sorted_sentiment = sorted(sentiment_dict.items(), key=lambda x: x[1], reverse=True)
            for k, v in sorted_sentiment:
                st.write(f"**{k.capitalize()}**: {v:.4f}")

        with col4:
            st.subheader("📏 Formality")
            formality = calculate_formality_score(email_text)
            st.metric("Formality Score", f"{formality}/100")

        # Save to database
        try:
            conn = connect_db()
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO email_analysis (email_text, emotion, tone, sentiment, formality)
                VALUES (%s, %s, %s, %s, %s)
            """, (email_text, emotion, tone, str(sentiment_dict), formality))
            conn.commit()
            conn.close()
            st.success("✅ Analysis saved to the database!")
        except Exception as e:
            st.error(f"❌ Failed to save to database: {e}")

        # 💡 Feedback
        st.markdown("### 📋 Feedback Summary")
        feedback = []

        if sentiment_dict["compound"] > 0.5:
            feedback.append("✅ Your email has a strong positive sentiment — great for professional communication.")
        if "fear" in tone.lower():
            feedback.append("😟 Your message might come across as slightly anxious. Consider using more confident language.")
        if formality > 70:
            feedback.append("🧑‍💼 Your email is formal and well-structured — suitable for professional settings.")

        if feedback:
            for item in feedback:
                st.write(item)
        else:
            st.write("✅ No specific feedback — your email looks balanced and appropriate.")

        # 🛠 Style Suggestions
        st.markdown("### 🛠 Style Suggestions")
        suggestions = []
        if sentiment_dict["compound"] < 0.3:
            suggestions.append("- Consider adding more positive phrasing or appreciation.")
        if formality < 60:
            suggestions.append("- Avoid contractions and use more formal sentence structures.")
        if sentiment_dict["neu"] > 0.6:
            suggestions.append("- Add more emotional nuance to connect better with the reader.")

        if suggestions:
            for s in suggestions:
                st.write(s)
        else:
            st.write("✅ No major suggestions — your email looks well-composed!")

        # 📡 Tone Radar Chart
        st.markdown("### 📊 Tone Radar")
        radar_labels = ["Formality", "Positivity", "Neutrality", "Negativity", "Compound"]
        radar_values = [
            formality / 100,
            sentiment_dict["pos"],
            sentiment_dict["neu"],
            sentiment_dict["neg"],
            (sentiment_dict["compound"] + 1) / 2
        ]

        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=radar_values + [radar_values[0]],
            theta=radar_labels + [radar_labels[0]],
            fill='toself',
            name='Email Tone'
        ))
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            showlegend=False,
            height=400
        )
        st.plotly_chart(fig)

# View Past Analyses page
def view_past_analyses():
    st.title("📜 Past Email Analyses")

    try:
        conn = connect_db()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM email_analysis ORDER BY id DESC")
        results = cursor.fetchall()
        conn.close()

        if results:
            df = pd.DataFrame(results, columns=["ID", "Email Text", "Emotion", "Tone", "Sentiment", "Formality Score"])
            st.dataframe(df, use_container_width=True)

            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("⬇️ Download Data as CSV", csv, "email_analysis.csv", "text/csv", key='download-csv')
        else:
            st.info("No past analyses found.")
    except Exception as e:
        st.error(f"Failed to fetch data: {e}")

# Run login or main app
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if st.session_state.logged_in:
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ("Analyze Email", "View Past Analyses"))

    if page == "Analyze Email":
        main()
    elif page == "View Past Analyses":
        view_past_analyses()
else:
    login()
