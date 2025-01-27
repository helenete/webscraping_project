# -- Standard Library Imports --
import re
import string
import unicodedata
from collections import Counter
from urllib.parse import unquote

# -- Third-Party Library Imports --
import pandas as pd
import plotly.express as px
import streamlit as st
from streamlit_option_menu import option_menu
from tqdm import tqdm

# -- NLTK Imports --
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# -- Transformers Imports --
from transformers import pipeline

# -- Download Necessary NLTK Data --
nltk.download('vader_lexicon')
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('omw-1.4')


# Configuration de la page et style
st.set_page_config(page_title="Tourism Analysis Dashboard", layout="wide")

# Custom CSS
st.markdown("""
<style>
    /* Couleurs pastel et style moderne */
    :root {
        --primary-color: #b8e0d2;
        --secondary-color: #eac4d5;
        --background-color: #f7f6f3;
        --text-color: #444444;
    }

    /* Style général */
    .stApp {
        background-color: var(--background-color);
    }

    /* Headers */
    h1, h2, h3 {
        color: #2c3e50;
        font-family: 'Helvetica Neue', sans-serif;
        padding: 1rem 0;
    }

    /* Metrics */
    div[data-testid="stMetricValue"] {
        background-color: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }

    /* Cards */
    div.stBlock {
        background-color: white;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }

    /* Buttons */
    .stButton button {
        background-color: var(--primary-color);
        color: var(--text-color);
        border: none;
        border-radius: 20px;
        padding: 0.5rem 2rem;
        transition: all 0.3s ease;
    }

    .stButton button:hover {
        background-color: var(--secondary-color);
        transform: translateY(-2px);
    }

    /* Selectbox */
    .stSelectbox {
        border-radius: 10px;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }

    .stTabs [data-baseweb="tab"] {
        background-color: transparent;
        border-radius: 4px 4px 0 0;
        padding: 0.5rem 2rem;
        color: var(--text-color);
    }

    .stTabs [aria-selected="true"] {
        background-color: var(--primary-color) !important;
    }
</style>
""", unsafe_allow_html=True)

#################################
# 1. Text Cleaner Class
#################################
class TextCleaner:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set()
        for lang in ['english', 'french', 'spanish', 'portuguese']:
            self.stop_words.update(stopwords.words(lang))

        self.custom_stops = {
            'captcha', 'please', 'confirm', 'human', 'completing', 'challenge',
            'below', 'reference', 'number', 'user', 'agent', 'access', 'click',
            'www', 'http', 'https', 'html', 'php', 'asp', 'com', 'net', 'org',
            'error', 'page', 'loading', 'website', 'browser', 'javascript',
            'cookie', 'cookies', 'privacy', 'policy', 'terms', 'conditions'
        }

        self.stop_words.update(self.custom_stops)

        # Corrected regex patterns (single backslash in raw strings).
        self.url_pattern = re.compile(r'https?://S+|www.S+')
        self.email_pattern = re.compile(r'S+@S+')
        self.special_char_pattern = re.compile(r'[^ws]')
        self.multiple_spaces_pattern = re.compile(r's+')
        self.numeric_pattern = re.compile(r'd+')

    def clean_text(self, text):
        if not isinstance(text, str):
            text = str(text)

        text = text.lower()
        text = self.url_pattern.sub(" ", text)
        text = self.email_pattern.sub(" ", text)
        text = self.remove_html_tags(text)
        text = self.remove_accents(text)
        text = self.special_char_pattern.sub(" ", text)
        text = self.numeric_pattern.sub(" ", text)

        tokens = word_tokenize(text)
        cleaned_tokens = [
            self.lemmatizer.lemmatize(token)
            for token in tokens
            if (len(token) > 2 and
                token not in self.stop_words and
                not token.isnumeric() and
                not all(c in string.punctuation for c in token))
        ]

        cleaned_text = " ".join(cleaned_tokens)
        cleaned_text = self.multiple_spaces_pattern.sub(" ", cleaned_text)
        return cleaned_text.strip()

    def remove_accents(self, text):
        return "".join(
            c for c in unicodedata.normalize("NFKD", text)
            if unicodedata.category(c) != "Mn"
        )

    def remove_html_tags(self, text):
        clean = re.compile("<.*?>")
        return re.sub(clean, "", text)

###########################################
# 2. Tourism Chatbot Class (Summarization)
###########################################
class TourismChatbot:
    def __init__(self):
        # Integrate T5 Large for summarization
        self.summarizer = pipeline(
            "summarization",
            model="t5-large",
            tokenizer="t5-large"
        )
        self.cleaner = TextCleaner()

    def summarize_text(self, text, max_length=130, min_length=30):
        cleaned_text = self.cleaner.clean_text(text)
        chunks = self._split_into_chunks(cleaned_text)
        summaries = []

        for chunk in chunks[:3]:  # limit to first 3 chunks
            if len(chunk) > 100:
                try:
                    summary = self.summarizer(
                        chunk,
                        max_length=max_length,
                        min_length=min_length,
                        do_sample=False
                    )
                    summaries.append(summary[0]["summary_text"])
                except Exception as e:
                    st.error(f"Error in summarization: {str(e)}")

        return " ".join(summaries)

    def _split_into_chunks(self, text, max_chunk_size=1000):
        return [text[i:i + max_chunk_size] for i in range(0, len(text), max_chunk_size)]

#################################
# 3. Sentiment Analysis Pipeline
#################################
@st.cache_resource
def load_sentiment_analyzer():
    """
    Loads a multilingual sentiment analysis pipeline from Hugging Face.
    You can change the model to any other that suits your needs.
    """
    return pipeline("sentiment-analysis", model="cardiffnlp/twitter-xlm-roberta-base-sentiment")

#################################
# 4. Load and Cache Data
#################################
@st.cache_data
def load_data():
    tourism_df = pd.read_csv("data_tourism.csv")
    articles_df = pd.read_csv("articles_tourisme.csv")
    tourism_all_df = pd.read_csv("data_tourism_all.csv")  # Add South American data file

    # Define a mask of rows to drop from articles_df based on 'titre'
    mask_to_drop = (
        (articles_df['titre'] == 'Are you a robot?') |
        (articles_df['titre'] == 'Via') |
        (articles_df['titre'] == "This page isn't working") |
        (articles_df['titre'] == "Page not found") |
        (articles_df['titre'].astype(str).str.lower() == 'nan') |
        (articles_df['titre'].str.contains('Access Check', case=False, na=False))
    )
    # Keep only rows that do NOT match the conditions above
    articles_df = articles_df[~mask_to_drop]

    return tourism_df, articles_df, tourism_all_df

#################################################
# 5. Dashboard
#################################################
PASTEL_COLORS = [
    '#b8e0d2',  # Mint
    '#eac4d5',  # Rose
    '#d4e4bc',  # Sage
    '#b8d4e3',  # Sky
    '#ffe5d9',  # Peach
    '#d8d3e8',  # Lavender
]

def create_dashboard(tourism_df, tourism_all_df):
    st.header("Tourism Analytics Dashboard")
    with st.container():
        st.markdown("""
        <div style='background-color: white; padding: 1.5rem; border-radius: 15px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);'>
        """, unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)
        with col1:
            latest_value = tourism_df["value"].iloc[0]
            st.metric("Latest Tourism Arrivals", f"{latest_value:,.0f}")

        with col2:
            avg_value = tourism_df["value"].mean()
            st.metric("Average Annual Arrivals", f"{avg_value:,.0f}")

        with col3:
            last_value = tourism_df["value"].iloc[-1]
            growth = ((latest_value - last_value) / last_value) * 100
            st.metric("Overall Growth", f"{growth:.1f}%")
        st.markdown("</div>", unsafe_allow_html=True)

    with st.container():
        st.markdown("""
        <div style='background-color: white; padding: 1.5rem; border-radius: 15px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); margin-top: 2rem;'>
        """, unsafe_allow_html=True)

        fig_global = px.line(
            tourism_df, x="date", y="value",
            title="International Tourism Arrivals Over Time",
            color_discrete_sequence=PASTEL_COLORS
        )
        fig_global.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font_family="Helvetica Neue",
        )
        st.plotly_chart(fig_global, use_container_width=True)

        # --- South America Analysis ---
        st.subheader("Detailed country Analysis")

        south_america_countries = [
            "Argentina", "Bolivia", "Brazil", "Chile", "Colombia", "Ecuador",
            "Guyana", "Paraguay", "Peru", "Suriname", "Uruguay", "Venezuela"
        ]

        south_america_data = tourism_all_df[
            tourism_all_df['country_value'].isin(south_america_countries)
        ]

        selected_country = st.selectbox(
            "Select a South American country:",
            south_america_countries,
            index=0
        )

        country_data = south_america_data[
            south_america_data['country_value'] == selected_country
        ].dropna(subset=['value'])

        country_data['date'] = pd.to_datetime(country_data['date'], format='%Y')

        fig_sa = px.line(
            country_data, x='date', y='value',
            title=f"Tourism Arrivals in {selected_country} Over Time",
            labels={'value': 'Number of Arrivals', 'date': 'Year'},
            markers=True
        )
        fig_sa.update_layout(
            xaxis_title="Year",
            yaxis_title="Number of Arrivals",
            template="plotly_white"
        )
        st.plotly_chart(fig_sa, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

#################################
# 6. Text Analysis
#################################
def create_text_analysis(articles_df):
    st.header("Text Analysis")

    cleaner = TextCleaner()

    st.subheader("A) Text Cleaning Demo")
    selected_article = st.selectbox(
        "Select an article to clean",
        options=articles_df["titre"]
    )

    if selected_article:
        original_text = articles_df[
            articles_df["titre"] == selected_article
        ]["contenu"].iloc[0]

        cleaned_text = cleaner.clean_text(original_text)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Original Text:**")
            st.text_area("", original_text[:1000] + "...", height=200)
        with col2:
            st.markdown("**Cleaned Text:**")
            st.text_area("", cleaned_text[:1000] + "...", height=200)

    st.subheader("B) Sentiment Analysis Demo")
    sentiment_analyzer = load_sentiment_analyzer()

    selected_article_sentiment = st.selectbox(
        "Select an article for sentiment analysis",
        options=articles_df["titre"],
        key="sentiment_box"
    )

    if selected_article_sentiment:
        if st.button("Analyze Sentiment"):
            article_text = articles_df[
                articles_df["titre"] == selected_article_sentiment
            ]["contenu"].iloc[0]

            cleaned_text = cleaner.clean_text(article_text)

            sentiment_result = sentiment_analyzer(cleaned_text[:512])
            st.markdown("**Sentiment Result:**")
            st.write(sentiment_result)
            """
            Typical output:
            [
              {
                'label': 'Positive' / 'Neutral' / 'Negative',
                'score': <float confidence>
              }
            ]
            """

    st.subheader("C) Full Automated Analysis")
    if st.button("Run Full Analysis"):
        st.info("Running a complete analysis on all articles. Please wait...")

        df_copy = articles_df.copy()
        df_copy["cleaned_content"] = df_copy["contenu"].apply(cleaner.clean_text)

        df_copy = df_copy[df_copy["cleaned_content"].str.len() > 10]

        results_label = []
        results_score = []

        for text in df_copy["cleaned_content"]:
            analysis = sentiment_analyzer(text[:512])
            label = analysis[0]["label"]
            score = analysis[0]["score"]
            results_label.append(label)
            results_score.append(score)

        df_copy["sentiment_label"] = results_label
        df_copy["sentiment_score"] = results_score

        summarizer = pipeline(
            "summarization",
            model="t5-large",
            tokenizer="t5-large"
        )
        summaries = []

        for text in df_copy["cleaned_content"]:
            text_trunc = text[:1000]
            if len(text_trunc) > 100:
                try:
                    summary = summarizer(
                        text_trunc,
                        max_length=130,
                        min_length=30,
                        do_sample=False
                    )
                    summaries.append(summary[0]["summary_text"])
                except Exception as e:
                    summaries.append(f"Error: {str(e)}")
            else:
                summaries.append(text_trunc)

        df_copy["summary"] = summaries

        st.success("Analysis complete!")
        st.dataframe(
            df_copy[[
                "titre",
                "cleaned_content",
                "sentiment_label",
                "sentiment_score",
                "summary"
            ]].head(10)
        )

#################################
# 7. Chatbot
#################################
def create_chatbot_interface(articles_df):
    st.header("Tourism Analysis Chatbot")

    if "chatbot" not in st.session_state:
        st.session_state.chatbot = TourismChatbot()

    selected_article = st.selectbox(
        "Select an article to summarize",
        options=articles_df["titre"],
        key="summary_box"
    )

    if st.button("Generate Summary"):
        with st.spinner("Generating summary..."):
            article_text = articles_df[
                articles_df["titre"] == selected_article
            ]["contenu"].iloc[0]

            summary = st.session_state.chatbot.summarize_text(article_text)

            st.markdown("### Summary")
            st.write(summary)

            # Display lengths before and after summarization
            st.write(f"**Length before summarization:** {len(article_text)} characters")
            st.write(f"**Length after summarization:** {len(summary)} characters")

#################################
# 8. Main Application
#################################
def main():
    st.title("🌎 Tourism Analysis Platform")

    selected = option_menu(
        menu_title=None,
        options=["Dashboard", "Text Analysis", "Chatbot"],
        icons=["graph-up", "file-text", "chat-dots"],
        menu_icon="cast",
        default_index=0,
        orientation="horizontal",
        styles={
            "container": {"padding": "0!important", "background-color": "#fafafa"},
            "icon": {"color": "#444444", "font-size": "25px"},
            "nav-link": {
                "font-size": "16px",
                "text-align": "left",
                "margin": "0px",
                "--hover-color": "#eee",
            },
            "nav-link-selected": {"background-color": "#b8e0d2"},
        }
    )

    try:
        tourism_df, articles_df, tourism_all_df = load_data()
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return

    tab1, tab2, tab3 = st.tabs([
        "Dashboard",
        "Text Analysis",
        "Chatbot",
    ])

    with tab1:
        create_dashboard(tourism_df, tourism_all_df)

    with tab2:
        create_text_analysis(articles_df)

    with tab3:
        create_chatbot_interface(articles_df)

if __name__ == "__main__":
    main()