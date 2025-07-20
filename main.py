import streamlit as st
from streamlit_local_storage import LocalStorage
import re
from collections import Counter
from nltk.corpus import stopwords
from nltk import word_tokenize, ngrams, download
import openai

# Download stopwords if not already present
download("punkt_tab")
download("stopwords")

st.set_page_config(
    page_title="ATS Keywords Extractor",  # This is your new app/browser tab name
    page_icon="🔍",
    layout="wide",
)

# Responsive style
st.markdown(
    """
<style>
@media (max-width: 900px) {
    .block-container { padding: 0.5rem !important; }
    .stTextArea textarea { min-height: 300px !important; font-size: 1rem; }
}
@media (max-width: 650px) {
    .split-col { flex-direction: column !important; }
    .stTextArea textarea { min-height: 180px !important; }
}
.split-col { display: flex; flex-direction: row; gap: 2vw; width: 100%; }
.split-left, .split-right { flex: 1; min-width: 0; }
</style>
""",
    unsafe_allow_html=True,
)

# # Inject custom CSS for split view and spacing
# st.markdown(
#     """
#     <style>
#     .split-col > div {
#         height: 80vh !important;
#     }
#     textarea {
#         min-height: 76vh !important;
#         font-size: 1.09rem;
#     }
#     .block-container {
#         padding-top: 1.5rem;
#         padding-bottom: 1.5rem;
#         padding-left: 2rem;
#         padding-right: 2rem;
#     }
#     </style>
#     """,
#     unsafe_allow_html=True,
# )

# Centered layout: narrow center column for title + input
col1, col2, col3 = st.columns([1, 1, 1], gap="large")

with col2:
    st.markdown(
        '<div style="text-align:center;"><h2 style="margin-bottom:0.2em;margin-top:0.2em;">ATS Keywords Extractor</h2></div>',
        unsafe_allow_html=True,
    )
    ls = LocalStorage()
    api_key_val = ls.getItem("openai_api_key") or ""
    # OpenAI key input with restricted width and centered
    st.markdown(
        """
        <div style="display:flex;justify-content:center;">
            <div style="width:270px;max-width:100%;">
        """,
        unsafe_allow_html=True,
    )
    api_key_input = st.text_input(
        "OpenAI API Key",
        value=api_key_val,
        type="password",
        placeholder="Open AI Key: sk-...",
        help="Key is saved in your browser only.",
        label_visibility="collapsed",  # You can use "collapsed" to hide the label
    )
    st.markdown("</div></div>", unsafe_allow_html=True)
    # Sync input to storage
    if api_key_input != api_key_val:
        ls.setItem("openai_api_key", api_key_input)
openai_key = api_key_input

st.markdown('<div class="split-col">', unsafe_allow_html=True)
cols = st.columns([1, 0.05, 1], gap="large")  # Left, gap, Right

with cols[0]:
    st.markdown("#### Paste your job description below:")
    job_desc = st.text_area(
        "Job Description",
        height=400,
        placeholder="Paste your job description here...",
        key="jd_area",
        label_visibility="collapsed",
    )


def clean_text(text):
    return re.sub(r"[^a-zA-Z0-9\s]", "", text).lower()


def get_stopwords():
    custom_stopwords = {
        "and",
        "the",
        "but",
        "or",
        "yet",
        "so",
        "for",
        "nor",
        "with",
        "on",
        "in",
        "at",
        "by",
        "to",
        "of",
        "a",
        "an",
        "is",
        "are",
        "as",
        "be",
        "was",
        "were",
        "this",
        "that",
    }
    return set(stopwords.words("english")).union(custom_stopwords)


with cols[2]:
    st.markdown("#### Results")
    if not job_desc.strip():
        st.markdown(
            """
            <div style="color: #888; font-size: 1.1rem;">
                <div style="font-size:1.3rem;font-weight:700;">Repeated Words (more than 3 times):</div>
                <span style="color: #bbb;">(List will appear here)</span>
                <br><br>
                <div style="font-size:1.3rem;font-weight:700;">Repeated Phrases:</div>
                <span style="color: #bbb;">(List will appear here for bigrams, trigrams, etc.)</span>
                <br><br>
                <div style="font-size:1.3rem;font-weight:700;">Technologies Detected:</div>
                <span style="color: #bbb;">(Technologies from the job description will be extracted by GPT-4o and shown here)</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        text_clean = clean_text(job_desc)
        stop_words = get_stopwords()
        tokens = [
            word
            for word in word_tokenize(text_clean)
            if word not in stop_words and word.isalpha()
        ]
        # Repeated words > 3
        word_counts = Counter(tokens)
        keywords = [(word, count) for word, count in word_counts.items() if count > 3]
        st.markdown(
            '<div style="font-size:1.5rem;font-weight:700;margin-bottom:0.5em;">Repeated Words (more than 3 times):</div>',
            unsafe_allow_html=True,
        )
        if keywords:
            st.write(
                ", ".join(
                    [
                        f"{word} ({count})"
                        for word, count in sorted(keywords, key=lambda x: -x[1])
                    ]
                )
            )
        else:
            st.write("(None)")

        # N-grams
        n = 2
        found_ngram = False
        while True:
            ngram_counts = Counter([" ".join(gram) for gram in ngrams(tokens, n)])
            frequent_ngrams = [
                (gram, count) for gram, count in ngram_counts.items() if count > 1
            ]
            if not frequent_ngrams:
                break
            found_ngram = True
            st.markdown(
                f'<div style="font-size:1.5rem;font-weight:700;margin-top:1.2em;margin-bottom:0.5em;">{n}-word phrases repeated more than once:</div>',
                unsafe_allow_html=True,
            )
            st.write(
                ", ".join(
                    [
                        f"{gram} ({count})"
                        for gram, count in sorted(frequent_ngrams, key=lambda x: -x[1])
                    ]
                )
            )
            n += 1
        if not found_ngram:
            st.markdown(
                '<div style="font-size:1.5rem;font-weight:700;margin-top:1.2em;margin-bottom:0.5em;">Repeated Phrases:</div>',
                unsafe_allow_html=True,
            )
            st.write("(None)")

        # Technologies (OpenAI GPT-4o)
        if openai_key and openai_key.startswith("sk-"):
            st.markdown(
                '<div style="font-size:1.5rem;font-weight:700;margin-top:1.2em;margin-bottom:0.5em;">Technologies Detected (AI-Powered):</div>',
                unsafe_allow_html=True,
            )

            def extract_technologies_openai(jd, api_key):
                openai.api_key = api_key
                prompt = (
                    "Extract a list of technologies (programming languages, frameworks, cloud services, tools, platforms, etc.) "
                    "from the following job description. Only return the technologies, one per line, no explanations or extra text:\n\n"
                    f"{jd}\n"
                )
                try:
                    response = openai.chat.completions.create(
                        model="gpt-4o",
                        messages=[
                            {
                                "role": "system",
                                "content": "You are an assistant that extracts a list of technologies from text.",
                            },
                            {"role": "user", "content": prompt},
                        ],
                        temperature=0,
                    )
                    tech_list = response.choices[0].message.content.strip().split("\n")
                    tech_list = [
                        re.sub(r"^\s*[\-\*\d\.\)]*\s*", "", t).strip()
                        for t in tech_list
                        if t.strip()
                    ]
                    return list(set([t for t in tech_list if len(t) > 1]))
                except Exception as e:
                    return [f"Error: {e}"]

            with st.spinner("Extracting technologies using GPT-4o..."):
                techs = extract_technologies_openai(job_desc, openai_key)
            if techs:
                tech_freqs = []
                for tech in techs:
                    pattern = r"\b{}\b".format(re.escape(tech))
                    count = len(re.findall(pattern, job_desc, flags=re.IGNORECASE))
                    tech_freqs.append((tech, count))
                tech_freqs_sorted = sorted(
                    tech_freqs, key=lambda x: (-x[1], x[0].lower())
                )
                st.write(
                    ", ".join(
                        [f"{tech} ({count})" for tech, count in tech_freqs_sorted]
                    )
                )
            else:
                st.write("(None detected)")
        else:
            st.info(
                "Enter your OpenAI API key above to extract technologies (GPT-4o powered)."
            )

st.markdown("</div>", unsafe_allow_html=True)
