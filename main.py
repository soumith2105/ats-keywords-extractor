import streamlit as st
from streamlit_local_storage import LocalStorage
import re
from collections import Counter, defaultdict
from nltk.corpus import stopwords
from nltk import word_tokenize, ngrams, download
import openai

# Download stopwords if not already present
download("punkt")
download("stopwords")

st.set_page_config(
    page_title="ATS Keywords Extractor",
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
        label_visibility="collapsed",
    )
    st.markdown("</div></div>", unsafe_allow_html=True)
    if api_key_input != api_key_val:
        ls.setItem("openai_api_key", api_key_input)
openai_key = api_key_input

st.markdown('<div class="split-col">', unsafe_allow_html=True)
cols = st.columns([1, 0.05, 1], gap="large")

with cols[0]:
    st.markdown("### Paste your job description below:")
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
        # Ultra-conservative: grammatical/filler only, no ATS verbs/skills/tools
        "a",
        "about",
        "above",
        "after",
        "again",
        "against",
        "all",
        "am",
        "an",
        "and",
        "any",
        "are",
        "aren't",
        "as",
        "at",
        "be",
        "because",
        "been",
        "before",
        "being",
        "below",
        "between",
        "both",
        "but",
        "by",
        "can",
        "cannot",
        "could",
        "couldn't",
        "did",
        "didn't",
        "do",
        "does",
        "doesn't",
        "doing",
        "don't",
        "down",
        "during",
        "each",
        "few",
        "for",
        "from",
        "further",
        "had",
        "hadn't",
        "has",
        "hasn't",
        "have",
        "haven't",
        "having",
        "he",
        "he'd",
        "he'll",
        "he's",
        "her",
        "here",
        "here's",
        "hers",
        "herself",
        "him",
        "himself",
        "his",
        "how",
        "how's",
        "i",
        "i'd",
        "i'll",
        "i'm",
        "i've",
        "if",
        "in",
        "into",
        "is",
        "isn't",
        "it",
        "it's",
        "its",
        "itself",
        "let's",
        "me",
        "more",
        "most",
        "mustn't",
        "my",
        "myself",
        "no",
        "nor",
        "not",
        "of",
        "off",
        "on",
        "once",
        "only",
        "or",
        "other",
        "ought",
        "our",
        "ours",
        "ourselves",
        "out",
        "over",
        "own",
        "same",
        "shan't",
        "she",
        "she'd",
        "she'll",
        "she's",
        "should",
        "shouldn't",
        "so",
        "some",
        "such",
        "than",
        "that",
        "that's",
        "the",
        "their",
        "theirs",
        "them",
        "themselves",
        "then",
        "there",
        "there's",
        "these",
        "they",
        "they'd",
        "they'll",
        "they're",
        "they've",
        "this",
        "those",
        "through",
        "to",
        "too",
        "under",
        "until",
        "up",
        "very",
        "was",
        "wasn't",
        "we",
        "we'd",
        "we'll",
        "we're",
        "we've",
        "were",
        "weren't",
        "what",
        "what's",
        "when",
        "when's",
        "where",
        "where's",
        "which",
        "while",
        "who",
        "who's",
        "whom",
        "why",
        "why's",
        "with",
        "won't",
        "would",
        "wouldn't",
        "you",
        "you'd",
        "you'll",
        "you're",
        "you've",
        "your",
        "yours",
        "yourself",
        "yourselves",
        "etc",
        "etc.",
        "per",
        "such",
        "within",
        "among",
        "across",
        "amongst",
        "various",
        "well",
        "new",
        "plus",
        "year",
        "years",
        "based",
        "related",
        "highly",
        "strongly",
        "successful",
        "able",
        "including",
        "minimum",
        "maximum",
        "must",
        "may",
        "will",
        "should",
        "required",
        "preferred",
        "desired",
        "someone",
        "everyone",
        "thing",
        "things",
        "nothing",
        "person",
        "people",
        "individual",
        "individuals",
        "self",
        "best",
        "better",
        "same",
        "fit",
        "core",
        "appropriate",
        "potential",
    }
    return set(stopwords.words("english")).union(custom_stopwords)


def filter_subsumed_ngrams(all_ngram_counts):
    result = defaultdict(list)
    longer_ngrams = set()
    longer_ngrams_count = dict()
    for n in sorted(all_ngram_counts.keys(), reverse=True):
        for phrase, count in all_ngram_counts[n]:
            longer_ngrams.add(phrase)
            longer_ngrams_count[phrase] = count

    for n in sorted(all_ngram_counts.keys()):
        for phrase, count in all_ngram_counts[n]:
            subsumed = False
            for longer_phrase in longer_ngrams:
                if len(longer_phrase.split()) <= len(phrase.split()):
                    continue
                if (
                    re.search(r"\b{}\b".format(re.escape(phrase)), longer_phrase)
                    and longer_ngrams_count[longer_phrase] >= count
                ):
                    subsumed = True
                    break
            if not subsumed:
                result[n].append((phrase, count))
    return result


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


def extract_and_count_ats_terms(jd, api_key):
    openai.api_key = api_key
    prompt = f"""Given the following job description, extract and list all possible ATS-relevant keywords, skills, technologies, and responsibilities. Categorize the terms into:  
1. Hard Skills & Tech Stack  
2. Tasks / Responsibilities  
3. Soft Skills / Traits  
4. Bonus / Optional

Output the list in each category as a comma-separated string, and do not use list items or tables.

Job Description:  
{jd}

Example Output Format:

1. Hard Skills & Tech Stack  
term1, term2, term3

2. Tasks / Responsibilities  
term1, term2

3. Soft Skills / Traits  
term1, term2

4. Bonus / Optional  
term1, term2
"""
    try:
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": "You are an assistant that extracts and categorizes ATS keywords and skills from job descriptions.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0,
        )
        result = response.choices[0].message.content
        cats = {
            "Hard Skills & Tech Stack": [],
            "Tasks / Responsibilities": [],
            "Soft Skills / Traits": [],
            "Bonus / Optional": [],
        }
        for cat in cats:
            match = re.search(rf"{re.escape(cat)}\s*\n([^\n]*)", result, re.IGNORECASE)
            if match:
                terms = [t.strip() for t in match.group(1).split(",") if t.strip()]
                cats[cat] = terms

        freq_dict = {}
        for cat, terms in cats.items():
            freq_dict[cat] = []
            for term in terms:
                pattern = r"\b{}\b".format(re.escape(term))
                count = len(re.findall(pattern, jd, flags=re.IGNORECASE))
                freq_dict[cat].append((term, count))
        return freq_dict
    except Exception as e:
        return {"Error": [f"Error: {e}"]}


with cols[2]:
    st.markdown("### Results")
    if not job_desc.strip():
        st.markdown(
            """
            <div style="color: #888; font-size: 1.1rem;">
                <div style="font-size:1.3rem;font-weight:700;">ATS Keywords & Skills (GPT-4o, Categorized):</div>
                <span style="color: #bbb;">(ATS keywords, skills, responsibilities will appear here, categorized by AI)</span>
                <div style="font-size:1.3rem;font-weight:700;">Repeated Words & Phrases (with frequencies):</div>
                <span style="color: #bbb;">(List will appear here)</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        # ATS Keywords & Skills (GPT-4o, Categorized)
        if openai_key and openai_key.startswith("sk-"):
            st.markdown(
                '<div style="font-size:1.5rem;font-weight:700;margin-bottom:0.5em;">ATS Keywords & Skills (GPT-4o, Categorized):</div>',
                unsafe_allow_html=True,
            )
            with st.spinner("Extracting ATS keywords, skills, and responsibilities..."):
                ats_terms = extract_and_count_ats_terms(job_desc, openai_key)
            for cat in [
                "Hard Skills & Tech Stack",
                "Tasks / Responsibilities",
                "Soft Skills / Traits",
                "Bonus / Optional",
            ]:
                if cat in ats_terms and ats_terms[cat]:
                    st.markdown(
                        f"##### {cat}",
                    )
                    line = ", ".join(
                        f"{term} ({count})"
                        for term, count in ats_terms[cat]
                        if term and count > 0
                    )
                    if line:
                        st.write(line)
                    else:
                        st.write("(None found)")
        else:
            st.info(
                "Enter your OpenAI API key above to extract technologies and categorized ATS keywords (GPT-4o powered)."
            )

        # Repeated Words & Phrases (with frequencies)
        text_clean = clean_text(job_desc)
        stop_words = get_stopwords()
        tokens = [
            word
            for word in word_tokenize(text_clean)
            if word not in stop_words and word.isalpha()
        ]
        word_counts = Counter(tokens)
        repeated = [(word, count) for word, count in word_counts.items() if count > 2]

        # Collect ngrams (2+), filter for repeats, subsume
        all_ngram_counts = {}
        max_ngram = 6
        for n in range(2, max_ngram + 1):
            ngram_counts = Counter([" ".join(gram) for gram in ngrams(tokens, n)])
            frequent_ngrams = [
                (gram, count) for gram, count in ngram_counts.items() if count > 1
            ]
            if frequent_ngrams:
                all_ngram_counts[n] = frequent_ngrams
        unique_ngram_counts = filter_subsumed_ngrams(all_ngram_counts)
        ngram_list = []
        for n in sorted(unique_ngram_counts.keys()):
            ngram_list += unique_ngram_counts[n]

        # Remove single words subsumed by any phrase with >= same count
        words_to_remove = set()
        for word, count in repeated:
            for phrase, pcount in ngram_list:
                if word in phrase.split() and pcount >= count:
                    words_to_remove.add(word)
                    break
        filtered_repeated = [
            (word, count) for word, count in repeated if word not in words_to_remove
        ]

        # Combine and sort
        combined = filtered_repeated + ngram_list
        combined_sorted = sorted(combined, key=lambda x: -x[1])

        st.markdown(
            '<div style="font-size:1.5rem;font-weight:700;margin-top:1.2em;margin-bottom:0.5em;">Repeated Words & Phrases (with frequencies):</div>',
            unsafe_allow_html=True,
        )
        if combined_sorted:
            st.write(", ".join([f"{w} ({c})" for w, c in combined_sorted]))
        else:
            st.write("(None)")

st.markdown("</div>", unsafe_allow_html=True)
