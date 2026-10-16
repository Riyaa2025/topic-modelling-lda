"""
Topic Modelling using LDA - Streamlit app
Run:  streamlit run app.py
Needs the 'lda_artifacts' folder next to this file.
"""
import re
import string
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from gensim.corpora import Dictionary, MmCorpus
from gensim.models import LdaModel

# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------
st.set_page_config(page_title="Topic Modelling using LDA", page_icon="📚", layout="wide")

ART = Path(__file__).parent / "lda_artifacts"
ACCENT = "#0f9d8a"
COHERENCE = 0.5038  # c_v coherence of the final model (from the notebook)

# Human-readable names for the 12 topics (read from the top keywords). Edit freely.
LABELS = {
    0: "Deep learning architectures (CNN / LSTM)",
    1: "Courses & learning resources",
    2: "Object detection (R-CNN)",
    3: "Interviews & podcasts",
    4: "Image recognition & TensorFlow",
    5: "GPUs & hardware for deep learning",
    6: "Policy-based reinforcement learning",
    7: "AI, technology & industry",
    8: "Games & AI research",
    9: "Backpropagation (+ non-English text)",
    10: "Neural network fundamentals",
    11: "Q-learning & RL agents",
}

SAMPLES = {
    "Reinforcement learning": (
        "The agent interacts with the environment and receives a reward after every action. "
        "Using a Q-table, it updates the value of each state and action pair so that the policy "
        "slowly improves over many episodes of training."
    ),
    "GPU for deep learning": (
        "Choosing a GPU for deep learning: the GTX 1080 Ti offers great performance per dollar, "
        "while a CPU alone is far too slow for training large models. Memory and power usage matter too."
    ),
    "Neural network basics": (
        "Each neuron multiplies its inputs by a weight, adds a bias and passes the result through an "
        "activation function. During backpropagation the gradient of the loss is used to update every "
        "weight in each layer."
    ),
}

# ----------------------------------------------------------------------------
# Styling
# ----------------------------------------------------------------------------
st.markdown(
    f"""
<style>
#MainMenu, footer {{visibility: hidden;}}
.block-container {{padding-top: 2.2rem; max-width: 1150px;}}
.hero h1 {{font-size: 2.2rem; margin: 0 0 .15rem 0; letter-spacing: -.5px;}}
.hero p {{opacity: .7; margin: 0 0 1.4rem 0; font-size: 1.02rem;}}
.card {{border: 1px solid rgba(128,128,128,.25); border-radius: 12px;
        padding: 1rem 1.2rem; background: rgba(128,128,128,.05);}}
.metric .v {{font-size: 1.75rem; font-weight: 700; color: {ACCENT}; line-height: 1.2;}}
.metric .l {{font-size: .75rem; text-transform: uppercase; letter-spacing: .08em; opacity: .65;}}
.result .l {{font-size: .75rem; text-transform: uppercase; letter-spacing: .08em; opacity: .65;}}
.result .t {{font-size: 1.45rem; font-weight: 700; line-height: 1.25; margin: .2rem 0 .1rem 0;}}
.result .p {{color: {ACCENT}; font-weight: 600; margin-bottom: .8rem;}}
.chip {{display: inline-block; padding: .2rem .7rem; margin: .15rem .3rem .15rem 0;
        border-radius: 999px; font-size: .85rem;
        background: rgba(15,157,138,.14); border: 1px solid rgba(15,157,138,.45);}}
.chip.dim {{background: transparent; border-color: rgba(128,128,128,.3); opacity: .75;}}
div[data-testid="stTabs"] button {{font-size: 1rem;}}
</style>
""",
    unsafe_allow_html=True,
)


# ----------------------------------------------------------------------------
# Loaders (cached)
# ----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading model...")
def load_artifacts():
    model = LdaModel.load(str(ART / "lda_model.gensim"))
    dictionary = Dictionary.load(str(ART / "id2word.dict"))
    corpus = MmCorpus(str(ART / "corpus.mm"))
    return model, dictionary, corpus


@st.cache_resource(show_spinner="Loading language tools...")
def load_nlp():
    import spacy

    try:
        return spacy.load("en_core_web_sm", disable=["parser", "ner"])
    except OSError:
        return None


@st.cache_resource
def load_stopwords():
    import nltk

    try:
        from nltk.corpus import stopwords

        return set(stopwords.words("english"))
    except LookupError:
        nltk.download("stopwords", quiet=True)
        from nltk.corpus import stopwords

        return set(stopwords.words("english"))


@st.cache_data(show_spinner=False)
def dominant_topic_counts():
    model, _, corpus = load_artifacts()
    counts = np.zeros(model.num_topics, dtype=int)
    for bow in corpus:
        dist = model.get_document_topics(bow, minimum_probability=0.0)
        counts[max(dist, key=lambda x: x[1])[0]] += 1
    return counts


# ----------------------------------------------------------------------------
# Preprocessing (same steps as the notebook)
# ----------------------------------------------------------------------------
_SPECIAL = {"can't": "cannot", "won't": "will not", "shan't": "shall not",
            "ain't": "are not", "let's": "let us", "y'all": "you all"}
_SUFFIX = (("'re", " are"), ("'ve", " have"), ("'ll", " will"),
           ("'d", " would"), ("'m", " am"), ("'s", " is"))
_PUNCT = re.compile("[%s]" % re.escape(string.punctuation + "’—”“"))


def expand_contractions(text: str) -> str:
    text = text.replace("’", "'")
    for k, v in _SPECIAL.items():
        text = text.replace(k, v)
    text = re.sub(r"n't\b", " not", text)
    for suf, full in _SUFFIX:
        text = text.replace(suf, full)
    return text


def preprocess(text: str, nlp, stop_words: set) -> list:
    text = expand_contractions(text.lower())
    text = " ".join(w for w in text.split() if w not in stop_words)
    text = _PUNCT.sub("", text)
    text = re.sub(r"\w*\d\w*", "", text)
    lemmas = [t.lemma_ for t in nlp(text)]
    return [w for w in lemmas if w.isalpha() and len(w) > 1 and w not in stop_words]


def analyze(text: str, model, dictionary, nlp, stop_words):
    tokens = preprocess(text, nlp, stop_words)
    known = [t for t in tokens if t in dictionary.token2id]
    dist = model.get_document_topics(dictionary.doc2bow(tokens), minimum_probability=0.0)
    dist = sorted(((int(i), float(p)) for i, p in dist), key=lambda x: -x[1])
    return tokens, known, dist


# ----------------------------------------------------------------------------
# Small UI helpers
# ----------------------------------------------------------------------------
def metric(col, value, label):
    col.markdown(
        f'<div class="card metric"><div class="v">{value}</div><div class="l">{label}</div></div>',
        unsafe_allow_html=True,
    )


def chips(words, highlight=None):
    highlight = set(highlight or [])
    return "".join(
        f'<span class="chip{"" if (not highlight or w in highlight) else " dim"}">{w}</span>'
        for w in words
    )


def topic_words(model, topic_id, n=10):
    return [(w, float(p)) for w, p in model.show_topic(topic_id, topn=n)]


def set_sample(name):
    st.session_state["user_text"] = SAMPLES[name]


# ----------------------------------------------------------------------------
# Load everything
# ----------------------------------------------------------------------------
if not ART.exists():
    st.error(f"Could not find the folder `{ART.name}` next to app.py.")
    st.stop()

model, dictionary, corpus = load_artifacts()
nlp = load_nlp()
stop_words = load_stopwords()
K = model.num_topics

# ----------------------------------------------------------------------------
# Header + key numbers
# ----------------------------------------------------------------------------
st.markdown(
    f"""
<div class="hero">
  <h1>Topic Modelling using LDA</h1>
  <p>Paste any AI / ML / Data Science text and see which of the {K} themes learned from
  Medium articles it belongs to.</p>
</div>
""",
    unsafe_allow_html=True,
)

m1, m2, m3, m4 = st.columns(4)
metric(m1, corpus.num_docs, "Articles")
metric(m2, K, "Topics")
metric(m3, f"{len(dictionary):,}", "Vocabulary")
metric(m4, f"{COHERENCE:.3f}", "Coherence (c_v)")
st.write("")

tab_try, tab_topics, tab_map, tab_about = st.tabs(
    ["Analyze text", "Explore topics", "Topic map", "About"]
)

# ----------------------------------------------------------------------------
# Tab 1 - Analyze text
# ----------------------------------------------------------------------------
with tab_try:
    if nlp is None:
        st.error("spaCy model missing. Run:  `python -m spacy download en_core_web_sm`")
        st.stop()

    st.caption("Try a sample")
    cols = st.columns(len(SAMPLES) + 2)
    for col, name in zip(cols, SAMPLES):
        col.button(name, on_click=set_sample, args=(name,), key=f"s_{name}")

    text = st.text_area(
        "Your text",
        key="user_text",
        height=190,
        placeholder="Paste a paragraph or a full article here...",
        label_visibility="collapsed",
    )
    run = st.button("Detect topics", type="primary")

    if run:
        if len(text.split()) < 5:
            st.warning("Please enter at least a sentence or two.")
        else:
            with st.spinner("Analyzing..."):
                tokens, known, dist = analyze(text, model, dictionary, nlp, stop_words)

            if not known:
                st.warning(
                    "None of the words in this text are in the model's vocabulary. "
                    "Try text about AI, machine learning or data science."
                )
            else:
                top_id, top_p = dist[0]
                left, right = st.columns([1, 1.5], gap="large")

                with left:
                    st.markdown(
                        f"""<div class="card result">
                        <div class="l">Most likely topic</div>
                        <div class="t">{LABELS.get(top_id, f"Topic {top_id}")}</div>
                        <div class="p">{top_p:.0%} match &nbsp;·&nbsp; Topic {top_id}</div>
                        <div class="l" style="margin-bottom:.3rem">Topic keywords</div>
                        {chips([w for w, _ in topic_words(model, top_id, 10)], known)}
                        </div>""",
                        unsafe_allow_html=True,
                    )
                    st.caption(
                        f"Highlighted keywords appear in your text. "
                        f"{len(set(known))} unique words matched the vocabulary "
                        f"({len(known)} of {len(tokens)} cleaned words)."
                    )
                    if top_p < 0.35:
                        st.info(
                            "Low confidence: the text is short or spans several themes, "
                            "so the probability is spread out."
                        )

                with right:
                    df = pd.DataFrame(
                        {
                            "Topic": [f"T{i} · {LABELS.get(i, i)}" for i, _ in dist],
                            "Probability": [p for _, p in dist],
                            "Top": [i == top_id for i, _ in dist],
                        }
                    )
                    chart = (
                        alt.Chart(df)
                        .mark_bar(cornerRadiusEnd=4)
                        .encode(
                            x=alt.X("Probability:Q", axis=alt.Axis(format="%"), title=None),
                            y=alt.Y("Topic:N", sort="-x", title=None,
                                    axis=alt.Axis(labelLimit=280)),
                            color=alt.condition(
                                alt.datum.Top, alt.value(ACCENT), alt.value("#9aa5a3")
                            ),
                            tooltip=["Topic", alt.Tooltip("Probability:Q", format=".1%")],
                        )
                        .properties(height=28 * len(df))
                    )
                    st.altair_chart(chart)

# ----------------------------------------------------------------------------
# Tab 2 - Explore topics
# ----------------------------------------------------------------------------
with tab_topics:
    c_sel, c_n = st.columns([2, 1])
    topic_id = c_sel.selectbox(
        "Topic", list(range(K)), format_func=lambda i: f"Topic {i} · {LABELS.get(i, i)}"
    )
    n_words = c_n.slider("Keywords to show", 5, 20, 10)

    counts = dominant_topic_counts()
    share = counts[topic_id] / counts.sum()

    left, right = st.columns(2, gap="large")
    with left:
        st.markdown(f"**Top keywords** · {counts[topic_id]} articles ({share:.0%}) lead with this topic")
        words = topic_words(model, topic_id, n_words)
        wdf = pd.DataFrame(words, columns=["Word", "Weight"])
        wchart = (
            alt.Chart(wdf)
            .mark_bar(cornerRadiusEnd=4, color=ACCENT)
            .encode(
                x=alt.X("Weight:Q", title=None),
                y=alt.Y("Word:N", sort="-x", title=None),
                tooltip=["Word", alt.Tooltip("Weight:Q", format=".4f")],
            )
            .properties(height=26 * len(wdf))
        )
        st.altair_chart(wchart)

    with right:
        st.markdown("**Articles per topic** (dominant topic)")
        cdf = pd.DataFrame(
            {
                "Topic": [f"T{i}" for i in range(K)],
                "Name": [LABELS.get(i, i) for i in range(K)],
                "Articles": counts,
                "Selected": [i == topic_id for i in range(K)],
            }
        )
        cchart = (
            alt.Chart(cdf)
            .mark_bar(cornerRadiusEnd=4)
            .encode(
                x=alt.X("Topic:N", sort=None, title=None, axis=alt.Axis(labelAngle=0)),
                y=alt.Y("Articles:Q", title=None),
                color=alt.condition(alt.datum.Selected, alt.value(ACCENT), alt.value("#9aa5a3")),
                tooltip=["Name", "Articles"],
            )
            .properties(height=300)
        )
        st.altair_chart(cchart)

    with st.expander("All topics at a glance"):
        for i in range(K):
            st.markdown(
                f"**Topic {i} · {LABELS.get(i, i)}**<br>"
                f"{chips([w for w, _ in topic_words(model, i, 8)])}",
                unsafe_allow_html=True,
            )

# ----------------------------------------------------------------------------
# Tab 3 - pyLDAvis map
# ----------------------------------------------------------------------------
with tab_map:
    st.caption(
        "Each circle is a topic; the further apart two circles are, the more different the topics. "
        "Hover a circle to see its most relevant terms."
    )
    html_path = ART / "lda_visualization.html"
    if html_path.exists():
        if hasattr(st, "iframe"):  # newer Streamlit
            st.iframe(html_path, height=820)
        else:  # older Streamlit
            import streamlit.components.v1 as components

            components.html(html_path.read_text(encoding="utf-8"), height=820, scrolling=True)
    else:
        st.warning("`lda_visualization.html` not found in lda_artifacts.")

# ----------------------------------------------------------------------------
# Tab 4 - About
# ----------------------------------------------------------------------------
with tab_about:
    st.markdown(
        f"""
**Dataset** — Medium articles on AI, Machine Learning and Data Science ({corpus.num_docs} used for training).

**Model** — Latent Dirichlet Allocation (gensim), {K} topics, tuned with a grid search over the
number of topics, alpha and eta using the c_v coherence score ({COHERENCE:.3f}).

**Text pipeline** (applied to your input exactly like the training data)
1. Lowercase and expand contractions
2. Remove stopwords, punctuation and words containing digits
3. Lemmatize with spaCy (`en_core_web_sm`)
4. Keep alphabetic words of 2+ letters, convert to bag-of-words with the saved dictionary
5. Ask the LDA model for the topic distribution

**Reading the result** — the model returns a probability for every topic and they sum to 100%.
Short text or text outside AI / ML gives a flat distribution, which is expected for LDA.
"""
    )