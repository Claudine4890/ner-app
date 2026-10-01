import streamlit as st
import spacy
from spacy import displacy
import pandas as pd

st.set_page_config(page_title="NER App", layout="wide")

# Add more names here, one per line. Multi-word names are allowed.
KNOWN_NAMES = [
    "Claudine",
    "Immaculee",
    "Emmanuel",
    "Eric",
    "Aline",
    "Jean Claude",
    "Uwase",
    "Mukamana",
    "Habimana",
]


@st.cache_resource
def load_model():
    nlp = spacy.load("en_core_web_sm")
    ruler = nlp.add_pipe("entity_ruler", before="ner")
    patterns = []
    for name in KNOWN_NAMES:
        words = [{"LOWER": w.lower()} for w in name.split()]
        # the optional last token lets "Claudine Uwase" match as one name
        patterns.append(
            {"label": "PERSON", "pattern": words + [{"IS_TITLE": True, "OP": "?"}]}
        )
    ruler.add_patterns(patterns)
    return nlp


nlp = load_model()

LABELS = {
    "PERSON": "A person's name",
    "NORP": "Nationality, religious or political group",
    "FAC": "Building, airport, road, bridge",
    "ORG": "Company, agency, university",
    "GPE": "Country, city or state (geopolitical entity)",
    "LOC": "Other location (mountain, river, region)",
    "PRODUCT": "Product",
    "EVENT": "Named event (Olympics, war)",
    "WORK_OF_ART": "Title of a book, song or film",
    "LAW": "Law or legal document",
    "LANGUAGE": "Language",
    "DATE": "Date or time period",
    "TIME": "Time of day",
    "PERCENT": "Percentage",
    "MONEY": "Money amount",
    "QUANTITY": "Measurement (weight, distance)",
    "ORDINAL": "first, second, third...",
    "CARDINAL": "Other numbers",
}

st.title("Named Entity Recognition")
st.write("Paste some text to see the people, places, organizations and dates that spaCy finds.")
st.caption(
    "Model: spaCy en_core_web_sm plus a list of known names (always labeled PERSON). "
    "Names that are not on the list can still be missed."
)

with st.sidebar:
    st.header("What the labels mean")
    for label, meaning in LABELS.items():
        st.write(f"**{label}**: {meaning}")

text = st.text_area(
    "Paste some text:",
    "Claudine Uwase studies at the University of Rwanda in Kigali. Paul Kagame spoke on Monday.",
    height=150,
)

if st.button("Find entities"):
    doc = nlp(text)

    if not doc.ents:
        st.warning("No entities found in this text.")
    else:
        st.subheader("Highlighted text")
        html = displacy.render(doc, style="ent")
        st.markdown(html.replace("\n", " "), unsafe_allow_html=True)

        st.subheader("Entities found")
        rows = [
            {
                "Entity": ent.text,
                "Label": ent.label_,
                "Meaning": LABELS.get(ent.label_, ""),
                "Sentence": ent.sent.text,
            }
            for ent in doc.ents
        ]
        st.dataframe(pd.DataFrame(rows))

        st.subheader("Count by label")
        counts = pd.Series([ent.label_ for ent in doc.ents]).value_counts()
        st.bar_chart(counts)
