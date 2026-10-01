import streamlit as st
import spacy
from spacy import displacy
import pandas as pd

st.set_page_config(page_title="NER App", layout="wide")


@st.cache_resource
def load_model():
    return spacy.load("en_core_web_sm")


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
st.caption("Model: spaCy en_core_web_sm (small English model). It can miss names and mislabel some entities.")

with st.sidebar:
    st.header("What the labels mean")
    for label, meaning in LABELS.items():
        st.write(f"**{label}**: {meaning}")

text = st.text_area(
    "Paste some text:",
    "Claudine studies at the University of Rwanda in Kigali. Paul Kagame spoke on Monday.",
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
