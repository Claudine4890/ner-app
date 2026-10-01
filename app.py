import streamlit as st
import spacy

nlp = spacy.load("en_core_web_sm")

st.title("Named Entity Recognition")
text = st.text_area("Paste some text:", "Kigali is the capital of Rwanda.")

if st.button("Find entities"):
    doc = nlp(text)
    for ent in doc.ents:
        st.write(ent.text, "->", ent.label_)