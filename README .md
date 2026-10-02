# Named Entity Recognition with Word Embeddings

Assignment: *Improving Your NLP Project with Word Embeddings*.
Live demo (Streamlit): https://claudine-ner-app-1.streamlit.app

## 1. The problem

Find **people (PER), organizations (ORG), locations (LOC) and other names (MISC)** in English text.

My first NLP project was SMS spam detection. The assignment says spam detection is not continued, so I changed to **Named Entity Recognition (NER)** and used word embeddings as input features.

The main question: **does representing words as GloVe embeddings help, compared with word-ID (bag-of-words style) features?**

## 2. What I did, in order

1. **Explored word embeddings** with GloVe (nearest neighbours, similarity scores, an analogy, and words from my own field).
2. **Tried NER with a ready-made model** (spaCy) on 1,000 news articles, and found where it makes mistakes.
3. **Built a Streamlit demo app** and deployed it online.
4. **Built the main experiment on a labeled dataset (CoNLL-2003):** a baseline without embeddings, then the same model with GloVe embeddings, compared with precision, recall and F1 per entity type.

## 3. Models and tools, and why I used them

| Choice | Why I used it | Limits / alternatives |
|---|---|---|
| **GloVe, 50 dimensions** (`glove-wiki-gigaword-50`) | Pre-trained on Wikipedia and news text, which fits the news text in CoNLL-2003. Small (66 MB), quick to download and easy to load with gensim. I did not have a large text collection or enough time to train my own embeddings. | 50 dimensions is small. Larger GloVe (100/300d) or my own Word2Vec might do better. A static vector gives a word one meaning only (see "runoff" below). |
| **Not Word2Vec or FastText** | GloVe was available pre-trained in one line, and I already knew it from my exploration. | FastText uses sub-word pieces, so it could handle rare or misspelled names better. I did not test it. |
| **CoNLL-2003 dataset** (`lhoestq/conll2003`) | A standard labeled NER dataset (train, validation, test). I needed correct labels to calculate precision, recall and F1. My spaCy experiment had no correct answers to compare with. | Western news text from 1996, so Rwandan names are rare. It has no DATE label. |
| **Logistic regression classifier** | Simple, fast, and the same for both versions, so any difference in results comes from the word representation and not from a different algorithm. It trains in under a minute on free Colab. | It labels each word on its own and only sees a 3-word window. A sequence model (CRF, LSTM) would be stronger. |
| **Window of 3 words** (previous, current, next) | Gives the classifier some context cheaply. | Misses longer-range context. |
| **4 capitalization/digit features in both models** | Capital letters are an important clue for names. Keeping them in both models makes the comparison fair. | |
| **My own evaluation function** | `seqeval` would not install on Colab. I wrote the same measure: an entity counts as correct only if its type and exact span match. | My total support (5,648 entities) matches the known size of the CoNLL-2003 test set, which suggests it works. |
| **spaCy `en_core_web_sm`** | A free, ready-made NER model, small enough to run on free Streamlit hosting. Good for a quick first look. | Small model trained on mostly Western news text. It gets African names and unusual formatting wrong (see below). |
| **Streamlit + Streamlit Community Cloud** | Free, and a working demo needs only Python code and a GitHub repository. | |
| **Google Colab for the main experiment** | `gensim` and the datasets installed there without problems. On my computer (Python 3.14) the `datasets` library failed with an `HfUriError`. | |

## 4. Exploring word embeddings (GloVe 50d)

Nearest neighbours of **rain**: rains (0.878), torrential (0.843), winds (0.833), downpour (0.801), snow (0.795).

Analogy **king - man + woman** gives **queen** (0.852).

Words from my field (rainwater harvesting), top 5 neighbours with similarity scores:

| catchment | cistern | runoff |
|---|---|---|
| floodplain 0.808 | cisterns 0.804 | run-off 0.830 |
| reservoir 0.774 | outfall 0.695 | landslide 0.675 |
| drainage 0.768 | chimney 0.670 | elections 0.667 |
| catchments 0.752 | earthen 0.655 | reelection 0.660 |
| streams 0.747 | cavern 0.652 | election 0.659 |

"Catchment" works well. "Runoff" returns election words, because a "runoff" is also a type of vote, and a static embedding gives each word only one vector, so the meanings are blended. This shows that a general embedding can misunderstand specialized words.

## 5. First attempt: spaCy on 1,000 news articles

Data: the first 1,000 articles of AG News (title + description), loaded as CSV with pandas, with a backslash glitch cleaned.

Entity counts found by spaCy: ORG 1762, GPE 946, DATE 800, PERSON 789, CARDINAL 456, NORP 342 (plus smaller types). The most common ORG entries were "AP" (245) and "Reuters" (193), which are only news-agency tags at the start of the articles.

**Mistakes I found:**
- **ATHENS (all capitals):** labeled ORG 23 times and GPE 6 times. When ", Greece" followed it: GPE 5, ORG 2 (7 cases). Without a country: ORG 21, GPE 1 (22 cases). In a test sentence "ATHENS (Reuters) - ...", spaCy did not label ATHENS at all, but "ATHENS, Greece (Reuters) - ..." gave GPE. Missing context matters.
- **"Sans Serif" labeled PERSON** (8 times): it is leftover HTML, a font name.
- **Kigali:** missed at the start of the sentence "Kigali is the capital of Rwanda.", but found in "She traveled to Kigali last week."
- **My own name, "Claudine" and "Claudine Uwase":** not recognized. My guess is that the name is rare in the training data (not tested).
- **"East Africa"** was labeled GPE, although LOC (a region) is arguably better.

## 6. Streamlit demo app

`app.py` shows the text with highlighted entities, a table with each label's meaning and sentence, a count chart, and a sidebar explaining the labels.

I also added a **list of known names** (spaCy `entity_ruler`) that are always labeled PERSON. I did this because the model misses some Rwandan names. This is a fix by list, not machine learning, and it does not use embeddings. Names not on the list can still be missed, and a name that is also an ordinary word would be labeled PERSON wrongly.

## 7. Main experiment: NER with word embeddings (CoNLL-2003)

**Data:** 14,041 training, 3,250 validation and 3,453 test sentences. Test set: 5,648 entities.

**Two models, same classifier, same evaluation:**
- **Before (no embeddings):** word ID of the current, previous and next word (lowercased) + 4 capitalization/digit features.
- **After (GloVe):** GloVe vectors of the current, previous and next word (150 numbers) + the same 4 features. Words missing from GloVe get zeros.

### Results (test set)

| Type | Precision before | Recall before | F1 before | Precision GloVe | Recall GloVe | F1 GloVe |
|---|---|---|---|---|---|---|
| LOC | 0.793 | 0.740 | 0.766 | 0.721 | 0.794 | 0.756 |
| MISC | 0.728 | 0.652 | 0.688 | 0.567 | 0.638 | 0.601 |
| ORG | 0.605 | 0.606 | 0.605 | 0.565 | 0.633 | 0.597 |
| PER | 0.649 | 0.785 | 0.710 | 0.803 | 0.843 | 0.823 |
| **ALL** | 0.683 | 0.703 | 0.693 | 0.677 | 0.742 | **0.708** |

### What improved, what did not, and why

- **PER improved a lot (F1 0.710 to 0.823).** A word ID says nothing about a name that never appeared in training, but a GloVe vector places it near other names. This is the most likely reason, but I did not test it directly.
- **Overall F1 improved slightly (0.693 to 0.708),** mostly through higher recall (0.703 to 0.742). Precision stayed about the same.
- **MISC got worse (0.688 to 0.601).** MISC mixes many kinds of things (for example nationalities and events). Exact word IDs may remember frequent words such as "German" better than blended vectors do. This is a guess I did not test.
- **LOC and ORG stayed about the same.**

## 8. Limitations

- One run, no tuning, no repeated runs, so small differences (such as LOC and ORG) may be noise.
- The model reached the 100-iteration limit (ConvergenceWarning) in both versions.
- Simple classifier with a 3-word window.
- GloVe 50d is small and static: one vector per word, so words with several meanings (like "runoff") are blended.
- CoNLL-2003 is Western news from 1996. It has no DATE label, so dates were not evaluated here, and it will not represent Rwandan names well.
- The live demo uses spaCy's model, not the classifier from section 7.

## 9. How to run

**Main experiment:**
1. Open `ner_embeddings.ipynb` in Google Colab.
2. Run the cells from top to bottom (about 5 minutes). If Colab restarts, run the first cell again.

**Demo app on your computer:**
```
pip install -r requirements.txt
streamlit run app.py
```

## 10. Files

- `ner_embeddings.ipynb`: baseline and GloVe experiment (section 7)
- `app.py`, `requirements.txt`: Streamlit demo (section 6)
- `README.md`: this file

## Sources

- Pennington, J., Socher, R., & Manning, C. (2014). *GloVe: Global Vectors for Word Representation*. Stanford NLP.
- Mikolov, T., et al. (2013). *Efficient Estimation of Word Representations in Vector Space*.
- Tjong Kim Sang, E. & De Meulder, F. (2003). *Introduction to the CoNLL-2003 shared task: Language-independent named entity recognition*.
- gensim, spaCy, scikit-learn and Streamlit documentation.
