# 📚 Semantic Book Recommender

A semantic search-powered book recommendation system that suggests books based on natural language descriptions of vibes, themes, or moods — with filtering by category and emotional tone.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue) ![Gradio](https://img.shields.io/badge/Gradio-UI-orange) ![LangChain](https://img.shields.io/badge/LangChain-vector%20search-green) ![HuggingFace](https://img.shields.io/badge/HuggingFace-embeddings-yellow)

---

## Demo

Type a natural language description like *"a mystery set in a small snowy town"* and the app returns the 16 most semantically relevant books, optionally filtered by genre and emotional tone.

---

## Features

- **Semantic search** — finds books by meaning, not just keywords, using vector embeddings
- **Emotion-aware filtering** — sort results by tone: Happy, Sad, Suspenseful, Angry, or Surprising
- **Category filtering** — narrow results by genre/category
- **Local inference** — runs entirely offline using `sentence-transformers/all-MiniLM-L6-v2`
- **Gradio UI** — clean gallery interface with book covers and descriptions

---

## Project Structure

```
├── gradio-dashboard.py          # Main app — UI and recommendation logic
├── vector-search.ipynb          # Notebook: building the Chroma vector database
├── data-exploration.ipynb       # Notebook: EDA on the books dataset
├── books_cleaned.csv            # Base dataset after cleaning
├── books_with_categories.csv    # Dataset with category labels added
├── books_with_emotions.csv      # Final dataset with emotion scores
├── tagged_description.txt       # Book descriptions formatted for vector indexing
└── cover-not-found.jpg          # Fallback image for missing book covers
```

---

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/your-username/semantic-book-recommender.git
cd semantic-book-recommender
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

**Key dependencies:**

- `gradio`
- `langchain`, `langchain-community`, `langchain-huggingface`, `langchain-chroma`
- `transformers`, `sentence-transformers`
- `pandas`, `numpy`
- `python-dotenv`

### 3. Configure environment variables

Create a `.env` file in the root directory:

```
HUGGINGFACEHUB_API_TOKEN=your_token_here
```

> The HuggingFace token is optional if you're using only local models, but required for some HF Hub features.

### 4. Run the app

```bash
python gradio-dashboard.py
```

The app will launch at `http://localhost:7860`.

---

## How It Works

1. **Data pipeline** — Books are cleaned, categorized, and scored for five emotions (joy, surprise, anger, fear, sadness) using NLP models. See the notebooks for the full pipeline.

2. **Vector indexing** — Book descriptions from `tagged_description.txt` are embedded using `sentence-transformers/all-MiniLM-L6-v2` and stored in a local [Chroma](https://www.trychroma.com/) vector database.

3. **Query time** — The user's natural language query is embedded and compared against the database using cosine similarity. The top 50 candidates are retrieved, then filtered and re-ranked by category and/or emotional tone before returning the final 16 results.

---

## Emotion Tones

| UI Label | Underlying Score |
|---|---|
| Happy | `joy` |
| Surprising | `surprise` |
| Angry | `anger` |
| Suspenseful | `fear` |
| Sad | `sadness` |

---

## Notebooks

| Notebook | Purpose |
|---|---|
| `data-exploration.ipynb` | Exploratory data analysis — distributions, missing values, category breakdowns |
| `vector-search.ipynb` | Building and querying the Chroma vector store |

---

## License

MIT
