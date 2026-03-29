import os
import warnings
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from transformers import logging

# 1. Load environment variables from .env
load_dotenv()

# 2. Set the HF token and suppress the loading reports
os.environ["HF_TOKEN"] = os.getenv("HUGGINGFACEHUB_API_TOKEN", "")
logging.set_verbosity_error()
warnings.filterwarnings("ignore", category=UserWarning)

# Now continue with your LangChain and Gradio imports
from langchain_community.document_loaders import TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma
import gradio as gr

# 1. Data Loading
# 1. Data Loading
books = pd.read_csv("books_with_emotions.csv")

# ADD THIS LINE TO DEBUG:
print("Available columns:", books.columns.tolist())

# Check if 'simple_categories' is in that list.
# If it is missing, the code below will fail.

books["large_thumbnail"] = books["thumbnail"] + "&fife=w800"
books["large_thumbnail"] = np.where(
    books["large_thumbnail"].isna(),
    "cover-not-found.jpg",
    books["large_thumbnail"],
)

# 2. Vector Database Setup (LOCAL)
raw_documents = TextLoader("tagged_description.txt", encoding="utf-8").load()
text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=0)
documents = text_splitter.split_documents(raw_documents)

# Using a popular, free local model from Hugging Face
model_name = "sentence-transformers/all-MiniLM-L6-v2"
embeddings = HuggingFaceEmbeddings(model_name=model_name)

# This will create the database in memory using the local embeddings
db_books = Chroma.from_documents(documents, embeddings)


# 3. Recommendation Logic
def retrieve_semantic_recommendations(
        query: str,
        category: str = "All",
        tone: str = "All",
        initial_top_k: int = 50,
        final_top_k: int = 16,
) -> pd.DataFrame:
    recs = db_books.similarity_search(query, k=initial_top_k)
    books_list = [str(rec.page_content.strip('"').split()[0]) for rec in recs]
    book_recs = books[books["isbn13"].astype(str).isin(books_list)].copy()

    if category != "All":
        book_recs = book_recs[book_recs["simple _categories"] == category]

    tone_map = {
        "Happy": "joy",
        "Surprising": "surprise",
        "Angry": "anger",
        "Suspenseful": "fear",
        "Sad": "sadness"
    }

    if tone in tone_map:
        column = tone_map[tone]
        book_recs.sort_values(by=column, ascending=False, inplace=True)

    return book_recs.head(final_top_k)


def recommend_books(query: str, category: str, tone: str):
    recommendations = retrieve_semantic_recommendations(query, category, tone)
    results = []

    for _, row in recommendations.iterrows():
        description = str(row["description"]) if pd.notna(row["description"]) else ""
        truncated_desc = " ".join(description.split()[:30]) + "..."
        caption = f"{row['title']} by {row['authors']}: {truncated_desc}"
        results.append((row["large_thumbnail"], caption))
    return results


# 4. Interface
# Using 'categories' which is the column name in your processed dataset
# Convert all categories to string and replace NaN with "Unknown"
categories = ["All"] + sorted(books["categories"].fillna("Unknown").astype(str).unique().tolist())
tones = ["All", "Happy", "Surprising", "Angry", "Suspenseful", "Sad"]

with gr.Blocks(theme=gr.themes.Glass()) as dashboard:
    gr.Markdown("# Semantic Book Recommender (Local Mode)")

    with gr.Row():
        user_query = gr.Textbox(label="Describe a book vibe:", placeholder="e.g. A mystery in a small snowy town")
        category_dropdown = gr.Dropdown(choices=categories, label="Category:", value="All")
        tone_dropdown = gr.Dropdown(choices=tones, label="Tone:", value="All")
        submit_button = gr.Button("Find Books")

    output = gr.Gallery(label="Recommendations", columns=8, rows=2)
    submit_button.click(fn=recommend_books, inputs=[user_query, category_dropdown, tone_dropdown], outputs=output)

if __name__ == "__main__":
    dashboard.launch()