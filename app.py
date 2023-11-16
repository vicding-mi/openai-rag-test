import pandas as pd
import openai
import numpy as np
from ast import literal_eval
from scipy.spatial.distance import cosine
from fastapi import FastAPI, Body
from fastapi.responses import HTMLResponse

OPENAI_API_KEY = '<YOUR-API-KEY>'
openai.api_key = OPENAI_API_KEY
embedding_csv = "embedding.csv"


def load_embeddings(file=embedding_csv):
    df = pd.read_csv(file, index_col=0)
    df['embeddings'] = df['embeddings'].apply(literal_eval).apply(np.array)
    return df


df = load_embeddings(file=embedding_csv)
app = FastAPI()


def get_form(action: str | None, input_name: str = "question_text"):
    if not action or action == "":
        return f"""
                <head><title>Answer</title></head>
                <form action="/" enctype="multipart/form-data" method="post">
                <textarea name="{input_name}" rows="4" cols="50"></textarea>
                <input type="submit">
                </form>
                """
    return f"""
    <head><title>Answer</title></head>
    <form action="/{action}" enctype="multipart/form-data" method="post">
    <textarea name="{input_name}" rows="4" cols="50"></textarea>
    <input type="submit">
    </form>
    """


@app.get("/")
def ask():
    return HTMLResponse(get_form(action=None, input_name="question_text"))


@app.post("/")
def answer(question_text: str = Body(...)):
    result = answer_question(df, question=question_text.strip())
    return HTMLResponse(f"<p>{result}</p>" +
                        get_form(action=None, input_name="question_text"))


def get_distance(emb1, emb2):
    len1 = len(emb1)
    len2 = len(emb2)
    if len1 > len2:
        emb2 = np.pad(emb2, (0, len1 - len2))
    elif len2 > len1:
        emb1 = np.pad(emb1, (0, len2 - len1))

    distance = cosine(emb1, emb2)
    return distance


def create_context(question, df, max_len=1800, size="ada"):
    """
    Create a context for a question by finding the most similar context from the dataframe
    """
    # Get the embeddings for the question
    q_embeddings = openai.embeddings.create(input=question, model='text-embedding-ada-002').dict()['data'][0][
        'embedding']

    # Get the distances from the embeddings
    # df['distances'] = distances_from_embeddings(q_embeddings, df['embeddings'].values, distance_metric='cosine')
    q_embeddings = np.array(q_embeddings)

    df["distances"] = df.embeddings.apply(lambda x: get_distance(x, q_embeddings))

    returns = []
    cur_len = 0

    # Sort by distance and add the text to the context until the context is too long
    for i, row in df.sort_values('distances', ascending=True).iterrows():

        # Add the length of the text to the current length
        cur_len += row['num_tokens'] + 4

        # If the context is too long, break
        if cur_len > max_len:
            break

        # Else add it to the text that is being returned
        returns.append(row["text"])

    # Return the context
    return "\n\n###\n\n".join(returns)


def answer_question(
        df,
        model="text-davinci-003",
        question="Am I allowed to publish model outputs to Twitter, without a human review?",
        max_len=1800,
        size="ada",
        debug=False,
        max_tokens=150,
        stop_sequence=None
):
    """
    Answer a question based on the most similar context from the dataframe texts
    """
    context = create_context(
        question,
        df,
        max_len=max_len,
        size=size,
    )
    # If debug, print the raw model response
    if debug:
        print("Context:\n" + context)
        print("\n\n")

    try:
        # Create a completions using the question and context
        response = openai.completions.create(
            prompt=f"Answer the question based on the context below, and if the question can't be answered based on the context, say \"I don't know\"\n\nContext: {context}\n\n---\n\nQuestion: {question}\nAnswer:",
            temperature=0,
            max_tokens=max_tokens,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            stop=stop_sequence,
            model=model,
        )
        return response.dict()["choices"][0]["text"].strip()
    except Exception as e:
        print(e)
        return ""
