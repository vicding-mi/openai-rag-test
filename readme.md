# OpenAI Trial Overview

## Execution
To have a quick run, please make use of the FastAPI-based web application. It contains the findings and tweaks found in 
the latest test, namely <sub>main2.ipynb</sub>

Begin by installing the necessary dependencies. Optionally, create a virtual environment:

```shell
pip install -r requirements.txt
```

Subsequently, add your key in the app.py then execute the app:

```shell
# in app.py line 9
OPENAI_API_KEY = '<YOUR-API-KEY>'

# run from command line
uvicorn app:app --host 0.0.0.0 --port 38000 --reload
```

Access the app via `http://localhost:38000`.

This application is a basic demonstration of RAG, focusing solely on providing answers from given documents. 
It's possible for expansion to include the entire NLP workflow to handle dynamically given documents.
