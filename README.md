# Research Bot for URL-Based Data Summarization and Insight Extraction
#### Link: https://huggingface.co/spaces/valleeneutral/article_research_bot

This project provides a robust research bot designed to assist users in summarizing and extracting meaningful insights from multiple URLs. By leveraging the capabilities of advanced text processing and vector storage, the bot ensures efficient data handling and insightful responses. The workflow involves loading data from URLs, splitting the data into manageable chunks, storing the data in a FAISS vector database, and responding to user prompts with accurate and concise summaries.

## Architechtural Overview
![arch_pix](https://github.com/fosetorico/tomato_disease_detection/assets/14139087/d06136c5-ae64-4c2f-9fe2-3eda2b80a352)

## Steps to run this Project?

#### Clone the repository
```
git clone https://github.com/fosetorico/article_research_bot.git
```

#### Create a conda environment after opening the repository
```
conda create -n your-chosen-name python=3.10 -y
```

```
conda activate your-chosen-name
```

#### install the requirements
```
pip install -r requirements.txt
```

#### Finally run the following command
```
streamlit run app.py
```
