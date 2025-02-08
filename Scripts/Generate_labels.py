import requests
import xml.etree.ElementTree as ET
from datetime import datetime
import pandas as pd


def fetch_categorie(categorie, max_results=10):
    base_url = 'http://export.arxiv.org/api/query?'
    query = f"""search_query=cat:"{categorie}"&sortBy=submittedDate&sortOrder=descending&max_results={max_results}"""
    response = requests.get(base_url + query)
    papers = []

    if response.status_code == 200:
        data = response.text
        root = ET.fromstring(data)
        entries = root.findall('.//{http://www.w3.org/2005/Atom}entry')

        for entry in entries:
            paper = {
                'paper_id': entry.find('{http://www.w3.org/2005/Atom}id').text.split('/')[-1],
                'title': entry.find('{http://www.w3.org/2005/Atom}title').text,
                'abstract': entry.find('{http://www.w3.org/2005/Atom}summary').text,
                'categories': ', '.join([category.get('term') for category in entry.findall('{http://www.w3.org/2005/Atom}category')]),
                'submission_date': entry.find('{http://www.w3.org/2005/Atom}updated').text
            }
            papers.append(paper)
    else:
        print(f"Failed to fetch data for categorie={categorie}: {response.status_code}")
    return papers


def get_interest(papers):
    ids = []
    for paper in papers:
        print("-" * 50)
        print(f"""Title: {paper["title"]}\n""")
        print(paper["abstract"])
        response = input("Garder ? (y/n): ")
        if response == "y":
            ids.append(paper["paper_id"])
        print()
    return ids


paper_ids = []
interest_ids = []
papers = []
max_results = 100
categories = ["cs.AI", "cs.LG", "cs.CL"]

for categorie in categories:
    papers_categorie = fetch_categorie(categorie, max_results=max_results)
    for paper in papers_categorie:
        id = paper["paper_id"]
        if not(id in paper_ids):
            papers.append(paper)
            paper_ids.append(id)
            print("-" * 50)
            print(f"""Title: {paper["title"]}\n""")
            print(paper["abstract"])
            response = input("Garder ? (y/n): ")
            if response == "y":
                interest_ids.append(paper["paper_id"])
            print()




df = pd.DataFrame(papers, columns=["paper_id", "title", "abstract", "categories", "submission_date"])
df["interest"] = df["paper_id"].apply(lambda id: id in interest_ids).astype(int)

now = datetime.now()
now = now.strftime("%Y-%M-%d_%Hh%M")
df.to_csv(f"/Users/lionellopes-quintas/Desktop/ArxivBase/Arxiv_labeled_{now}.csv", index=False)
