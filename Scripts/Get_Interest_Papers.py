import requests
import xml.etree.ElementTree as ET
from datetime import datetime
import pandas as pd


def fetch_paper(paper_id, max_results=10):
    base_url = 'http://export.arxiv.org/api/query?'
    query = f"""search_query=id:'{paper_id}'"""
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
        print(f"Failed to fetch data for id={paper_id}: {response.status_code}")
    return papers[0]


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


interest_paper_id = ["2311.12983", "2308.08155", "2402.05120", "2210.03629", "2402.01030", "2310.06770", "2401.00812"]
papers = []

for paper_id in interest_paper_id:
    paper = fetch_paper(paper_id)
    papers.append(paper)




df = pd.DataFrame(papers, columns=["paper_id", "title", "abstract", "categories", "submission_date"])
df["interest"] = 1

now = datetime.now()
now = now.strftime("%Y-%M-%d_%Hh%M")
df.to_csv(f"/Users/lionellopes-quintas/Desktop/ArxivBase/Arxiv_labeled_{now}.csv", index=False)
