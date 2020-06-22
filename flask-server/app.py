import os
from flask import Flask, request, jsonify
from elasticsearch import Elasticsearch
from bert_serving.client import BertClient
from dotenv import load_dotenv

load_dotenv()

INDEX_NAME = os.environ['INDEX_NAME']
BERT_HOST = os.environ['BERT_HOST']
ES_HOST = os.environ['ES_HOST']
ES_USER = os.environ['ES_USER']
ES_PASSWORD = os.environ['ES_PASSWORD']


app = Flask(__name__)


@app.route('/', methods=['GET'])
def index():
  return 'Index page'


@app.route('/search', methods=['GET'])
def search():
  bc = BertClient(ip=BERT_HOST, output_fmt='list')
  es = Elasticsearch(ES_HOST, http_auth=(ES_USER, ES_PASSWORD))
  
  query = request.args.get('q')
  query_vector = bc.encode([query])[0]
  
  script_query = {
    "script_score": {
      "query": {"match_all": {}},
        "script": {
          "source": "cosineSimilarity(params.query_vector, doc['title_vector']) + 1.0",
          "params": {"query_vector": query_vector}
        }
      }
    }
  
  response = es.search(
    index=INDEX_NAME,
    body={
      'size': 10,
      'query': script_query
    }
  )
  return jsonify(response)


if __name__ == '__main__':
  app.run(host='0.0.0.0', port=8000)