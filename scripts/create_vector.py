import argparse 
import csv
import json
from bert_serving.client import BertClient


bc = BertClient(ip='104.197.11.222')


def main(args):
  with open(args.input) as input:
    read_input = csv.reader(input, delimiter=",")
    
    count = 1
    for line in read_input:
      print('Indexing document', count)
        
      # create an object from the csv row
      document = create_document(line)

      # create the text vector 
      title_vector = create_vector_field(document['title'])
  
      # add the title vector to the document
      document['title_vector'] = title_vector[0]
      
      # add the document with the vector to the output file
      write_document(args.output, document)
      
      count += 1
  


def create_document(line):  
  document = {
    'title': line[0],
    'text': line[1],
    'subject': line[2],
    'date': line[3]
  }
  return document


def create_vector_field(title):
  return bc.encode([title]).tolist()


def write_document(output, document):
  with open(output, 'a+') as output: 
    output.write(json.dumps(document) + '\n')
    output.close()
  
  

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description="Creating documents with a text vector")
  parser.add_argument("--input", help="Input csv file for creating elasticsearch documents")
  parser.add_argument("--output", help="Output file for elasticsearch documents")
  args = parser.parse_args()
  main(args) 