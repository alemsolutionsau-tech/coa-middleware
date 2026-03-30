import sys
sys.stdout.reconfigure(encoding='utf-8')
from dotenv import load_dotenv
load_dotenv()
print('Starting...')
from google.cloud import bigquery
print('BQ imported')
bq = bigquery.Client(project='alem-coa-ai')
print('BQ connected')
query = 'SELECT COUNT(*) as n FROM `alem-coa-ai.cannabis_coa.coa_results`'
rows = list(bq.query(query).result())
print('Rows:', rows[0].n)