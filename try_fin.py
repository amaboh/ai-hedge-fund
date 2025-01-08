import requests

# add your API key to the headers
headers = {
    "X-API-KEY": "dd2be6cd-5932-42bf-bb1f-5c5d64597407"
}

# set your query params
ticker = 'AAPL'     # stock ticker
period = 'ttm'      # possible values are 'annual', 'quarterly', or 'ttm'
limit = 5          # number of statements to return

# create the URL
url = (
    f'https://api.financialdatasets.ai/financials/income-statements'
    f'?ticker={ticker}'
    f'&period={period}'
    f'&limit={limit}'
)

# make API request
response = requests.get(url, headers=headers)

# parse income_statements from the response
income_statements = response.json().get('income_statements')

print(income_statements)