import requests
from datetime import datetime

prometheus_url = "http://localhost:9090"
query_string = "go_memstats_alloc_bytes_total[5m]"
current_time = datetime.now().strftime("%s")

# response = requests.post(
#     f"{prometheus_url}/api/v1/query",
#     params={"query": query_string, "time": current_time},
# )

# Get all merics
response = requests.get(f"{prometheus_url}/api/v1/label/__name__/values")

if response.status_code == 200:
    prom_output = response.json()["data"]
    print(f"Available output: {prom_output}")
else:
    print(f"Request failed with status code {response.status_code}: {response.text}")

