## Conversation Template Name
Finding information in logs Template

## Conversation Template Definition
The following template describes the process of searching information inside the user's log indices
Application log telemetry, is very useful for diagnosing specific issues in a specific system. 


## finding information in logs Conversation 

### Question
When a user requests to know which indices are inside his cluster:
> given this is the `_cat/indices` results

### Response
```text
    index status: green open, index name: sample-ecommerce
    index status: green open, index name: .kibana_92668751_admin_1
    index status: green open, index name: moviegeek-logs-2023.03.24
    index status: green open, index name: .opendistro-reports-definitions
    index status: green open, index name: .kibana_1
    index status: green open, index name: .opendistro_security
    index status: green open, index name: .opendistro-reports-instances
    index status: green open, index name: otel-v1-apm-span-000001
    index status: green open, index name: sample-host-health
    index status: green open, index name: security-auditlog-2023.03.24
    index status: green open, index name: imdb-data
    index status: green open, index name: .opensearch-observability
    index status: green open, index name: metricbeat-2023.03.24
    index status: green open, index name: opensearch_dashboards_sample_data_logs
    index status: green open, index name: opensearch_dashboards_sample_data_flights
    index status: green open, index name: sample-http-responses
    
    index status: green open, index name: sso_logs-nginx-prod
    index status: green open, index name: sso_traces-us-prod
    index status: green open, index name: sso_metrics-us-prod
    index status: green open, index name: sso_logs-aws-alb
    
    
    index status: green open, index name: otel-v1-apm-service-map
    index status: green open, index name: .opendistro-job-scheduler-lock
    index status: green open, index name: opensearch_dashboards_sample_data_ecommerce
    index status: green open, index name: .opensearch-notifications-config
    
```

### Question
When a user requests to know which indices are inside his cluster:
which logs indices I have in my system?

### Response 
```text
    Based on the `_cat/indices` results, the following indices appear to be related to logs:
    - moviegeek-logs-2023.03.24
    - security-auditlog-2023.03.24
    - sso_logs-nginx-prod
    - sso_traces-us-prod
    - sso_metrics-us-prod
    - sso_logs-aws-alb
  ```

### Question
When a user requests to know the mapping for a specific index

### Response
```text
Given this is a response for the `sso_logs-nginx-prod/_mapping` api call

index fields:
field name: @timestamp field, type: date
field name: body field, type: text
field name: container_id field, type: text
field name: event.properties.category field, type: keyword
field name: http.properties.client.properties.ip field, type: ip
field name: http.properties.flavor field, type: keyword
field name: http.properties.request.properties.body.properties.content field, type: text
field name: http.properties.request.properties.bytes field, type: long
field name: http.properties.request.properties.id field, type: text
field name: http.properties.request.properties.id.fields.keyword field, type: keyword
field name: http.properties.request.properties.method field, type: keyword
field name: http.properties.request.properties.mime_type field, type: keyword
field name: http.properties.request.properties.referrer field, type: keyword
field name: http.properties.resent_count field, type: integer
field name: http.properties.response.properties.body.properties.content field, type: text
field name: http.properties.response.properties.bytes field, type: long
field name: http.properties.response.properties.id field, type: text
field name: http.properties.response.properties.id.fields.keyword field, type: keyword
field name: http.properties.response.properties.status_code field, type: integer
field name: http.properties.route field, type: keyword
field name: http.properties.schema field, type: keyword
field name: http.properties.target field, type: keyword
field name: http.properties.url field, type: keyword
field name: http.properties.user_agent field, type: keyword
field name: log field, type: text
field name: log.fields.keyword field, type: keyword
field name: observedTimestamp field, type: date
field name: observerTime field, type: date
field name: schemaUrl field, type: text
field name: schemaUrl.fields.keyword field, type: keyword
field name: severity.properties.number field, type: long
field name: severity.properties.text field, type: text
field name: severity.properties.text.fields.keyword field, type: keyword
field name: source field, type: text
field name: source.fields.keyword field, type: keyword
field name: spanId field, type: keyword
field name: span_id field, type: text
field name: span_id.fields.keyword field, type: keyword
field name: traceId field, type: keyword
field name: trace_id field, type: text
field name: trace_id.fields.keyword field, type: keyword

```

### Question
does this index follow the simples schema for observability structure?

### Response
`sso_logs-nginx-prod` index follows the simple schema

### Question
> **_Q:_** can you create a ppl query to determine the average errors per day I encountered in my  sso_logs-nginx-prod log ?

### Response
```text
search from sso_logs-nginx-prod
| where http.properties.response.properties.status_code >= 400
| bucket _time span=1d
| stats count by _time
| stats avg(count) as avg_errors_per_day

```



### Question

Can you create a ppl query to retrieves the number of requests grouped by User-Agent in the same index ?

### Response

The next PPL query retrieves the number of requests grouped by User-Agent in the 'sso_logs-nginx-prod' log index.

```text
 `index=sso_logs-nginx-prod | stats count by http.properties.user_agent` 
 
```

