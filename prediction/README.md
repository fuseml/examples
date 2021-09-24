# Examples of parameters passed to prediction services

NOTE: Use full json related headers to curl request for seldon-core based prediction service. (See https://github.com/SeldonIO/seldon-core/issues/1770)

For example

```bash
curl -d @prediction/data-sklearn-seldon.json -X POST http://$INGRESS_HOST/seldon/fuseml-workloads/project-02-mlflow-seldon/api/v1.0/predictions -H "Accept: application/json" -H "Content-Type: application/json"
```

(You'll get the right URL from `fuseml application list` call)
