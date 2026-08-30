def detect_technology(question):

    question = question.lower()

    technology_keywords = {

        "Apache Spark": [
            "spark",
            "pyspark",
            "executor",
            "partition",
            "shuffle"
        ],

        "Apache Airflow": [
            "airflow",
            "dag",
            "scheduler",
            "task"
        ],

        "Apache Kafka": [
            "kafka",
            "consumer",
            "producer",
            "topic",
            "consumer lag"
        ],

        "dbt": [
            "dbt",
            "incremental model",
            "dbt model"
        ],

        "Data Quality": [
            "null",
            "duplicate",
            "duplicates",
            "schema drift",
            "data freshness",
            "data quality"
        ]
    }

    for technology, keywords in technology_keywords.items():

        for keyword in keywords:

            if keyword in question:
                return technology

    return "Unknown"