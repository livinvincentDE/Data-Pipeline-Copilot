from app.search import load_documents, search


# Load our knowledge base
documents = load_documents()

print(f"Loaded {len(documents)} documents")


# Test question
query = "Why is my Spark job slow?"

print("\nQuery:")
print(query)


# Search
results = search(
    query=query,
    documents=documents,
    top_k=3
)


print("\nTop Results:")

for result in results:

    document = result["document"]

    print("\n-------------------------")

    print("Score:", result["score"])

    print("ID:", document["id"])

    print("Technology:", document["technology"])

    print("Title:", document["title"])

    print("Content:", document["content"])