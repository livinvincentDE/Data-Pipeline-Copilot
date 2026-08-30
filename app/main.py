from app.rag import answer_question


def main():

    print("=" * 60)
    print("🤖 Data Pipeline Copilot")
    print("Type 'exit' to quit")
    print("=" * 60)

    while True:

        question = input("\nAsk a question: ")

        if question.lower() == "exit":
            print("\nGoodbye! 👋")
            break

        result = answer_question(question)

        print("\n" + "=" * 60)
        print("ANSWER")
        print("=" * 60)

        print(result["answer"])

        print("\n" + "=" * 60)
        print("RETRIEVED DOCUMENTS")
        print("=" * 60)

        for item in result["results"]:

            document = item["document"]

            print(
                f"- {document['id']} | "
                f"{document['title']} | "
                f"Score: {item['score']}"
            )


if __name__ == "__main__":
    main()