import create_vector_store as cvs

llms = [
    "gpt2",
    "gpt2-large",
    "distilgpt2",
    "microsoft/phi-2",
]
queries = [
    "What is the grading breakdown?",
    "Is there a final exam?",
    "What are the course prereqs?",
    "What books do I need for the class?",
    "What is the late policy?",
    "Tell me about the final project."
]

for llm in llms:
    print(f"\nLLM Model: {llm}\n")
    for query in queries:
        print(f"Query: {query}\n")
        doc_chat = cvs.DocChat(model_id=llm)
        response = doc_chat.qNa(query, just_answer=True)
        print(f"Response from {llm} LLM model:\n{response}\n")

        filename = f"responses/{llm}_syllabus_response.txt"
        with open(filename, "a") as file:
            file.write(
                f"Query:\n\t{query}\n\nResponse:\n\t{response}\n--------------------\n"
            )
