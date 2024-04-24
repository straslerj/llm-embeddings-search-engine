import create_vector_store as cvs

if __name__ == "__main__":
    doc_chat = cvs.DocChat(model_id="gpt2", mode="nfl")
    response = doc_chat.qNa(
        input("What would you like to know about the NFL rulebook?\n")
    )
    print(response)
