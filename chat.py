import create_vector_store as cvs
import time

if __name__ == "__main__":
    start_time = time.time()
    doc_chat = cvs.DocChat(model_id="gpt2", mode="nfl")
    response = doc_chat.qNa(
        input("What would you like to know about the NFL rulebook?\n")
    )
    print(response)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print("Elapsed time: {:.2f} seconds".format(elapsed_time))
