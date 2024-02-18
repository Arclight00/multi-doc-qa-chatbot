from flask import Flask, render_template, request, redirect, url_for
from main import QAChatbot, DataIngestion
from config import models_used, pinecone_index_name


app = Flask(__name__)


@app.route("/")
def main():
    return render_template("main.html")


@app.route("/data_ingest", methods=["GET", "POST"])
def data_ingest():
    if request.method == "POST":
        if "delete_kb" in request.form:
            # Delete knowledge base
            index_name = request.form["index_name"]
            if index_name == "":
                index_name = pinecone_index_name
            DataIngestion.delete_knowledgebase(index_name)
        else:
            # Data ingestion logic
            directory_path = request.form["directory_path"]
            file_path = request.form["file_path"]

            # Create knowledge base using the provided directory or file path
            data_ingest = DataIngestion(directory_path, file_path)
            data_ingest.run()
            return redirect(url_for("chatbot_route"))

    return render_template("data_ingest.html")


qa_chatbot = None


@app.route("/chatbot", methods=["GET", "POST"])
def chatbot_route():
    global qa_chatbot

    if request.method == "POST":
        selected_model = request.form["chat_model"]
        user_query = request.form["user_query"]

        # Update the chatbot with the selected model
        qa_chatbot = QAChatbot(model_type=selected_model)

        # Get chatbot response based on the user's query
        response = qa_chatbot.chat(user_query)

        # Separate the response into result and top_3_matching_chunks
        result = response.get("result", "")
        top_3_matching_chunks = response.get("source_documents", [])

        return render_template(
            "chatbot.html",
            available_models=models_used.keys(),
            selected_model=qa_chatbot.model_type,
            user_query=user_query,
            result=result,
            top_3_matching_chunks=top_3_matching_chunks,
        )

    return render_template(
        "chatbot.html",
        available_models=models_used.keys(),
        selected_model=qa_chatbot.model_type if qa_chatbot else "",
    )


if __name__ == "__main__":
    app.run()
