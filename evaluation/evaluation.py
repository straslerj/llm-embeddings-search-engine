import argparse
import matplotlib.pyplot as plt
import pandas as pd
import os
import seaborn as sns
import sys

from sentence_transformers import SentenceTransformer, util


model = SentenceTransformer("bert-base-nli-mean-tokens")


def compute_similarity(sentence1, sentence2):
    embeddings = model.encode([sentence1, sentence2])
    cosine_similarities = util.pytorch_cos_sim(
        embeddings[0].reshape(1, -1), embeddings[1].reshape(1, -1)
    )
    return cosine_similarities.item()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Flag for which set of questions and document to use"
    )

    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["nfl", "syllabus"],
        help="Specify the type (nfl or syllabus)",
    )

    args = parser.parse_args()

    mode_flag = args.mode

    if mode_flag == "nfl":
        print("Evaluating responses for NFL...")
    elif mode_flag == "syllabus":
        print("Evaluating responses for syllabus...")
    else:
        print(f"Invalid type: {mode_flag}")
        sys.exit(1)

    df = pd.read_csv(f"responses/{mode_flag}_responses.csv")

    context_col = "context"
    answer_col = "answer"
    llm_response_col = "llm_response"

    df["context_score"] = df.apply(
        lambda row: compute_similarity(row[context_col], row[answer_col]), axis=1
    )
    df["llm_score"] = df.apply(
        lambda row: compute_similarity(row[llm_response_col], row[answer_col]), axis=1
    )

    directory_path = "results/"
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
    df.to_csv(f"{directory_path}{mode_flag}_results.csv", index=False)

    df["context_score"] = pd.to_numeric(df["context_score"], errors="coerce")
    df["llm_score"] = pd.to_numeric(df["llm_score"], errors="coerce")

    df["context_score"] = pd.to_numeric(df["context_score"], errors="coerce")
    df["llm_score"] = pd.to_numeric(df["llm_score"], errors="coerce")

    df["context_score"] = pd.to_numeric(df["context_score"], errors="coerce")
    df["llm_score"] = pd.to_numeric(df["llm_score"], errors="coerce")

    print("Similarity scores calculated.")

    grouped_df = (
        df.groupby("llm")
        .agg(
            {
                "context_score": ["mean", "std"],
                "llm_score": ["mean", "std"],
            }
        )
        .reset_index()
    )

    grouped_df.columns = ["_".join(col).strip() for col in grouped_df.columns.values]

    melted_df = pd.melt(
        grouped_df, id_vars=["llm_"], var_name="type_stat", value_name="value"
    )

    melted_df["type"] = melted_df["type_stat"].apply(lambda x: x.split("_")[0])
    melted_df["stat"] = melted_df["type_stat"].apply(lambda x: x.split("_")[1])

    melted_df = melted_df.drop_duplicates(subset=["value"], keep="first")

    melted_df.loc[
        melted_df["type_stat"].isin(["context_score_mean", "context_score_std"]), "llm_"
    ] = "context"

    melted_df.to_csv(f"{directory_path}{mode_flag}_results_STATS.csv", index=False)

    sns.set_style("whitegrid", {"grid.linestyle": "--", "grid.alpha": 0.7})
    plt.figure(figsize=(15, 6))
    sns.barplot(x="llm_", y="value", hue="llm_", errorbar="sd", data=melted_df)
    plt.title("Mean and Standard Deviation by LLM")
    plt.xlabel("LLM")
    plt.ylabel("Similarity Score")
    directory_path = "results/figures/"
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
    plot_path = f"{directory_path}{mode_flag}.png"
    plt.tight_layout()
    plt.savefig(plot_path)
    print(f"Plot saved to {plot_path}")
