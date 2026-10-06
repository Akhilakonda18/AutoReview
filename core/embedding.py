from sklearn.feature_extraction.text import TfidfVectorizer


# Create the TF-IDF embedding model
vectorizer = TfidfVectorizer(
    stop_words="english",
    max_features=5000
)


def generate_embeddings(texts):
    """
    Convert a list of documents into numerical vectors.
    """

    embeddings = vectorizer.fit_transform(texts)

    return embeddings


def generate_query_embedding(query):
    """
    Convert a new query into the same vector space.
    """

    embedding = vectorizer.transform([query])

    return embedding


if __name__ == "__main__":

    documents = [
        """
        Retrieval Augmented Generation improves large language
        model responses by retrieving relevant research papers.
        """,

        """
        Automated research paper review uses artificial intelligence
        to evaluate methodology, novelty and experimental results.
        """,

        """
        Peer review is an important process for evaluating
        scientific research and improving manuscript quality.
        """
    ]

    embeddings = generate_embeddings(documents)

    print("=" * 60)
    print("EMBEDDING GENERATOR TEST")
    print("=" * 60)

    print("Number of documents:", embeddings.shape[0])

    print("Number of features:", embeddings.shape[1])

    print("Embedding matrix shape:", embeddings.shape)