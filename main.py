from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def get_document_content(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        return content
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {file_path}")
    except IOError as e:
        raise IOError(f"Error reading file {file_path}: {str(e)}")


def check_plagiarism_cosine(doc1, doc2):
    
    vectorizer = TfidfVectorizer()

    
    tfidf_matrix = vectorizer.fit_transform([doc1, doc2])

    
    similarity_matrix = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])

    return similarity_matrix[0][0]


def main():
    # Example usage
    document1 = get_document_content("doc1.txt")
    document2 = get_document_content("doc2.txt")
    document3 = get_document_content("doc3.txt")   

    similarity1 = check_plagiarism_cosine(document1, document2)
    similarity2 = check_plagiarism_cosine(document1, document3)

    print(f"Cosine similarity between document1 and document2: {similarity1:.2f} \n {document1} \n {document2}")
    print(f"Cosine similarity between document1 and document3: {similarity2:.2f} \n {document1} \n {document3}")

if __name__ == "__main__":
    main()


