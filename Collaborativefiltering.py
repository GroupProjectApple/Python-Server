from pymongo import MongoClient
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
import pandas as pd
import numpy as np
from bson import ObjectId
import pika
import json
import os
import socket
import time

# Connect to RabbitMQ
def connect_to_rabbitmq():
    CLOUDAMQP_URL = "amqps://tkmzwfdi:L8M8wKlnoUA37hyz1GRrlch8ufJY3mys@fuji.lmq.cloudamqp.com/tkmzwfdi"

    # Parse the URL into a connection parameters dictionary
    params = pika.URLParameters(CLOUDAMQP_URL)

    # Create a connection and a channel
    connection = pika.BlockingConnection(params)
    channel = connection.channel()
    channel.queue_declare(queue='user_actions', durable=False)
    return channel

# Process the received message
def process_message(ch, method, properties, body):
    message = json.loads(body)
    print(f"Received message: {message}")
    
    # Example AIML processing logic
    user_id = message.get("userId")
    #product_id = message.get("productId")
    #event = message.get("event")

    # Placeholder for AIML processing logic
    recommend(user_id)

    ch.basic_ack(delivery_tag=method.delivery_tag)

def start_consuming():
    channel = connect_to_rabbitmq()
    channel.basic_consume(queue='user_actions', on_message_callback=process_message)
    print("Waiting for messages in 'user_actions' queue...")
    channel.start_consuming()
    

# MongoDB connection setup
client = MongoClient('mongodb+srv://groupprojectapple:SamarahaisGreat%402025@cluster0.8cy6o.mongodb.net/mydatabase?retryWrites=true&w=majority')
db = client["mydatabase"]
search_phrases = db["search_phrases"]
user_interactions = db["user_interactions"]
product_collection = db["product1"]
users=db["users"]

def fetch_data_for_user(uid):
    """Fetch user search phrases, visited product details, and product data."""
    # Fetch user search phrases
    user_phrases = ""
    user_doc = search_phrases.find_one({"Uid": uid})
    if user_doc:
        user_phrases = " ".join(user_doc.get("phrases", []))

    # Fetch visited product details
    visited_details = []
    interactions_doc = user_interactions.find_one({"Uid": uid})
    if interactions_doc:
        for product in interactions_doc.get("products", []):
            product_context = " ".join([
                product.get("name", ""),
                " ".join(product.get("tags", [])),
                product.get("category_level_2", ""),
                product.get("category_level_3", ""),
                product.get("description", ""),
                str(product.get("rating", ""))
            ])
            visited_details.append(product_context)

    # Fetch all product data
    product_data = []
    for product in product_collection.find():
        product_details = product.get("product_details", {})
        product_data.append({
            "name": product_details.get("name", "Unnamed Product"),
            "tags": " ".join(product_details.get("tags", [])),
            "category_level_2": product_details.get("category_level_2", ""),
            "category_level_3": product_details.get("category_level_3", ""),
            "description": product_details.get("description", ""),
            "img": str(product_details.get("imageUrl", "")),
            "_id": str(product.get("_id", ObjectId())),
            "rating": str(product_details.get("rating", ""))
        })

    return user_phrases, visited_details, product_data

def process_data_for_user(user_phrases, visited_details, product_data):
    """Preprocess and vectorize data for a user."""
    product_df = pd.DataFrame(product_data)
    if product_df.empty:
        return None, None, None

    product_df["content"] = (
        product_df["tags"] + " " +
        product_df["category_level_2"] + " " +
        product_df["category_level_3"] + " " +
        product_df["description"] + " " +
        product_df["rating"]
    )

    user_context = user_phrases + " " + " ".join(visited_details)
    all_text = [user_context] + product_df["content"].tolist()

    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(all_text)

    user_vector = tfidf_matrix[0]
    product_vectors = tfidf_matrix[1:]

    return product_df, user_vector, product_vectors

def recommend_for_user(product_df, user_vector, product_vectors, visited_details):
    """Generate product recommendations for a user."""
    similarities = cosine_similarity(user_vector, product_vectors).flatten()
    product_df["similarity"] = similarities
    
    visited_set = set(visited_details)
    product_df["is_visited"] = product_df["content"].isin(visited_set)
    
    unvisited_products = product_df[~product_df["is_visited"]]
    grouped = unvisited_products.groupby("category_level_2")

    diverse_recommendations = []
    for _, group in grouped:
        top_products = group.sort_values(by=["similarity", "rating"], ascending=[False, False]).head(3)
        diverse_recommendations.append(top_products)

    if diverse_recommendations:
        diverse_recommendations = pd.concat(diverse_recommendations).sort_values(by=["similarity","rating"], ascending=[False, False])
    else:
        diverse_recommendations = pd.DataFrame(columns=product_df.columns)

    final_recommendations = diverse_recommendations.head(5)[["name", "_id", "rating", "similarity","img"]]
    return final_recommendations

def generate_all_user_recommendations():
    """Generate recommendations for all users."""
    all_user_recommendations = []
    for user_doc in search_phrases.find():
        uid = user_doc.get("Uid")
        if uid == -1:
            continue

        user_phrases, visited_details, product_data = fetch_data_for_user(uid)
        if user_phrases or visited_details:
            product_df, user_vector, product_vectors = process_data_for_user(user_phrases, visited_details, product_data)
            if product_df is not None:
                recommendations = recommend_for_user(product_df, user_vector, product_vectors, visited_details)
                all_user_recommendations.append({
                    "uid": uid,
                    "recommendation_scores": recommendations.to_dict("records")
                })
    return all_user_recommendations

def find_optimal_clusters(recommendation_matrix):
    """Determine the optimal number of clusters using the elbow method."""
    inertia_values = []
    for k in range(1, 11):
        kmeans = KMeans(n_clusters=k, random_state=0)
        kmeans.fit(recommendation_matrix)
        inertia_values.append(kmeans.inertia_)

    optimal_k = np.argmin(np.diff(inertia_values)) + 2
    return optimal_k

def cluster_users(recommendations):
    """Cluster users based on recommendation scores."""
    # Extract similarity scores into a matrix
    recommendation_matrix = []
    for user in recommendations:
        scores = [rec.get("similarity", 0) for rec in user["recommendation_scores"]]
        recommendation_matrix.append(scores)

    # Ensure all rows have the same length
    max_length = max(len(row) for row in recommendation_matrix)
    recommendation_matrix = np.array([row + [0] * (max_length - len(row)) for row in recommendation_matrix])

    # Find the optimal number of clusters
    optimal_k = find_optimal_clusters(recommendation_matrix)

    # Perform clustering
    kmeans = KMeans(n_clusters=optimal_k, random_state=0)
    user_clusters = kmeans.fit_predict(recommendation_matrix)

    # Assign clusters back to users
    for i, user in enumerate(recommendations):
        user["cluster"] = user_clusters[i]

    return recommendations


def collaborative_filtering_within_cluster(user_id, recommendations):
    """Apply collaborative filtering within a user's cluster with higher priority for content-based filtering."""
    user_cluster = next((user["cluster"] for user in recommendations if user["uid"] == user_id), None)
    if user_cluster is None:
        return None

    cluster_users = [user for user in recommendations if user["cluster"] == user_cluster]

    # Create a matrix of similarities for content-based filtering
    user_matrix = np.array([
        [rec.get("similarity", 0) for rec in user["recommendation_scores"]]
        for user in cluster_users
    ])

    # Find the index of the target user
    user_index = next((i for i, user in enumerate(cluster_users) if user["uid"] == user_id), None)
    if user_index is None:
        return None

    # Compute similarities between users in the cluster
    user_similarities = cosine_similarity(user_matrix)[user_index]
    top_similar_users_indices = np.argsort(user_similarities)# Exclude the target user, sort descending

    # Step 1: Start with the content-based recommendations for the user
    user_recommendations = pd.DataFrame(cluster_users[user_index]["recommendation_scores"])

    # Step 2: Refine the content-based recommendations with collaborative filtering (use similar users' recommendations)
    collaborative_recommendations = []
    for i in top_similar_users_indices:
        # Get the recommendations of top similar users
        collaborative_recommendations.extend(cluster_users[i]["recommendation_scores"])

    collaborative_recommendations = pd.DataFrame(collaborative_recommendations)
    
    # Combine the content-based and collaborative recommendations, giving priority to the content-based ones
    final_recommendations = pd.concat([user_recommendations, collaborative_recommendations], ignore_index=True)
    
    # Rank the final recommendations by similarity (from content-based) and rating (to prioritize better-rated products)
    final_recommendations = final_recommendations.sort_values(by=["similarity", "rating"], ascending=[False, False])
    final_recommendations = final_recommendations.drop_duplicates(subset=["_id"])

    return final_recommendations.head(5)



def recommend(uid):
    """Main function to execute the recommendation pipeline."""
    recommendations = generate_all_user_recommendations()
    if not recommendations:
        print("No recommendations generated.")
        return

    clustered_users = cluster_users(recommendations)
    user_id = uid  # Example user ID

    final_recommendations = collaborative_filtering_within_cluster(user_id, clustered_users)
    final_recommendations = final_recommendations.to_dict("records") if final_recommendations is not(None) else {}
    users.update_one(
        { "_id": ObjectId(user_id) },
        {"$set": { "recommendation": final_recommendations } })

    print(f"Final Recommendations for User {user_id}:")
    

if __name__ == "__main__":
    start_consuming()
     # Bind a port to satisfy Render's requirements
    port = int(os.environ.get("PORT", 8080))  # Render sets PORT via environment
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind(("0.0.0.0", port))
    print(f"Port {port} bound to keep the service alive.")

    # Keep the script running indefinitely
 
    
