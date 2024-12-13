#!/usr/bin/env python
# coding: utf-8

# # Local Deployment 

# In[ ]:


import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import random
import os

# ----------------------------- Helper Functions -----------------------------

@st.cache_resource
def load_pickle_file(file_path):
    """Load a pickle file with caching."""
    with open(file_path, "rb") as file:
        return pickle.load(file)

@st.cache_data
def load_csv_file(file_path):
    """Load a CSV file with caching."""
    return pd.read_csv(file_path)

def calculate_cosine_similarity(vec1, vec2):
    """Calculate cosine similarity between two vectors."""
    return cosine_similarity(vec1.reshape(1, -1), vec2.reshape(1, -1))[0][0]

def recommend_news(user_profile, news_embeddings_map, news_dataset, sample_size=1000, top_n=20):
    """
    Recommend news based on user profile.
    """
    sampled_news_ids = random.sample(list(news_embeddings_map.keys()), min(sample_size, len(news_embeddings_map)))
    recommendations = []

    for news_id in sampled_news_ids:
        embedding = news_embeddings_map[news_id]
        preference_score = calculate_cosine_similarity(user_profile["preference_profile"], embedding)
        non_preference_score = calculate_cosine_similarity(user_profile["non_preference_profile"], embedding)

        if preference_score > non_preference_score:
            recommendations.append((news_id, preference_score, non_preference_score))

    recommendations = sorted(recommendations, key=lambda x: x[1], reverse=True)
    top_news = recommendations[:top_n]

    recommended_news = news_dataset[news_dataset["News ID"].isin([x[0] for x in recommendations])]
    recommended_news["Preference Score"] = recommended_news["News ID"].map(dict((x[0], x[1]) for x in recommendations))
    recommended_news["Non-Preference Score"] = recommended_news["News ID"].map(dict((x[0], x[2]) for x in recommendations))

    return recommended_news, recommended_news.sort_values(by="Preference Score", ascending=False).head(top_n)

# ----------------------------- Initialization -----------------------------

if "all_feedback" not in st.session_state:
    st.session_state.all_feedback = {}
if "feedback" not in st.session_state:
    st.session_state.feedback = {"liked": [], "neutral": [], "disliked": [], "read": [], "skipped": []}
if "current_recommendations" not in st.session_state:
    st.session_state.current_recommendations = pd.DataFrame()
if "last_user" not in st.session_state:
    st.session_state.last_user = None
if "active_category" not in st.session_state:
    st.session_state.active_category = None

# Display loading message while data is being prepared
with st.spinner("Loading data..."):
    # Load data with caching for faster performance
    news_embeddings_map = load_pickle_file('/Users/n7/Desktop/ie University SAMBD Acadamics/Capstone Project Revised/Data/MINDlarge_train/Cleaned Datasets/news_embeddings_map.pkl')
    user_profiles = load_pickle_file('/Users/n7/Desktop/ie University SAMBD Acadamics/Capstone Project Revised/Machine Learning models/Final Codes/Trained Models Ver. 9.0/user_profiles.pkl')
    news_dataset = load_csv_file('/Users/n7/Desktop/ie University SAMBD Acadamics/Capstone Project Revised/Data/MINDlarge_train/Cleaned Datasets/News_cleaned.csv')
    default_user_profile = load_pickle_file('/Users/n7/Desktop/ie University SAMBD Acadamics/Capstone Project Revised/Machine Learning models/Final Codes/Trained Models Ver. 9.0/default_user_profile.pkl')

# ----------------------------- Streamlit App -----------------------------

# App title
st.markdown(
    """
    <style>
        .main-title {
            text-align: center;
            font-size: 2.5em;
            font-weight: bold;
            color: #4CAF50;
            margin-bottom: 10px;
        }
        .sub-title {
            text-align: center;
            font-size: 1.5em;
            color: #555;
            margin-bottom: 30px;
        }
        .news-container {
            background-color: #f9f9f9;
            border: 1px solid #ddd;
            border-radius: 10px;
            padding: 15px;
            margin-bottom: 15px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            position: relative;
        }
        .news-title {
            font-size: 1.2em;
            font-weight: bold;
            color: #333;
        }
        .news-category {
            font-size: 1em;
            font-weight: bold;
            color: #777;
        }
        .news-abstract {
            font-size: 1em;
            font-style: italic;
            color: #555;
            margin-bottom: 10px;
        }
        .news-score {
            position: absolute;
            bottom: 10px;
            right: 10px;
            font-size: 0.9em;
            font-weight: bold;
            color: #4CAF50;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="main-title">News Recommendation System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Discover Your Personalized News Feed</div>', unsafe_allow_html=True)

# Sidebar User Selection
user_selection = st.sidebar.selectbox(
    "Select User ID or Create New",
    options=["New User"] + list(user_profiles.keys())
)

if user_selection == "New User":
    new_user_id = st.sidebar.text_input("Enter New User ID (Format: UXXXXXXX):")
    if new_user_id:
        if new_user_id in user_profiles:
            st.sidebar.error("User ID already exists! Please use another ID.")
        elif not new_user_id.startswith("U") or not new_user_id[1:].isdigit():
            st.sidebar.error("Invalid User ID format!")
        else:
            user_selection = new_user_id
            st.session_state.all_feedback[user_selection] = {"liked": [], "neutral": [], "disliked": [], "read": [], "skipped": []}
            st.session_state.feedback = {"liked": [], "neutral": [], "disliked": [], "read": [], "skipped": []}
            active_user_profile = default_user_profile
            st.sidebar.success(f"Welcome, User {new_user_id}! Your personalized news feed awaits. The more you interact the more presonlized news feed will be displayed to you saving time and increasing productivity over time.")
else:
    active_user_profile = user_profiles[user_selection]
    st.sidebar.success(f"Welcome back, User {user_selection}! We’re glad to have you here.")

if user_selection != st.session_state.last_user:
    if st.session_state.last_user:
        st.session_state.all_feedback[st.session_state.last_user] = st.session_state.feedback
    st.session_state.feedback = st.session_state.all_feedback.get(user_selection, {"liked": [], "neutral": [], "disliked": [], "read": [], "skipped": []})
    st.session_state.last_user = user_selection

# Generate Recommendations
if st.sidebar.button("Generate Recommendations"):
    with st.spinner("Computing recommendations..."):
        all_recommendations, top_20_recommendations = recommend_news(
            active_user_profile, news_embeddings_map, news_dataset
        )
        st.session_state.current_recommendations = all_recommendations

# Export User Interactions
if st.sidebar.button("Export User Interactions"):
    feedback_path = '/Users/n7/Desktop/ie University SAMBD Acadamics/Capstone Project Revised/Machine Learning models/Final Codes/Trained Models Ver. 9.0/feedback_summary.csv'
    
    # Save current user's feedback before exporting
    if user_selection not in st.session_state.all_feedback:
        st.session_state.all_feedback[user_selection] = st.session_state.feedback
    else:
        st.session_state.all_feedback[user_selection].update(st.session_state.feedback)

    # Combine feedback for all users
    feedback_combined = {
        user: {
            "liked": ", ".join(data["liked"]),
            "neutral": ", ".join(data["neutral"]),
            "disliked": ", ".join(data["disliked"]),
            "read": ", ".join(data["read"]),
            "skipped": ", ".join(data["skipped"]),
        }
        for user, data in st.session_state.all_feedback.items()
    }
    
    feedback_df = pd.DataFrame.from_dict(feedback_combined, orient="index").reset_index()
    feedback_df.rename(columns={"index": "User ID"}, inplace=True)
    
    # Export to CSV
    feedback_df.to_csv(feedback_path, index=False)
    st.sidebar.success(f"User interactions exported to {feedback_path}")

# Display Recommendations
if not st.session_state.current_recommendations.empty:
    tabs = st.tabs(["Trending News Recommendations for You", "News You Might Be Interested In"])

    # Top 20 Recommendations Tab
    with tabs[0]:
        st.markdown("## Trending News Recommendations for You")
        for idx, row in st.session_state.current_recommendations.sort_values(by="Preference Score", ascending=False).head(20).iterrows():
            st.markdown(f"""
                <div class="news-container">
                    <h3 class="news-title">{row['Title']}</h3>
                    <p class="news-category">Category: {row['Category']} | Subcategory: {row['Subcategory']}</p>
                    <p class="news-abstract">{row['Abstract']}</p>
                    <div class="news-score">Score: {row['Preference Score']:.4f}</div>
                </div>
            """, unsafe_allow_html=True)

            # Interaction buttons
            cols = st.columns(5)
            if cols[0].button("📖", key=f"read_{row['News ID']}"):
                st.session_state.feedback["read"].append(row["News ID"])
            if cols[1].button("⏭️", key=f"skip_{row['News ID']}"):
                st.session_state.feedback["skipped"].append(row["News ID"])
            if cols[2].button("👍", key=f"like_{row['News ID']}"):
                st.session_state.feedback["liked"].append(row["News ID"])
            if cols[3].button("🤷‍♂️", key=f"neutral_{row['News ID']}"):
                st.session_state.feedback["neutral"].append(row["News ID"])
            if cols[4].button("👎", key=f"dislike_{row['News ID']}"):
                st.session_state.feedback["disliked"].append(row["News ID"])

    # News You Might Be Interested In Tab
    with tabs[1]:
        preferred_categories = st.session_state.current_recommendations[
            st.session_state.current_recommendations["Preference Score"] >
            st.session_state.current_recommendations["Non-Preference Score"]
        ]["Category"].unique()

        st.markdown("## News You Might Be Interested In")

        # Display categories in a grid layout
        category_cols = st.columns(3)
        for i, category in enumerate(preferred_categories):
            if category_cols[i % 3].button(category):
                st.session_state.active_category = category

        # Show relevant news for the active category
        if st.session_state.active_category:
            st.markdown(f"### {st.session_state.active_category} News")
            category_news = st.session_state.current_recommendations[
                st.session_state.current_recommendations["Category"] == st.session_state.active_category
            ].sort_values(by="Preference Score", ascending=False)
            for idx, row in category_news.iterrows():
                unique_key = f"{row['News ID']}_{idx}"  # Add the index to make the key unique
                st.markdown(f"""
                    <div class="news-container">
                        <h3 class="news-title">{row['Title']}</h3>
                        <p class="news-category">Category: {row['Category']} | Subcategory: {row['Subcategory']}</p>
                        <p class="news-abstract">{row['Abstract']}</p>
                        <div class="news-score">Score: {row['Preference Score']:.4f}</div>
                    </div>
                """, unsafe_allow_html=True)

                # Interaction buttons
                cols = st.columns(5)
                if cols[0].button("📖", key=f"read_{unique_key}"):
                    st.session_state.feedback["read"].append(row["News ID"])
                if cols[1].button("⏭️", key=f"skip_{unique_key}"):
                    st.session_state.feedback["skipped"].append(row["News ID"])
                if cols[2].button("👍", key=f"like_{unique_key}"):
                    st.session_state.feedback["liked"].append(row["News ID"])
                if cols[3].button("🤷‍♂️", key=f"neutral_{unique_key}"):
                    st.session_state.feedback["neutral"].append(row["News ID"])
                if cols[4].button("👎", key=f"dislike_{unique_key}"):
                    st.session_state.feedback["disliked"].append(row["News ID"])


# End
