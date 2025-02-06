import streamlit as st
from transformers import pipeline
import pandas as pd
import plotly.express as px
import re
import time
import torch
from collections import Counter
import numpy as np

# Load sentiment analysis model with GPU support
@st.cache_resource
def load_sentiment_model():
    device = 0 if torch.cuda.is_available() else -1
    return pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment", device=device)

sentiment_model = load_sentiment_model()
# Preprocess reviews (remove slangs, typos, emojis, etc.)
def preprocess_review(review):
    review = re.sub(r'[^\w\s]', '', review)  # Remove special characters
    review = re.sub(r'\s+', ' ', review).strip()  # Remove extra spaces
    return review.lower()  # Convert to lowercase

# Check authenticity of review
def check_authenticity(review):
    return "Verified" if "verified purchase" in review.lower() else "Unverified"

# Function to preprocess reviews
def preprocess_reviews(df):
    df["Cleaned_Review"] = df["Review"].apply(preprocess_review)
    df["Authenticity"] = df["Review"].apply(check_authenticity)
    st.write(df)
    return df

# Function to perform sentiment analysis with batch processing
def perform_sentiment_analysis(df, progress_bar, status_text):
    reviews = df["Cleaned_Review"].tolist()

    status_text.text("🔄 Running Sentiment Analysis...")  # Update title
    progress_bar.progress(40)  # Update progress to 40%

    # Assuming sentiment_model is defined elsewhere
    results = sentiment_model(reviews, truncation=True, max_length=512, batch_size=16)

    sentiments, scores = [], []

    for result in results:
        sentiment_score = float(result['score']) * 5  # Convert to a 5-point scale
        sentiment_label = map_sentiment_label(sentiment_score)

        sentiments.append(sentiment_label)
        scores.append(sentiment_score)

    df["Sentiment"] = sentiments
    df["Sentiment_Score"] = scores

    status_text.text("✅ Sentiment Analysis Completed!")  # Final title update
    progress_bar.progress(100)  # Update progress to 100%
    return df

def map_sentiment_label(score):
    if score >= 4.5:
        return "Very Positive"
    elif score >= 3.5:
        return "Positive"
    elif score >= 2.5:
        return "Neutral"
    elif score >= 1.5:
        return "Negative"
    else:
        return "Very Negative"
# Function to calculate the average star rating and display stars
def calculate_avg_star_rating(df):

    avg_star_rating = df["Star_Rating"].mean()  # Calculate average star rating from the dataset
    star_rating_rounded = round(avg_star_rating)  # Round the value to the nearest whole number
    return avg_star_rating, star_rating_rounded

# Function to calculate avg sentiment based on sentiment label
def calculate_avg_sentiment(df):
    sentiment_map = {
        "Very Positive": 5,
        "Positive": 4,
        "Neutral": 3,
        "Negative": 2,
        "Very Negative": 1
    }
    
    sentiment_scores = [sentiment_map[sentiment] for sentiment in df["Sentiment"]]
    avg_sentiment = np.mean(sentiment_scores)  # Average sentiment based on processed labels
    return round(avg_sentiment, 2)  # Keep avg sentiment to 2 decimal places

# Function to map avg sentiment score to sentiment label for delta
def map_avg_sentiment_label_for_delta(avg_sentiment):
    if 4.5 <= avg_sentiment <= 5:
        return "Very Positive"
    elif 3.8 <= avg_sentiment < 4.5:
        return "Positive"
    elif 3 <= avg_sentiment < 3.8:
        return "Mixed"
    elif 2 <= avg_sentiment < 3:
        return "Negative"
    else:
        return "Very Negative"

# Display metrics for product reviews
def display_metrics(product_reviews):
    total_reviews = len(product_reviews)

    # Calculate average star rating
    avg_star_rating, star_rating_rounded = calculate_avg_star_rating(product_reviews)

    # Calculate average sentiment
    avg_sentiment = calculate_avg_sentiment(product_reviews)
    
    # Display Avg Star Rating with stars
    star_rating_display = "⭐" * round(avg_star_rating) + "☆" * (5 - round(avg_star_rating))
    st.markdown(f"### Average Star Rating: {avg_star_rating:.2f} ({star_rating_display})")
    
    # Display Average Sentiment and Delta based on Avg Sentiment
    avg_sentiment_label = map_avg_sentiment_label_for_delta(avg_sentiment)

    # Display Average Sentiment
   

    # Sentiment percentages
    positive_reviews = product_reviews[product_reviews["Sentiment"].isin(["Positive", "Very Positive"])]
    negative_reviews = product_reviews[product_reviews["Sentiment"].isin(["Negative", "Very Negative"])]
    neutral_reviews = product_reviews[product_reviews["Sentiment"] == "Neutral"]

    positive_percentage = (len(positive_reviews) / total_reviews) * 100
    negative_percentage = (len(negative_reviews) / total_reviews) * 100
    neutral_percentage = (len(neutral_reviews) / total_reviews) * 100

    # Verified purchase percentage
    verified_reviews = product_reviews[product_reviews["Authenticity"] == "Verified"]
    verified_percentage = (len(verified_reviews) / total_reviews) * 100

    # Display Metrics using Streamlit columns
    col1, col2, col3 = st.columns(3)
    if avg_sentiment_label == "Very Positive" or avg_sentiment_label == "Positive":
        col1.metric("Average Sentiment", avg_sentiment, f"{avg_sentiment_label}", border=True)
    elif avg_sentiment_label == "Mixed":
        col1.metric("Average Sentiment", avg_sentiment,avg_sentiment_label,delta_color="off", border=True)
    else:
        col1.metric("Average Sentiment", avg_sentiment, f"- {avg_sentiment_label}", border=True)
    col2.metric("Total Reviews", total_reviews, border=True)
    col3.metric("Verified Purchase (%)", f"{verified_percentage:.2f}%", border=True)
    

    col4, col5, col6 = st.columns(3)
    col4.metric("Positive Reviews (%)", f"{positive_percentage:.2f}%", border=True)
    col5.metric("Neutral Reviews (%)", f"{neutral_percentage:.2f}%", border=True)
    col6.metric("Negative Reviews (%)", f"{negative_percentage:.2f}%", border=True)


# Function to display product analysis including reviews
def display_product_analysis(df):
    product_names = df["Product_Name"].unique()
    selected_product = st.selectbox("Select Product", product_names, key="product_selector")

    product_reviews = df[df["Product_Name"] == selected_product]

    # Adding a custom serial number starting from 1
    product_reviews.insert(0, 'Sr No', range(1, len(product_reviews) + 1))

    # Resetting the index and dropping the old index
    product_reviews = product_reviews.reset_index(drop=True)

    # Renaming the index to 'S.No' and ensuring custom serial numbers are displayed
    product_reviews = product_reviews.rename_axis("S.No").reset_index(drop=True)

    st.write(f"### Reviews for {selected_product}")

    # Display the dataframe with only the necessary columns and hide the index
    st.dataframe(product_reviews[["Sr No", "Review", "Sentiment", "Sentiment_Score", "Authenticity"]], hide_index=True)

    with st.container(border=True):
        st.markdown("### Sentiment Metrics & Visuals 📊")
        st.divider()  # Border Separator
        display_metrics(product_reviews)

    with st.container(border=True):
        display_features(product_reviews)
    with st.container(border=True):
        st.markdown("### Fake Review Analysis 🕵️‍♂️")
        st.divider()  # Border Separator
        analyze_fake_reviews(product_reviews)
    with st.container(border=True):
        st.markdown("### 📈 Sentiment Analysis Overview")
        display_product_overview(product_reviews)

# Function to visualize sentiment distribution using Plotly
def visualize_sentiment_distribution(product_reviews):
    sentiment_distribution = product_reviews["Sentiment"].value_counts().reset_index()
    sentiment_distribution.columns = ["Sentiment", "Count"]

    fig_bar = px.bar(
        sentiment_distribution,
        x="Sentiment",
        y="Count",
        color="Sentiment",
        title="Sentiment Distribution",
        labels={"Count": "Number of Reviews", "Sentiment": "Sentiment"},
    )
    st.plotly_chart(fig_bar)

    # Sentiment Pie Chart
    fig_pie = px.pie(
        sentiment_distribution,
        names="Sentiment",
        values="Count",
        title="Sentiment Breakdown",
        color="Sentiment"
    )
    st.plotly_chart(fig_pie)

# Function to visualize sentiment vs authenticity (Stacked Bar Chart)
def visualize_sentiment_vs_authenticity(product_reviews):
    sentiment_auth = product_reviews.groupby(["Sentiment", "Authenticity"]).size().reset_index(name="Count")

    fig_stack = px.bar(
        sentiment_auth,
        x="Sentiment",
        y="Count",
        color="Authenticity",
        barmode="stack",
        title="Sentiment vs. Authenticity",
        labels={"Count": "Number of Reviews", "Sentiment": "Sentiment", "Authenticity": "Review Type"},
    )
    st.plotly_chart(fig_stack)


# Extract features from reviews
def extract_features(reviews, sentiment):
    feature_keywords = ["sound", "quality", "battery", "comfort", "design", "performance", "durability", "price"]
    extracted_features = []

    for review in reviews:
        for keyword in feature_keywords:
            if re.search(rf"\b{keyword}\b", review, re.IGNORECASE):
                extracted_features.append(keyword)

    feature_counts = Counter(extracted_features)
    return feature_counts

# Display good & bad features
def display_features(product_reviews):
    st.write("###  Key Features Mentioned in Reviews 🔍")
    st.divider()

    # Separate positive and negative reviews
    positive_reviews = product_reviews[product_reviews["Sentiment"].isin(["Positive", "Very Positive"])]["Review"]
    negative_reviews = product_reviews[product_reviews["Sentiment"].isin(["Negative", "Very Negative"])]["Review"]

    # Extract features from both categories
    positive_features = extract_features(positive_reviews, "positive")
    negative_features = extract_features(negative_reviews, "negative")

    # Display positive features (Green Buttons)
    st.write("####  Good Features")
    st.markdown("""
        The following features are frequently mentioned in positive reviews.""")
    for feature, count in positive_features.items():
        if st.button(f"🟢 {feature.capitalize()} ({count})", key=f"good_{feature}"):
            show_reviews(product_reviews, feature, "positive")

    # Display negative features (Red Buttons)
    st.write("####  Bad Features")
    st.markdown("""
        These features are often associated with negative reviews.""")
    for feature, count in negative_features.items():
        if st.button(f"🔴 {feature.capitalize()} ({count})", key=f"bad_{feature}"):
            show_reviews(product_reviews, feature, "negative")

    # Additional info section
    st.info("Click on above features to view reviews mentioning it.", icon="ℹ️")

    # Provide a brief explanation about the significance of features
    st.markdown("""
        - **Good Features**: These are attributes that customers appreciate, highlighting what makes the product stand out positively.
        - **Bad Features**: These are aspects that users criticize, pointing out areas for improvement or dissatisfaction.
    """)

    st.markdown("""
        **Tip:** Keep an eye on both good and bad features to understand the product's strengths and weaknesses before purchasing.
    """)


# Show reviews based on selected feature
def show_reviews(product_reviews, feature, sentiment_type):
    sentiment_filter = ["Positive", "Very Positive"] if sentiment_type == "positive" else ["Negative", "Very Negative"]
    
    filtered_reviews = product_reviews[
        (product_reviews["Sentiment"].isin(sentiment_filter)) & 
        (product_reviews["Review"].str.contains(feature, case=False, na=False))
    ]

    if not filtered_reviews.empty:
        st.markdown(f"<h6 style='margin-bottom:10px;'>Reviews mentioning '{feature}' ({'✅ Good' if sentiment_type == 'positive' else '❌ Bad'})</h6>", unsafe_allow_html=True)

        # Create a formatted DataFrame with Serial Numbers
        formatted_reviews = filtered_reviews.reset_index(drop=True)
        formatted_reviews.index = formatted_reviews.index + 1  # Serial number starts from 1
        formatted_reviews = formatted_reviews[["Review", "Sentiment"]].rename_axis("S.No").reset_index()

        # Apply custom CSS for better UI
        st.markdown("""
            <style>
                .dataframe { 
                    border-radius: 10px; 
                    overflow: hidden; 
                    box-shadow: 0px 2px 10px rgba(0,0,0,0.1);
                }
                table { width: 100%; border-collapse: collapse; }
                th { background-color: #f8f9fa; text-align: left; padding: 10px; }
                td { padding: 10px; border-bottom: 1px solid #ddd; }
                tr:hover { background-color: #f1f1f1; }
            </style>
        """, unsafe_allow_html=True)

        st.dataframe(formatted_reviews, hide_index=True, use_container_width=True)
    else:
        st.warning(f"No reviews found mentioning '{feature}'.")


# Function to display product overview based on sentiment and fake review analysis
def display_product_overview(product_reviews):
    st.divider()

    # Sentiment analysis breakdown
    positive_reviews = len(product_reviews[product_reviews["Sentiment"].isin(["Positive", "Very Positive"])])
    negative_reviews = len(product_reviews[product_reviews["Sentiment"].isin(["Negative", "Very Negative"])])
    total_reviews = len(product_reviews)

    sentiment_percentage = (positive_reviews / total_reviews) * 100

    # Display sentiment overview
    st.write(f"**Sentiment Overview:**")
    st.write(f"Total Reviews: {total_reviews}")
    st.write(f"Positive Sentiment: {positive_reviews} ({sentiment_percentage:.2f}%)")
    st.write(f"Negative Sentiment: {negative_reviews} ({100 - sentiment_percentage:.2f}%)")

    # Combine sentiment analysis with fake review analysis for a comprehensive overview
    fake_review_percentage = len(product_reviews[product_reviews["Authenticity"] == "Unverified"]) / total_reviews * 100

    # Define status based on both fake review percentage and sentiment analysis
    if fake_review_percentage > 50:
            product_status = "This product is likely a scam."
            product_advice = "Consider carefully before making a purchase."
            st.error("**Product Overview:** The high percentage of unverified reviews raises doubts about the authenticity of this product. Proceed with caution.")
    elif sentiment_percentage > 70 and fake_review_percentage <= 30:
        product_status = "This product is highly praised."
        product_advice = "It's generally well-received and trustworthy."
        st.success("**Product Overview:** This product has overwhelmingly positive reviews and low fake review percentages, suggesting it’s a trustworthy option.")
    elif 30 <= fake_review_percentage <= 50:
        product_status = "This product might be suspicious. Proceed with caution."
        product_advice = "There's a significant amount of unverified reviews. Investigate further."
        st.warning("**Product Overview:** While sentiment is mostly positive, the significant percentage of fake reviews calls for further investigation.")
    elif sentiment_percentage < 30 and fake_review_percentage < 30:
        # New condition: More verified reviews but low sentiment score
        product_status = "This product has genuine but negative feedback."
        product_advice = "Although reviews are verified, they are largely negative. You might want to reconsider the purchase."
        st.warning("**Product Overview:** Despite the higher percentage of verified reviews, the sentiment is overwhelmingly negative. Consider other factors before purchasing.")
    else:
        product_status = "This product is generally well-received."
        product_advice = "The product has mostly positive feedback with a low number of fake reviews."
        st.success("**Product Overview:** The product has mostly positive reviews with few fake ones, making it a generally trustworthy choice.")



    # Display comprehensive product overview
    st.info(f"**Advice:** {product_advice}")
    st.markdown("""
    - **Sentiment Analysis**: Indicates general customer satisfaction (positive or negative).
    - **Fake Reviews**: Reflects the authenticity of the product's reviews, with higher percentages of fake reviews suggesting possible misleading information.
    """)

    # Add a final disclaimer for caution
    st.markdown("**Disclaimer:** Always read through reviews carefully, and consider reviewing multiple sources before making a purchase.")

def analyze_fake_reviews(product_reviews):
    # Filter fake reviews
    fake_reviews = product_reviews[product_reviews["Authenticity"] == "Unverified"]
    
    # Calculate percentage of fake reviews
    fake_review_percentage = len(fake_reviews) / len(product_reviews) * 100

    # Add custom serial numbers starting from 1
    formatted_reviews = fake_reviews.reset_index(drop=True)
    formatted_reviews.index = formatted_reviews.index + 1  # Serial number starts from 1
    formatted_reviews = formatted_reviews[["Review", "Sentiment"]].rename_axis("Sr No").reset_index()

    # Display progress bar
    with st.spinner('Analyzing fake reviews...'):
        time.sleep(3)  # Simulate time for processing, can be removed if not necessary

    # Display the fake reviews with serial numbers
    st.markdown(f"#### Fake Reviews (Total: {len(fake_reviews)} reviews)")
    st.markdown(f"Below is a list of reviews flagged as **Unverified**. These reviews may contain misleading information or false claims.")

    st.dataframe(formatted_reviews, hide_index=True, use_container_width=True)

    # Display percentage of fake reviews with a prominent message
    st.write(f"**Percentage of Fake Reviews:** {fake_review_percentage:.2f}%")

    # Add some additional insights for clarity
    st.markdown("#### Insights:")
    st.markdown("""
        - A high percentage of **Unverified** reviews could suggest that the product is not trustworthy.
        - Fake reviews often aim to mislead customers by boosting product ratings or downplaying negative feedback.
        - Consider reading a mix of verified reviews from different sources before making a purchasing decision.
    """)

    # Display the opinion based on the percentage of fake reviews
    if fake_review_percentage > 50:
        st.error("❌ **This product is likely a scam.**")
        st.markdown("""
            **Warning:** More than half of the reviews are flagged as unverified, which indicates a high likelihood of false claims. 
            Be cautious and do thorough research before making any purchase decision.
        """)
    elif 30 <= fake_review_percentage <= 50:
        st.warning("⚠️ **This product might be suspicious. Proceed with caution.**")
        st.markdown("""
            **Advice:** There's a significant portion of unverified reviews. While the product might not be entirely fake, proceed with caution 
            and consider checking alternative review platforms for more authentic feedback.
        """)
    else:
        st.success("✅ **This product is generally well-received.**")
        st.markdown("""
            **Good News:** The low percentage of unverified reviews indicates that the product has received mostly authentic feedback.
            You can likely trust the reviews, but it's still important to read through some of them to form a balanced opinion.
        """)

    # Final note for users
    st.markdown("""
        **Note:** Even though the product may have mostly authentic reviews, always consider the overall context and other factors such as 
        product details, brand reputation, and user experience.
    """)


# Main app function with tabs and session state
# Custom CSS Styling
st.markdown(
    """
    <style>
    /* Centered Heading */
    .title {
        text-align: center;
        color: #FF6F61;
        font-size: 36px;
        font-weight: bold;
    }
    
    /* Divider Styling */
    .custom-divider {
        border: 2px solid #FF6F61;
        margin-top: 15px;
        margin-bottom: 15px;
    }
    
    /* Features Box */

.feature-box {
    background-color: #FCEFF5; /* Light pink background */
    padding: 15px;
    border-radius: 12px;
    width: 220px; /* Square box */
    height: 220px; /* Square box */
    display: flex;
    flex-direction: column; /* Stack title and text vertically */
    align-items: center; /* Centers content horizontally */
    justify-content: center; /* Centers content vertically */
    text-align: center;
    margin: 15px; /* Adds spacing between boxes */
    justify-self: center; /* Centers the box */
}

/* Ensuring text alignment inside the box */
.feature-box b {
    display: block;
    text-align: center; /* Centers the title */
    font-size: 18px;
    margin-bottom: 5px; /* Adds spacing below title */
}

/* Left-aligned text below the title */
.feature-box p {
    text-align: center; /* Left-aligns the bullet point */
    font-size: 14px;
    width: 90%; /* Ensures proper alignment */
    margin: 0; /* Removes default margin */
}

/* Important Notes Styling */
.important-notes {
    background-color: #FFF8E1;
    padding: 15px;
    border-radius: 10px;
    box-shadow: 0px 2px 10px rgba(0, 0, 0, 0.1);
}

    </style>
    """,
    unsafe_allow_html=True,
)

def main():
    st.markdown("<h1 class='title'>🛍️ Amazon Review Sentiment Analysis and Authenticity Verification</h1>", unsafe_allow_html=True)

    if "df" not in st.session_state:
        st.session_state.df = None

    tab1, tab2 = st.tabs(["🏠 Home", "📊 Analysis | Authenticity"])

    ### ---- HOME TAB ---- ###
    with tab1:
        st.write("")
        st.markdown("### 💬 **Unveiling the App!**")
        st.markdown("<hr class='custom-divider'>", unsafe_allow_html=True)

        # Two-column layout for a more structured design
        col1, col2 = st.columns([2, 1])

        with col1:
            st.markdown(
                """
                This application is designed to help you **analyze Amazon product reviews**, 
                **detect fake reviews**, and **gain insights into customer sentiment**.  
                
                It utilizes **AI-powered sentiment analysis and fake review detection** to ensure 
                **authentic product feedback analysis**.
                """
            )

        with col2:
            st.image("https://cdn-icons-png.flaticon.com/512/869/869636.png", width=180)
        st.write("")
        st.write("")
        st.markdown("### 🌟 **Key Features**")
        # Features section in two columns with added spacing
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("<div class='feature-box'><b>Sentiment Analysis</b><p>Identify Positive, Neutral, and Negative Reviews.</p></div>", unsafe_allow_html=True)
            st.markdown("<div class='feature-box'><b>Fake Review Detection</b><p>Spot Unverified or Misleading Reviews.</p></div>", unsafe_allow_html=True)

        with col2:
            st.markdown("<div class='feature-box'><b>Feature Analysis</b><p>Highlight Key Pros and Cons of a Product.</p></div>", unsafe_allow_html=True)
            st.markdown("<div class='feature-box'><b>Product Overview</b><p>Combines Sentiment & Authenticity for Insights.</p></div>", unsafe_allow_html=True)

        st.write("")
        st.write("")
        st.markdown(
                    """
                    <style>
                    /* Wide box container */
                    .how-it-works-box {
                        background-color: #EFF7E4; /* Light green background */
                        padding: 25px;
                        border-radius: 12px;
                        display: flex;
                        align-items: center;
                        justify-content: space-between;
                        width: 100%;
                    }

                    .how-it-works-left {
                    font-size: 26px;
                    font-weight: bold;
                    padding-left: 20px;
                    margin-top: 0; /* Remove any default margin */
                    }

                    /* Right side grid */
                    .how-it-works-grid {
                        display: grid;
                        grid-template-columns: 1fr 1fr;
                        gap: 10px;
                        flex: 2;
                    }

                    /* Individual step box */
                    .step-box {
                        padding: 10px;
                        display: flex;
                        align-items: center;
                        gap: 10px;
                    }

                    /* Step number */
                    .step-number {
                        color: #4DE560; /* Green number */
                        font-size: 24px;
                        font-weight: bold;
                        display: flex;
                        align-self: flex-start; 
                    }

                    /* Step text */
                    .step-text {
                        font-size: 16px;
                    }
                    </style>    """,
                    unsafe_allow_html=True
                )
        st.markdown(
                    """
                    <div class='how-it-works-box'>
                    <div class='how-it-works-left'>How does it work?</div>
                    <div class='how-it-works-grid'>
                        <div class='step-box'>
                        <div class='step-number'>1</div>
                        <div class='step-text'>
                            <b>📂 Upload Amazon Reviews (CSV)</b><br>
                            Select and upload a CSV file containing product reviews. The system will process the data automatically.
                        </div>
                    </div>

                    <div class='step-box'>
                        <div class='step-number'>2</div>
                        <div class='step-text'>
                            <b>🧠 AI-Powered Analysis</b><br>
                            Our advanced AI model will analyze the sentiment (positive, neutral, negative) and detect fake reviews.
                        </div>
                    </div>

                    <div class='step-box'>
                        <div class='step-number'>3</div>
                        <div class='step-text'>
                            <b>🔍 Review Breakdown</b><br>
                            View a detailed summary of reviews, including sentiment scores and authenticity verification.
                        </div>
                    </div>

                    <div class='step-box'>
                        <div class='step-number'>4</div>
                        <div class='step-text'>
                            <b>📊 Get a Final Trust Score</b><br>
                            The system will generate an overall trust score based on sentiment and fake review analysis.
                        </div>
                    </div>
                        </div>
                    </div>
                    </div>
                    """,unsafe_allow_html=True)
        st.write("")
        # Custom CSS for styling the boxes
        st.markdown(
            """
            <style>
            .model-box-1{
                background-color: #E12C6E;
                padding: 15px;
                border-radius: 10px;
                text-align: center;
                margin: 5px;
                height: 150px;
            }
            .model-box-2{
                background-color: #B5ED80;
                padding: 15px;
                border-radius: 10px;
                text-align: center;
                margin: 5px;
                height: 150px;
            }
            .model-box-3{
                background-color: #100208;
                padding: 15px;
                color: white;
                border-radius: 10px;
                text-align: center;
                margin: 5px;
            }
            .model-box-4{
                background-color: #4DE560;
                padding: 15px;
                border-radius: 10px;
                text-align: center;
                margin: 5px;
            }
            .model-box-title {
                font-size: 22px;
                font-weight: bold;
                margin-bottom: 10px;
            }
            .model-box-text {
                font-size: 16px;
            }
            .container {
                display: flex;
                justify-content: space-around;
            }
            </style>
            """, 
            unsafe_allow_html=True
        )
        st.write("")
        # Heading
        st.markdown("### 🧠 NLP Model Used")

        # First row of boxes
        col1, col2 = st.columns([3,7])

        with col1:
            st.markdown(
                f"""
                <div class='model-box-1'>
                    <div class='model-box-title'>Hugging Face Transformers</div>
                    <div class='model-box-text'>DistilBERT</div>
                </div>
                """, unsafe_allow_html=True)

        with col2:
            st.markdown(
                f"""
                <div class='model-box-2'>
                    <div class='model-box-title'>60% Fewer Parameters</div>
                    <div class='model-box-text'>Faster & Lighter</div>
                </div>
                """, unsafe_allow_html=True)

        # Second row of boxes
        col3, col4 = st.columns([6,4])

        with col3:
            st.markdown(
                f"""
                <div class='model-box-3'>
                    <div class='model-box-title'>Sentiment Classification</div>
                    <div class='model-box-text'>Positive, Neutral, and Negative</div>
                </div>
                """, unsafe_allow_html=True)

        with col4:
            st.markdown(
                f"""
                <div class='model-box-4'>
                    <div class='model-box-title'>Fake Review Detection</div>
                    <div class='model-box-text'>Unverified or Misleading Reviews</div>
                </div>
                """, unsafe_allow_html=True)
        st.write("")
        st.write("")
        st.markdown("### 💡 Why Use This Tool?")
        st.markdown(
            """
            -  **Quickly Identify Genuine Reviews.**  
            -  **Make Smarter Purchasing Decisions.**  
            -  **Improve Product Reputation & Customer Trust.**  
            """
        )
        st.write("")
        st.markdown("### 📢 **Important Notes**")
        st.markdown("<div class='important-notes'>⚡ This tool helps you filter fake reviews and analyze sentiment trends, but final judgment should be based on multiple sources.</div>", unsafe_allow_html=True)

    ### ---- ANALYSIS TAB ---- ###
    with tab2:
        st.write("")
        st.markdown("### 📊 **Start Your Analysis**")
        st.markdown("<hr class='custom-divider'>", unsafe_allow_html=True)

        uploaded_file = st.file_uploader("📂 Upload a CSV file containing reviews", type="csv")

        if uploaded_file is not None:
            if st.button("🚀 Start Processing", use_container_width=True):
                status_text = st.empty()  # Placeholder for progress title
                progress_bar = st.progress(0)  # Initialize progress bar

                status_text.text("🔄 Loading CSV File...")
                progress_bar.progress(10)
                time.sleep(1)

                # Read file
                st.session_state.df = pd.read_csv(uploaded_file)

                status_text.text("🔄 Preprocessing Reviews...")
                progress_bar.progress(30)
                time.sleep(1)

                # Preprocess reviews
                st.session_state.df = preprocess_reviews(st.session_state.df)

                # Perform sentiment analysis
                st.session_state.df = perform_sentiment_analysis(st.session_state.df, progress_bar, status_text)

                progress_bar.empty()  # Remove progress bar after completion
                st.success("🎉 Processing Completed!")

        # Display results if data exists
        if st.session_state.df is not None:
            display_product_analysis(st.session_state.df)
    # Footer
    st.markdown(
        """
        <hr class='custom-divider'>
        <p style='text-align: center; font-size: 14px; color: grey;'>Developed with ❤️ By rohitp189</p>
        """,
        unsafe_allow_html=True,
    )
if __name__ == "__main__":
    main()
