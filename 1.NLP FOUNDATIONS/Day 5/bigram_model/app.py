import streamlit as st
import joblib

# Load model
bigram_probabilities = joblib.load("bigram_probs.pkl")

def predict_next_word(word, top_k=5):
    word = word.lower()
    if word not in bigram_probabilities:
        return [("No prediction found", 0)]

    sorted_words = sorted(
        bigram_probabilities[word].items(),
        key=lambda x: x[1],
        reverse=True
    )
    
    return sorted_words[:top_k]

# UI
st.title("🔮 Bigram Text Autocomplete Model")
st.write("Enter a word and see the predicted next words based on Twitter dataset bigram model.")

user_input = st.text_input("Enter text:")

if user_input.strip():
    last_word = user_input.split()[-1].lower()
    predictions = predict_next_word(last_word, top_k=5)

    st.subheader("Suggested next words:")
    for word, prob in predictions:
        st.write(f"➡ **{word}** (probability: {prob:.4f})")
