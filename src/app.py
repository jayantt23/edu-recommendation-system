import streamlit as st
import pandas as pd
import numpy as np
import os
import sys

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.nlp_pipeline.lda_trainer import LDATrainer
from src.modeling.recommender import RecommenderEngine

# --- Page Config ---
st.set_page_config(page_title="School RecSys", page_icon="🎓", layout="wide")

# --- Constants ---
DATA_PATH = "data/processed/final_hybrid_dataset.pkl"
MODEL_DIR = "models"

# --- Load Data & Models ---
@st.cache_resource
def load_resources():
    if not os.path.exists(DATA_PATH):
        return None, None, None
    df = pd.read_pickle(DATA_PATH)
    
    # Pre-calculate city centers for easier location picking
    city_centers = df.groupby('city_location')[['latitude', 'longitude']].mean().to_dict('index')
    
    trainer = LDATrainer()
    if os.path.exists(os.path.join(MODEL_DIR, "lda_model.pkl")):
        trainer.load_model(MODEL_DIR)
    return df, trainer, city_centers

df, trainer, city_centers = load_resources()

# --- Sidebar: User Inputs ---
st.sidebar.title("🔍 Search Preferences")

# 1. Improved Query Input (Drop-down)
query_options = {
    "STEM and Robotics": "Rigorous STEM curriculum with advanced labs in robotics, coding, and biotechnology.",
    "Performing Arts & Music": "Vibrant arts program offering deep immersion in visual arts, classical music, and theater production.",
    "Competitive Athletics": "Focus on high participation rate in athletics, fostering leadership and teamwork through competitive sports.",
    "Academic Excellence & AP": "Personalized academic support and a wide range of advanced AP course offerings.",
    "Language & Cultural Studies": "Emphasis on cultural immersion and diverse language studies.",
    "Other (Custom Search)": ""
}

selected_label = st.sidebar.selectbox("What are you looking for?", options=list(query_options.keys()))

if selected_label == "Other (Custom Search)":
    query = st.sidebar.text_input("Enter your specific interests:", placeholder="e.g. Montessori education")
else:
    query = query_options[selected_label]

st.sidebar.markdown("---")

# 2. Improved Location Input (City Selector)
st.sidebar.subheader("📍 Where are you?")
if city_centers:
    all_cities = sorted(list(city_centers.keys()))
    # Default to a well-known city or first in list
    default_city = "Los Angeles" if "Los Angeles" in all_cities else all_cities[0]
    selected_city = st.sidebar.selectbox("Select your City (California):", all_cities, index=all_cities.index(default_city))
    
    # Allow fine-tuning
    default_lat = city_centers[selected_city]['latitude']
    default_lon = city_centers[selected_city]['longitude']
    
    lat = st.sidebar.number_input("Fine-tune Latitude", value=float(default_lat), format="%.4f")
    lon = st.sidebar.number_input("Fine-tune Longitude", value=float(default_lon), format="%.4f")
else:
    lat = st.sidebar.number_input("Your Latitude", value=34.49, format="%.4f")
    lon = st.sidebar.number_input("Your Longitude", value=-118.21, format="%.4f")

st.sidebar.markdown("---")
st.sidebar.subheader("Adjust Importance")
alpha = st.sidebar.slider("Curriculum Match", 0.0, 1.0, 0.5, help="Matches your selected interest above.")
beta = st.sidebar.slider("School Scale", 0.0, 1.0, 0.3, help="Prioritizes larger/smaller schools based on administrative metrics.")
gamma = st.sidebar.slider("Proximity Importance", 0.0, 1.0, 0.2, help="How much you care about the physical distance.")

# --- Main App ---
st.title("🎓 California School Recommender")
st.markdown(f"**Current Search:** *{selected_label}* in **{selected_city if city_centers else 'Selected Area'}**")

if df is None:
    st.error("Dataset not found! Please run `python src/main_pipeline.py` first.")
else:
    if query:
        # 1. Process Query
        with st.spinner("Analyzing and ranking schools..."):
            theta_q = trainer.transform(query)[0]
            recommender = RecommenderEngine(alpha=alpha, beta=beta, gamma=gamma)
            user_loc = (lat, lon)
            
            # 2. Get Recommendations
            results = recommender.get_recommendations(theta_q, user_loc, df)
            top_5 = results.head(5)

        # 3. Display Results
        st.subheader("Top Matches Near You")
        
        cols = st.columns([1, 1])
        
        with cols[0]:
            for idx, row in top_5.iterrows():
                with st.expander(f"🏫 {row['school_name']} (Score: {row['final_score']:.3f})"):
                    st.write(f"📍 **{row.get('city_location', 'Unknown')}**")
                    st.write(f"📏 **Distance:** {row['distance']:.2f} km")
                    
                    # Explain the components
                    sim_jsd = recommender.get_jsd_similarity(theta_q, row.get('theta_s', [0.2]*5))
                    st.progress(sim_jsd, text=f"Curriculum Similarity: {sim_jsd:.1%}")
                    
                    # Show brochure snippet if available
                    sid = str(row['ncessch']).zfill(12)
                    brochure_path = f"data/raw/brochures/{sid}.txt"
                    if os.path.exists(brochure_path):
                        with open(brochure_path, "r", encoding="utf-8") as f:
                            st.info(f"**School Profile:**\n\n{f.read()}")
        with cols[1]:
            # Improved Map View with Pydeck
            import pydeck as pdk

            # Schools data
            map_data = top_5[['latitude', 'longitude', 'school_name']].copy()
            map_data['color'] = [[0, 0, 255, 160]] * len(map_data) # Blue for schools

            # User location
            user_data = pd.DataFrame({
                'latitude': [lat],
                'longitude': [lon],
                'school_name': ['Your Location'],
                'color': [[255, 0, 0, 200]] # Red for you
            })

            combined_points = pd.concat([map_data, user_data])

            st.pydeck_chart(pdk.Deck(
                map_style='mapbox://styles/mapbox/light-v9',
                initial_view_state=pdk.ViewState(
                    latitude=lat,
                    longitude=lon,
                    zoom=11,
                    pitch=0,
                ),
                layers=[
                    pdk.Layer(
                        'ScatterplotLayer',
                        data=combined_points,
                        get_position='[longitude, latitude]',
                        get_color='color',
                        get_radius=200,
                        pickable=True,
                    ),
                    pdk.Layer(
                        'TextLayer',
                        data=combined_points,
                        get_position='[longitude, latitude]',
                        get_text='school_name',
                        get_size=16,
                        get_color=[0, 0, 0],
                        get_alignment_baseline="'bottom'",
                    )
                ],
                tooltip={"text": "{school_name}"}
            ))

    else:
        st.info("Select a category in the sidebar to begin searching for schools in California.")
        
        # System Stats
        st.subheader("Explore Our Data")
        st.write(f"We have indexed **{len(df)}** schools across **{len(all_cities)}** California cities.")
        
        # Display sample from the selected city
        if city_centers:
            city_samples = df[df.city_location == selected_city].head(3)
            st.write(f"Sample schools in **{selected_city}**:")
            st.table(city_samples[['school_name', 'school_level', 'enrollment']])
