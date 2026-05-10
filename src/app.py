import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
import pickle

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.nlp_pipeline.lda_trainer import LDATrainer
from src.modeling.recommender import RecommenderEngine

# --- Page Config ---
st.set_page_config(page_title="School RecSys", page_icon="🎓", layout="wide")

# --- Constants ---
DATA_PATH = "data/processed/final_hybrid_dataset.pkl"
USERS_PATH = "data/processed/synthetic_users.pkl"
INTERACT_PATH = "data/processed/synthetic_interactions.pkl"
MODEL_DIR = "models"

# --- Load Data & Models ---
@st.cache_resource
def load_resources():
    if not os.path.exists(DATA_PATH):
        return None, None, None, None, None
    
    df = pd.read_pickle(DATA_PATH)
    
    # Pre-calculate city centers for easier location picking
    city_centers = df.groupby('city_location')[['latitude', 'longitude']].mean().to_dict('index')
    
    trainer = LDATrainer(n_topics=5)
    if os.path.exists(os.path.join(MODEL_DIR, "lda_model.pkl")):
        trainer.load_model(MODEL_DIR)
    
    # Load CF Data
    historical_users_df = None
    interactions = None
    if os.path.exists(USERS_PATH) and os.path.exists(INTERACT_PATH):
        historical_users_df = pd.read_pickle(USERS_PATH)
        with open(INTERACT_PATH, "rb") as f:
            interactions = pickle.load(f)
            
    return df, trainer, city_centers, historical_users_df, interactions

df, trainer, city_centers, historical_users_df, interactions = load_resources()

# --- Sidebar: User Inputs ---
st.sidebar.title("🔍 Search Preferences")

# Input Method Toggle
input_method = st.sidebar.radio("Search Method", ["Text Query", "Preference Sliders"])

if input_method == "Text Query":
    # 1. Improved Query Input (Drop-down + Text)
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
    
    theta_q = None
    if query and trainer:
        theta_q = trainer.transform(query)[0]
else:
    # 2. Explicit User Preference Sliders
    st.sidebar.subheader("Weighted Preferences")
    stem = st.sidebar.slider("STEM / Robotics", 0.0, 1.0, 0.2)
    arts = st.sidebar.slider("Arts / Music", 0.0, 1.0, 0.2)
    athletics = st.sidebar.slider("Athletics / Sports", 0.0, 1.0, 0.2)
    academics = st.sidebar.slider("Academic / AP", 0.0, 1.0, 0.2)
    cultural = st.sidebar.slider("Cultural / Language", 0.0, 1.0, 0.2)
    
    # Normalize
    weights = np.array([stem, arts, athletics, academics, cultural])
    total = weights.sum()
    if total > 0:
        theta_q = weights / total
    else:
        theta_q = np.full(5, 0.2)
    query = "Custom Weighted Profile"

st.sidebar.markdown("---")

# 3. Location Input
st.sidebar.subheader("📍 Location & Range")
if city_centers:
    all_cities = sorted(list(city_centers.keys()))
    default_city = "Los Angeles" if "Los Angeles" in all_cities else all_cities[0]
    selected_city = st.sidebar.selectbox("Select your City:", all_cities, index=all_cities.index(default_city))
    
    default_lat = city_centers[selected_city]['latitude']
    default_lon = city_centers[selected_city]['longitude']
    
    lat = st.sidebar.number_input("Latitude", value=float(default_lat), format="%.4f")
    lon = st.sidebar.number_input("Longitude", value=float(default_lon), format="%.4f")
else:
    lat = st.sidebar.number_input("Your Latitude", value=34.05, format="%.4f")
    lon = st.sidebar.number_input("Your Longitude", value=-118.24, format="%.4f")

radius_km = st.sidebar.slider("Search Radius (km)", 5, 100, 40)

st.sidebar.markdown("---")

# 4. Hard Filters
st.sidebar.subheader("Filters")
school_levels = ["All", "Elementary", "Middle", "High"]
selected_level = st.sidebar.selectbox("School Level", school_levels)

st.sidebar.markdown("---")
st.sidebar.subheader("Algorithm Tuning")
alpha = st.sidebar.slider("Content Match Weight (α)", 0.0, 1.0, 0.8)
beta = st.sidebar.slider("School Quality Weight (β)", 0.0, 1.0, 0.2)
gamma = st.sidebar.slider("Distance Penalty (γ)", 0.0, 1.0, 0.0)

# --- Main App ---
st.title("🎓 Hybrid School Recommender")

if df is None:
    st.error("Dataset not found! Please run `python run_full_pipeline.py` first.")
else:
    # 1. Apply Hard Filters First
    filtered_df = df.copy()
    
    # Radius Filter
    recommender = RecommenderEngine(alpha=alpha, beta=beta, gamma=gamma, k_target=30, lambda_max=0.8)
    filtered_df['dist_km'] = filtered_df.apply(lambda r: recommender.haversine(lat, lon, r['latitude'], r['longitude']), axis=1)
    filtered_df = filtered_df[filtered_df['dist_km'] <= radius_km]
    
    # Level Filter
    if selected_level != "All":
        level_val_map = {"Elementary": 1, "Middle": 2, "High": 3}
        target_val = level_val_map.get(selected_level)
        if target_val:
            filtered_df = filtered_df[filtered_df['school_level'] == target_val]

    if theta_q is not None:
        # 2. Process Recommendations
        with st.spinner("Analyzing and ranking schools..."):
            wu = theta_q
            
            # Get Recommendations
            raw_results = recommender.get_recommendations(
                user_query_theta = theta_q,
                user_loc = (lat, lon),
                candidate_df = filtered_df,
                wu = wu,
                historical_users_df = historical_users_df,
                interactions = interactions
            )
            
            # THE FIX: Merge back with filtered_df to get the missing metadata
            metadata_cols = [
                'ncessch', 'city_location', 'school_level', 'enrollment', 
                'theta_s', 'ap_count', 'total_athletes', 'student_teacher_ratio',
                'stem_score', 'arts_participants', 'world_lang'
            ]
            results = raw_results.merge(filtered_df[metadata_cols], on='ncessch', how='left')
            
            top_5 = results.head(5)

        # 3. LDA Insights
        if input_method == "Text Query" and trainer:
            best_topic_idx = np.argmax(theta_q)
            feature_names = trainer.vectorizer.get_feature_names_out()
            top_words = [feature_names[i] for i in trainer.lda.components_[best_topic_idx].argsort()[:-6:-1]]
            st.success(f"**LDA Interpretation:** Your query matches topics related to: *{', '.join(top_words)}*")

        # 4. Display Results
        st.subheader(f"Top 5 Matches within {radius_km}km")
        
        if top_5.empty:
            st.warning("No schools found matching your criteria. Try increasing the radius or relaxing filters.")
        else:
            cols = st.columns([1, 1])
            
            with cols[0]:
                for idx, row in top_5.iterrows():
                    # Format display
                    lam = row['lambda_u']
                    lam_pct = lam / 0.8 # lambda_max is 0.8
                    
                    with st.expander(f"🏫 {row['school_name']} (Final Score: {row['final_score']:.3f})"):
                        c1, c2, c3 = st.columns(3)
                        c1.metric("Distance", f"{row['distance']:.1f} km")
                        c2.metric("CF Confidence (λ_u)", f"{row['lambda_u']:.2f}")
                        c3.metric("Content Score", f"{row['content_utility']:.2f}")
                        
                        # Map Level ID to Text
                        level_names = {1: "Elementary", 2: "Middle", 3: "High", 4: "Other", 7: "K-12"}
                        level_text = level_names.get(row.get('school_level'), "N/A")
                        
                        st.write(f"📍 **{row.get('city_location', 'Unknown')}** | Level: **{level_text}**")
                        
                        # Progress bar for CF vs Content
                        st.write(f"**Hybrid Balance:**")
                        st.progress(lam_pct, text=f"CF Influence: {lam:.2f} / Content Influence: {1-lam:.2f}")

                        # --- NEW: Unique Strength/Weakness Analysis ---
                        st.markdown("---")
                        st.write("#### 🎯 Strengths & Specializations")
                        
                        if 'theta_s' in row and trainer:
                            theta_s = row['theta_s']
                            feature_names = trainer.vectorizer.get_feature_names_out()
                            
                            # Get top 2 topics for the school
                            top_topics = np.argsort(theta_s)[::-1][:2]
                            
                            s_cols = st.columns(2)
                            for i, t_idx in enumerate(top_topics):
                                if theta_s[t_idx] > 0.1: # Significant topic
                                    top_words = [feature_names[w_idx] for w_idx in trainer.lda.components_[t_idx].argsort()[:-4:-1]]
                                    strength_label = "Primary Focus" if i == 0 else "Secondary Focus"
                                    s_cols[i].write(f"**{strength_label}:**\n{', '.join(top_words).title()}")

                        # --- NEW: Hyper-Unique Narrative Profile ---
                        st.markdown("---")
                        st.write("#### 📝 School Profile")
                        
                        narrative = []
                        
                        # 1. Academic & AP Context (Granular)
                        ap_val = int(row.get('ap_count', 0))
                        if ap_val > 25:
                            narrative.append(f"**{row['school_name']}** stands as an elite academic institution with a massive catalog of {ap_val} AP subjects.")
                        elif ap_val > 10:
                            narrative.append(f"**{row['school_name']}** offers a competitive college-prep environment featuring {ap_val} distinct AP pathways.")
                        elif ap_val > 0:
                            narrative.append(f"Academic offerings at **{row['school_name']}** include {ap_val} specialized AP courses alongside core requirements.")
                        else:
                            narrative.append(f"**{row['school_name']}** prioritizes a foundational curriculum with focused individual student tracking.")

                        # 2. STEM & Technical (Relative)
                        stem_score = row.get('stem_score', 0)
                        if stem_score > 0.8:
                            narrative.append("The campus is recognized for its high-tech STEM labs and rigorous science initiatives.")
                        elif stem_score > 0.4:
                            narrative.append("Students here benefit from a balanced integration of technology and science in daily learning.")

                        # 3. Athletics & Engagement
                        athletes = int(row.get('total_athletes', 0))
                        enrollment = int(row.get('enrollment', 1))
                        participation_rate = (athletes / enrollment) if enrollment > 0 else 0
                        
                        if athletes > 200:
                            narrative.append(f"With {athletes} active athletes, the school's sports culture is a defining pillar of student life.")
                        elif participation_rate > 0.3:
                            narrative.append(f"A high percentage of the student body ({participation_rate:.0%}) is involved in the school's diverse athletic programs.")

                        # 4. Arts & Humanities
                        arts = row.get('arts_participants', 0)
                        if arts > 100:
                            narrative.append(f"The arts program is exceptionally large, with {arts} students engaged in creative and performing productions.")
                        elif arts > 20:
                            narrative.append("The school maintains active visual and performing arts electives for creative development.")

                        # 5. Support & Environment
                        st_ratio = row.get('student_teacher_ratio', 0)
                        if 0 < st_ratio < 12:
                            narrative.append(f"Instruction is highly personalized, facilitated by an intimate {st_ratio:.1f}:1 student-teacher ratio.")
                        elif st_ratio > 22:
                            narrative.append(f"The school manages a larger, more social environment with a {st_ratio:.1f}:1 ratio, focusing on collaborative learning.")

                        # Final output with a fallback that is still somewhat unique to the level
                        if not narrative:
                            level_map = {1: "elementary-focused", 2: "middle-school", 3: "high-school prep"}
                            narrative.append(f"This is a {level_map.get(row.get('school_level'), 'specialized')} campus serving the {row.get('city_location', 'local')} community.")

                        st.write(" ".join(narrative))

                        st.write("#### 📊 Key Metrics")
                        m1, m2, m3 = st.columns(3)
                        m1.metric("AP Courses", ap_val)
                        m2.metric("Athletics", athletes)
                        m3.metric("S/T Ratio", f"{st_ratio:.1f}:1")
            
            with cols[1]:
                # Map View
                import pydeck as pdk

                # Schools data
                map_data = top_5[['latitude', 'longitude', 'school_name', 'final_score']].copy()
                map_data['color'] = [[0, 0, 255, 160]] * len(map_data) # Blue for schools

                # User location
                user_data = pd.DataFrame({
                    'latitude': [lat],
                    'longitude': [lon],
                    'school_name': ['Your Location'],
                    'final_score': [0],
                    'color': [[255, 0, 0, 200]] # Red for you
                })

                combined_points = pd.concat([map_data, user_data])

                st.pydeck_chart(pdk.Deck(
                    map_style='mapbox://styles/mapbox/light-v9',
                    initial_view_state=pdk.ViewState(
                        latitude=lat,
                        longitude=lon,
                        zoom=10,
                        pitch=0,
                    ),
                    layers=[
                        pdk.Layer(
                            'ScatterplotLayer',
                            data=combined_points,
                            get_position='[longitude, latitude]',
                            get_color='color',
                            get_radius=300,
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
                    tooltip={"text": "{school_name}\nScore: {final_score}"}
                ))

    else:
        st.info("Enter a query or adjust sliders in the sidebar to get school recommendations.")
        
        # System Stats
        if city_centers:
            st.subheader("System Overview")
            st.write(f"Indexing **{len(df)}** schools across California.")
            st.write(f"Hybrid Engine loaded with **{len(historical_users_df) if historical_users_df is not None else 0}** historical user profiles.")
            
            city_samples = df[df.city_location == selected_city].head(3)
            st.write(f"Sample schools in **{selected_city}**:")
            st.table(city_samples[['school_name', 'school_level', 'enrollment']])
