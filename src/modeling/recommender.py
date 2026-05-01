import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from math import radians, cos, sin, asin, sqrt

class RecommenderEngine:
    def __init__(self, alpha=0.5, beta=0.3, gamma=0.2, k_target=5, lambda_max=0.8):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.k_target = k_target
        self.lambda_max = lambda_max

    def haversine(self, lat1, lon1, lat2, lon2):
        """Calculate the great circle distance between two points."""
        lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * asin(sqrt(a))
        r = 6371  # Radius of earth in kilometers
        return c * r

    def get_jsd_similarity(self, p, q):
        """Jensen-Shannon Similarity = 1 - JSD."""
        # Avoid zeros for KL divergence
        p = np.array(p) + 1e-10
        q = np.array(q) + 1e-10
        # normalize
        p /= p.sum()
        q /= q.sum()
        # jensenshannon in scipy returns sqrt(JSD)
        js_dist = jensenshannon(p, q)
        return 1 - (js_dist**2)

    def calculate_content_utility(self, theta_q, theta_s, score_met, dist_geo):
        """U(q, s) = alpha * (1-JSD) + beta * Score_met - gamma * Pen_geo."""
        sim_jsd = self.get_jsd_similarity(theta_q, theta_s)
        
        # Scale distance to 0-1 penalty (100km max range for normalization example)
        # In practice, you'd want a more robust decay function
        pen_geo = min(dist_geo / 100.0, 1.0) 
        
        return self.alpha * sim_jsd + self.beta * score_met - self.gamma * pen_geo

    def calculate_cf_score(self, sim_uv_list, interactions_list):
        """S_CF(u, s) = weighted average of historical selections."""
        if not sim_uv_list:
            return 0
        
        numerator = sum(sim * inter for sim, inter in zip(sim_uv_list, interactions_list))
        denominator = sum(sim_uv_list)
        
        return numerator / denominator if denominator > 0 else 0

    def calculate_confidence_factor(self, n_neighbors):
        """Adaptive Confidence Factor lambda_u."""
        return self.lambda_max * min(1.0, n_neighbors / self.k_target)

    def get_recommendations(self, user_query_theta, user_loc, candidate_df, wu=None, historical_users_df=None, interactions=None):
        """
        candidate_df: DataFrame with ['ncessch', 'latitude', 'longitude', 'theta_s', 'norm_enrollment']
        wu: User's preference weight vector (e.g. [stem, arts, athletics, academic, cultural])
        historical_users_df: DataFrame of previous users with their preference vectors
        interactions: Dict mapping user_id to set of selected ncessch_ids
        """
        results = []
        
        # 1. Collaborative Filtering Prep
        n_neighbors = 0
        s_cf_dict = {}
        lambda_u = 0
        
        if wu is not None and historical_users_df is not None and interactions is not None:
            # sim(u, v) using cosine similarity as per Eq (3)
            similarities = []
            for _, v_row in historical_users_df.iterrows():
                wv = v_row['wv']
                sim = np.dot(wu, wv) / (np.linalg.norm(wu) * np.linalg.norm(wv) + 1e-10)
                if sim > 0.7: # Tau threshold as per Alg 2 step 6 — raised to 0.7 for meaningful neighbours
                    similarities.append((v_row['user_id'], sim))
            
            n_neighbors = len(similarities)
            lambda_u = self.calculate_confidence_factor(n_neighbors)
            
            if n_neighbors > 0:
                # Calculate S_CF(u, s) for each school that neighbors interacted with
                for sid in candidate_df['ncessch']:
                    sim_sum = 0
                    weighted_sum = 0
                    for uid, sim in similarities:
                        interacted = 1 if sid in interactions.get(uid, []) else 0
                        weighted_sum += sim * interacted
                        sim_sum += sim
                    s_cf_dict[sid] = weighted_sum / sim_sum if sim_sum > 0 else 0

        # 2. Hybrid Scoring
        for idx, row in candidate_df.iterrows():
            sid = row['ncessch']
            dist = self.haversine(user_loc[0], user_loc[1], row['latitude'], row['longitude'])
            u_qs = self.calculate_content_utility(
                user_query_theta, 
                row['theta_s'], 
                row.get('score_met', row.get('norm_enrollment', 0.5)),
                dist
            )
            
            s_cf = s_cf_dict.get(sid, 0)
            
            final_score = (1 - lambda_u) * u_qs + lambda_u * s_cf
            
            results.append({
                'ncessch': sid,
                'school_name': row.get('school_name', 'Unknown'),
                'final_score': final_score,
                'content_utility': u_qs,
                'cf_score': s_cf,
                'distance': dist,
                'lambda_u': lambda_u,
                'norm_enrollment': row.get('norm_enrollment', 0),
                'latitude': row.get('latitude'),
                'longitude': row.get('longitude')
            })
            
        return pd.DataFrame(results).sort_values(by='final_score', ascending=False)

if __name__ == "__main__":
    # Small test
    recommender = RecommenderEngine()
    theta_q = [0.8, 0.1, 0.1]
    theta_s = [0.7, 0.2, 0.1]
    print(f"Content Utility Test: {recommender.calculate_content_utility(theta_q, theta_s, 0.9, 10)}")
