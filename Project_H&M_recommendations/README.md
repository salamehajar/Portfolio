# Advanced-ML
H&amp;M recommendations

Objectif & Contexte: 
- Plateforme : Kaggle Competition.
- Domaine : Retail / Fast Fashion (H&M).
- But : Predict the next 12 articles each customer is likely to purchase based on historical shopping behavior.

Dataset & Scale
- Transactions : (~2 years, > 30M lines).
- Customer Metadata : Age, Newsletter engagement, Postal code...etc
- Product Metadata : Type (Robe/Pantalon), color, Description, Image (ID)Évaluation..etc

Why is the challenge useful ? 
- Recommendation Systems: Moving beyond traditional classification to personalized ranking problems
- Sequential Modeling: Understanding how customer preferences evolve over time
- Real-world Constraints: Balancing model sophistication with computational limitations
- Cold-start Problems: Handling new customers and seasonal products with limited interaction history
  
Problems and challenges encountered: 
- Data Volume & Computational Constraints
    - RAM limitations: Unable to load the full 30M+ transaction dataset into memory
    - GPU constraints: Limited access on Collab
    - Image metadata: Could load the images data, it was unused during the challenge.


The best technical solution :

We achieved best results by combining ALS and lightgbm, which offered:

- Efficiency: ALS handles the heavy lifting of candidate retrieval; LightGBM works on small, focused sets
- Complementarity:
  ALS captures collaborative patterns (implicit similarities)
  LightGBM adds explicit personalization (customer/product attributes)
- Metadata integration: Overcomes ALS's inability to use product/customer features
- Scalability: Can't run LightGBM on all 100K articles per customer, but can on top 100-500 candidates

Implementation Details:

The hybrid approach is implemented as a two-stage pipeline designed for efficiency and scalability on large-scale transactional data.

1. Candidate Generation (Recall Stage):

  - A time-weighted implicit ALS model (implicit.als.AlternatingLeastSquares) is trained on the user–item interaction matrix.
  - Each transaction is weighted using exponential time decay, giving more importance to recent purchases and capturing fast-changing fashion trends.
  - For every user, the model generates a cached list of top-N candidate items (typically top-50).
  - To increase recall and robustness, ALS candidates are enriched with:
      -Globally popular items (top sellers in the training window)
      -Repurchase candidates (items previously bought by the user)
  This results in approximately 50 candidates per user, reducing the ranking search space by several orders of magnitude.

2. Feature Construction:

For each (user, candidate item) pair, a feature vector is built using:
  - Candidate-level features: ALS score , Popular-item flag , Repurchase flag
  - User features: Age group , Club membership status , Newsletter engagement , Purchase recency and frequency, Spending and diversity statistics
  - Item features: Product type and group, Color group and index group, Historical sales volume, Short-term trend indicators (1–4 weeks),Repurchase rate
  - User–item interaction features: Number of past purchases, Recency in days, Average purchase price ,Category and color affinity

All interaction features are computed only for candidate items, ensuring strict memory control.

3-Ranking Model:
- **Model:** LightGBM (binary classification)  
- **Input:** ALS candidate items + engineered features  
- **Output:** Final relevance score used to rank candidates  
- **Training strategy:**  
  - One training instance per *(user, candidate item)* pair  
  - Binary labels indicating whether the item was purchased in the prediction window  

4-Evaluation and Metrics:
- **Metric:** **MAP@12 (Mean Average Precision at 12)** – official Kaggle evaluation metric  
  - Measures the quality of the ranked recommendation list **up to 12 items per user**  
  - **How it works:**  
    - Computes **average precision** for each user based on whether purchased items appear in the top-12 predictions  
    - Then averages over all users  
  - **Range:** 0 (worst, no relevant items in top-12) → 1 (best, all relevant items correctly ranked at the top)  

- **Validation strategy:**  
  - Time-based train / validation split  
  - Labels defined as items purchased in the following week  
  - One ranking group per customer  

- **Offline Results (Validation Set):**
  - **ALS only:** MAP@12 ≈ **0.0175**  
  - **ALS + LightGBM (baseline features):** MAP@12 ≈ **0.057**  
  - **ALS + LightGBM (enhanced features):** MAP@12 ≈ **0.069**  


Results:
  - The ALS-only baseline provides strong recall but limited personalization.
  - The hybrid ALS + LightGBM approach consistently improves ranking quality, achieving a significantly higher MAP@12.

Feature importance analysis confirms:
  -ALS score is the strongest signal
  -Interaction recency and trend-based item features contribute substantially
  -User metadata stabilizes predictions for sparse and cold-start users
  
Why SASRec Underperformed?
We initially tried SASRec (Self-Attentive Sequential Recommendation), but encountered limitations:

- Metadata blindness: SASRec focuses purely on item sequences, ignoring rich product attributes (color, type, price)
- Temporal granularity: Treats purchases as ordered sequences but doesn't capture real calendar time (seasonal trends, promotions)
- Cold-start weakness: Struggles with new products that lack sequential interaction history


.


