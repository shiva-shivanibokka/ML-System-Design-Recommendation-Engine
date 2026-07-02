// Plain-language definitions surfaced by the "?" info popovers across the app.
// Written from the user's side of the screen: what it is and why it matters.

export const GLOSSARY: Record<string, { term: string; body: string }> = {
  neumf: {
    term: "NeuMF (Neural CF)",
    body: "A neural network that learns each user and movie as a vector and predicts how much they'll match. It captures non-linear taste patterns a linear model can't. Stronger, slower.",
  },
  svd: {
    term: "SVD (Matrix Factorization)",
    body: "A classic linear model that factorizes the user–movie rating matrix into embeddings. Scoring is a fast dot product. A solid, cheap baseline to beat.",
  },
  bandit: {
    term: "Thompson Sampling",
    body: "The A/B decision-maker. Each model has a Beta(α, β) belief over its click rate. Every request samples from those beliefs and routes to the model that looks best right now — so winning models get more traffic automatically, without a fixed 50/50 split.",
  },
  alphabeta: {
    term: "α / β",
    body: "The two counters behind each model's Beta distribution. A click adds 1 to α (a win), a no-click adds 1 to β (a loss). The more evidence, the tighter and more confident the belief curve becomes.",
  },
  faiss: {
    term: "FAISS retrieval",
    body: "Stage 1 of the pipeline. Instead of scoring all 3,533 movies, an approximate-nearest-neighbor index (IVF+PQ) grabs the ~500 closest candidates to the user in milliseconds. This is how real systems handle millions of items.",
  },
  mmr: {
    term: "MMR re-ranking",
    body: "Maximal Marginal Relevance. After ranking, it trades a little relevance for diversity so you don't get ten near-identical picks, and caps how many share one genre.",
  },
  coldstart: {
    term: "Cold start",
    body: "When a user (or item) has too little history to personalize. The system falls back to popularity and content-based picks until it learns enough.",
  },
  ctr: {
    term: "CTR (Click-Through Rate)",
    body: "Clicks ÷ recommendations shown. The core online signal for whether recommendations are actually good. Here it's simulated by the 'Register click' buttons.",
  },
  coverage: {
    term: "Catalog coverage",
    body: "The share of the whole catalog that got recommended recently. Low coverage means the system keeps pushing the same few popular titles (a filter bubble) and ignoring the long tail.",
  },
  psi: {
    term: "PSI (Population Stability Index)",
    body: "Measures how much the distribution of model scores has drifted vs a baseline. Rising PSI is an early warning that the world changed and the model may be going stale. Alert at 0.2.",
  },
  score: {
    term: "Relevance score",
    body: "The model's predicted match between this user and this movie. Higher means more confident. NeuMF and SVD produce them on different scales, so compare within a single result set.",
  },
  fresh: {
    term: "Trending / fresh",
    body: "The item got a freshness boost in re-ranking because it's recently popular — a light recency signal layered on top of pure relevance.",
  },
  registerclick: {
    term: "Register click",
    body: "Simulates you clicking this movie. It sends a reward to the bandit for whichever model produced this recommendation — so clicking teaches the system which model to trust. Watch the Bandit A/B page update.",
  },
  embedding: {
    term: "Embedding galaxy",
    body: "Every movie is a 64-dimensional vector learned by NeuMF. This is a 2D t-SNE projection of those vectors — movies that sit near each other are ones the model considers similar. Retrieval literally searches this space.",
  },
  latency: {
    term: "Pipeline latency",
    body: "Time spent in each of the 5 stages for the last request, versus its budget. The whole pipeline targets a p99 under 100ms — the kind of SLA a real recommendation API runs to.",
  },
  ndcg: {
    term: "NDCG@10",
    body: "Normalized Discounted Cumulative Gain at 10. Rewards putting the right items near the top of the list, not just somewhere in it. 1.0 is perfect ranking.",
  },
  hr: {
    term: "HR@10 (Hit Rate)",
    body: "How often the held-out item the user actually engaged with appears in the top 10. A simple, intuitive ranking-quality measure.",
  },
  trafficsplit: {
    term: "Traffic split",
    body: "The share of requests the bandit is currently routing to each model. It shifts on its own as evidence accumulates — no manual dial.",
  },
};

export type GlossaryKey = keyof typeof GLOSSARY;
