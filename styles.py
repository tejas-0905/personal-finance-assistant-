def load_css():
    return """
<style>

/* ------------------ GLOBAL LAYOUT ------------------ */

html, body, [data-testid="stAppViewContainer"] {
    background: radial-gradient(circle at top left, #101827 0%, #020617 50%, #020617 100%) !important;
    color: #e5e7eb !important;
    font-family: "Inter", system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}

/* Main app container */
[data-testid="stAppViewContainer"] > .main {
    padding-top: 1.5rem;
    padding-left: 2.5rem;
    padding-right: 2.5rem;
}

/* Sidebar styling */
section[data-testid="stSidebar"] {
    background: linear-gradient(160deg, rgba(15,23,42,0.98), rgba(15,23,42,0.96)) !important;
    border-right: 1px solid rgba(148,163,184,0.35);
    box-shadow: 0 0 40px rgba(15,23,42,0.9);
}

section[data-testid="stSidebar"] * {
    color: #e5e7eb !important;
}

/* Scrollbar */
::-webkit-scrollbar {
    width: 8px;
}
::-webkit-scrollbar-track {
    background: transparent;
}
::-webkit-scrollbar-thumb {
    background: linear-gradient(180deg, #6366f1, #a855f7);
    border-radius: 999px;
}

/* ------------------ TITLES & TEXT ------------------ */

h1, h2, h3 {
    font-weight: 700 !important;
    letter-spacing: 0.02em;
}

h1 {
    font-size: 2.1rem !important;
    background: linear-gradient(90deg, #a855f7, #22d3ee);
    -webkit-background-clip: text;
    color: transparent !important;
}

h2 {
    color: #e5e7eb !important;
    margin-top: 1.5rem !important;
}

h3 {
    color: #cbd5f5 !important;
}

/* Sub headings / labels */
[data-testid="stMarkdown"] p {
    color: #e5e7eb;
}

/* ------------------ GLASS CARDS ------------------ */

/* Generic glass container for cards, expanders, etc. */
.block-container, .stTabs, .stDataFrame, .element-container {
    transition: all 300ms ease;
}

/* Streamlit metric cards */
[data-testid="stMetric"] {
    background: radial-gradient(circle at top left,
        rgba(148,163,253,0.12),
        rgba(15,23,42,0.96)
    );
    border-radius: 18px;
    padding: 1.2rem 1rem;
    margin: 0.4rem;
    border: 1px solid rgba(148,163,253,0.45);
    box-shadow:
        0 18px 45px rgba(15,23,42,0.95),
        0 0 0 0 rgba(129,140,248,0.0);
    backdrop-filter: blur(18px);
    -webkit-backdrop-filter: blur(18px);
    transition: transform 220ms ease, box-shadow 220ms ease, border-color 220ms ease;
}

[data-testid="stMetric"]:hover {
    transform: translateY(-4px) scale(1.01);
    box-shadow:
        0 24px 65px rgba(15,23,42,0.95),
        0 0 0 1px rgba(129,140,248,0.25);
    border-color: rgba(129,140,248,0.9);
}

/* Metric label & value */
[data-testid="stMetric"] > div:nth-child(1) {
    color: #9ca3af !important;
    font-size: 0.85rem;
    text-transform: uppercase;
    letter-spacing: 0.15em;
}

[data-testid="stMetric"] > div:nth-child(2) {
    color: #e5e7eb !important;
    font-size: 1.4rem;
    font-weight: 700;
}

/* ------------------ CUSTOM RECOMMENDATION CARDS ------------------ */

.custom-card {
    background: linear-gradient(135deg,
        rgba(15,23,42,0.95),
        rgba(15,23,42,0.90)
    );
    border-radius: 18px;
    padding: 0.9rem 1rem;
    margin-bottom: 0.6rem;
    border: 1px solid rgba(148,163,253,0.45);
    box-shadow: 0 18px 40px rgba(15,23,42,0.95);
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    color: #e5e7eb;
    font-size: 0.95rem;
    position: relative;
    overflow: hidden;
    transition: transform 230ms ease, box-shadow 230ms ease, border-color 230ms ease;
}

/* Animated gradient border glow using pseudo-element */
.custom-card::before {
    content: "";
    position: absolute;
    inset: -40%;
    background: conic-gradient(
        from 90deg,
        #6366f1,
        #22d3ee,
        #a855f7,
        #6366f1
    );
    opacity: 0;
    z-index: -1;
    animation: spin-border 10s linear infinite;
}

/* Inner mask to create subtle border glow */
.custom-card::after {
    content: "";
    position: absolute;
    inset: 1px;
    background: radial-gradient(circle at top left,
        rgba(15,23,42,0.96),
        rgba(15,23,42,0.98)
    );
    border-radius: inherit;
    z-index: -1;
}

.custom-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 24px 60px rgba(15,23,42,0.98);
    border-color: rgba(129,140,248,0.9);
}

.custom-card:hover::before {
    opacity: 0.9;
}

/* Priority-based recommendation cards */
.priority-high {
    background: linear-gradient(135deg,
        rgba(239,68,68,0.15),
        rgba(220,38,38,0.10)
    );
    border: 1px solid rgba(239,68,68,0.6);
    box-shadow: 0 18px 40px rgba(239,68,68,0.2);
}

.priority-high::before {
    background: conic-gradient(
        from 90deg,
        #ef4444,
        #f87171,
        #fca5a5,
        #ef4444
    );
}

.priority-high:hover {
    border-color: rgba(239,68,68,0.9);
    box-shadow: 0 24px 60px rgba(239,68,68,0.3);
}

.priority-medium {
    background: linear-gradient(135deg,
        rgba(245,158,11,0.15),
        rgba(217,119,6,0.10)
    );
    border: 1px solid rgba(245,158,11,0.6);
    box-shadow: 0 18px 40px rgba(245,158,11,0.2);
}

.priority-medium::before {
    background: conic-gradient(
        from 90deg,
        #f59e0b,
        #fbbf24,
        #fcd34d,
        #f59e0b
    );
}

.priority-medium:hover {
    border-color: rgba(245,158,11,0.9);
    box-shadow: 0 24px 60px rgba(245,158,11,0.3);
}

.priority-low {
    background: linear-gradient(135deg,
        rgba(34,197,94,0.15),
        rgba(22,163,74,0.10)
    );
    border: 1px solid rgba(34,197,94,0.6);
    box-shadow: 0 18px 40px rgba(34,197,94,0.2);
}

.priority-low::before {
    background: conic-gradient(
        from 90deg,
        #22c55e,
        #4ade80,
        #86efac,
        #22c55e
    );
}

.priority-low:hover {
    border-color: rgba(34,197,94,0.9);
    box-shadow: 0 24px 60px rgba(34,197,94,0.3);
}

/* ------------------ EXPANDERS & DATA TABLES ------------------ */

details {
    border-radius: 18px !important;
    border: 1px solid rgba(148,163,253,0.45) !important;
    background: linear-gradient(145deg,
        rgba(15,23,42,0.97),
        rgba(15,23,42,0.93)
    ) !important;
    box-shadow: 0 18px 40px rgba(15,23,42,0.95);
    backdrop-filter: blur(18px);
    -webkit-backdrop-filter: blur(18px);
}

details > summary {
    color: #e5e7eb !important;
    font-weight: 600;
}

/* Dataframe container */
[data-testid="stDataFrame"] {
    border-radius: 18px;
    overflow: hidden;
    border: 1px solid rgba(148,163,253,0.45);
    box-shadow: 0 18px 40px rgba(15,23,42,0.95);
}

/* ------------------ INPUTS & WIDGETS ------------------ */

input, textarea, select {
    background: rgba(15,23,42,0.95) !important;
    border-radius: 12px !important;
    border: 1px solid rgba(148,163,253,0.45) !important;
    color: #e5e7eb !important;
    box-shadow: 0 0 0 0 rgba(129,140,248,0.0);
    transition: border-color 200ms ease, box-shadow 200ms ease, transform 150ms ease;
}

input:focus, textarea:focus, select:focus {
    outline: none !important;
    border-color: rgba(129,140,248,0.95) !important;
    box-shadow: 0 0 0 1px rgba(129,140,248,0.5);
    transform: translateY(-1px);
}

/* File uploader */
[data-testid="stFileUploader"] section {
    border-radius: 16px !important;
    border: 1px dashed rgba(148,163,253,0.7) !important;
    background: radial-gradient(circle at top left,
        rgba(30,64,175,0.3),
        rgba(15,23,42,0.95)
    ) !important;
}

/* Checkboxes & sliders */
[data-baseweb="checkbox"] > div {
    background: rgba(15,23,42,0.95) !important;
    border-radius: 10px !important;
    border: 1px solid rgba(148,163,253,0.5) !important;
}

/* Slider track */
[data-baseweb="slider"] > div > div {
    background: linear-gradient(90deg, #6366f1, #a855f7) !important;
}

/* ------------------ BUTTONS ------------------ */

button[kind="primary"], .stButton button {
    background: linear-gradient(135deg, #6366f1, #a855f7) !important;
    color: #f9fafb !important;
    border-radius: 999px !important;
    border: none !important;
    padding: 0.55rem 1.3rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.04em;
    box-shadow: 0 14px 35px rgba(79,70,229,0.55);
    transition: transform 150ms ease, box-shadow 150ms ease, filter 150ms ease;
}

button[kind="primary"]:hover, .stButton button:hover {
    transform: translateY(-1.5px) scale(1.02);
    box-shadow: 0 20px 45px rgba(79,70,229,0.85);
    filter: brightness(1.05);
}

button[kind="primary"]:active, .stButton button:active {
    transform: translateY(0px) scale(0.99);
    box-shadow: 0 10px 25px rgba(79,70,229,0.6);
}

/* ------------------ CHART CONTAINERS ------------------ */

[data-testid="stPlotlyChart"] {
    background: radial-gradient(circle at top left,
        rgba(15,23,42,0.98),
        rgba(15,23,42,0.94)
    );
    border-radius: 20px;
    padding: 0.75rem;
    border: 1px solid rgba(148,163,253,0.5);
    box-shadow: 0 20px 55px rgba(15,23,42,0.98);
    backdrop-filter: blur(18px);
    -webkit-backdrop-filter: blur(18px);
    transition: transform 230ms ease, box-shadow 230ms ease, border-color 230ms ease;
}

[data-testid="stPlotlyChart"]:hover {
    transform: translateY(-4px);
    box-shadow: 0 26px 75px rgba(15,23,42,1);
    border-color: rgba(129,140,248,0.9);
}

/* ------------------ ANIMATIONS ------------------ */

/* Subtle page fade-in */
@keyframes fade-in-up {
    from {
        opacity: 0;
        transform: translateY(10px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

/* Animated gradient border around custom cards */
@keyframes spin-border {
    to {
        transform: rotate(360deg);
    }
}

/* Apply fade-in to main content blocks */
.block-container > * {
    animation: fade-in-up 420ms ease-out;
}

/* ------------------ BADGES / ALERTS ------------------ */

/* Success / warning messages */
div.stAlert {
    border-radius: 16px !important;
    border-width: 1px !important;
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
}

/* ------------------ MOBILE TWEAKS ------------------ */

@media (max-width: 768px) {
    [data-testid="stAppViewContainer"] > .main {
        padding-left: 1rem;
        padding-right: 1rem;
    }

    h1 {
        font-size: 1.6rem !important;
    }

    [data-testid="stMetric"] {
        margin: 0.25rem 0;
    }
}

</style>
"""