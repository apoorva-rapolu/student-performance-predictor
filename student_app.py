import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── CSS ───────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600&family=Playfair+Display:wght@700&display=swap');

html, body, [class*="css"] { font-family: 'Outfit', sans-serif; color: #e2e8f0; }
.main { background: #0f172a; }
.block-container { padding: 2rem 3rem; max-width: 1150px; }

.hero {
    background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
    border: 1px solid #334155;
    border-radius: 20px;
    padding: 2.8rem 3rem;
    color: white;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.hero::after {
    content: "🎓";
    position: absolute;
    right: 3rem; top: 50%;
    transform: translateY(-50%);
    font-size: 7rem;
    opacity: 0.12;
}
.hero h1 {
    font-family: 'Playfair Display', serif !important;
    color: #4ade80 !important;
    font-size: 2.6rem;
    margin: 0 0 0.4rem 0;
}
.hero p { color: #94a3b8; margin: 0; font-size: 1rem; font-weight: 300; }

.section-label {
    font-size: 0.8rem;
    font-weight: 600;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #00d2ff;
    margin: 1.5rem 0 0.8rem 0;
}

div[data-testid="stNumberInput"] label,
div[data-testid="stSelectbox"] label,
div[data-testid="stSlider"] label {
    font-size: 0.85rem !important;
    font-weight: 500 !important;
    color: white !important;
}

div[data-testid="stNumberInput"] input,
div[data-testid="stSelectbox"] > div > div {
    border: 1.5px solid #475569 !important;
    border-radius: 8px !important;
    background: white !important;
    color: black !important;
}

div[data-testid="stSelectbox"] > div * {
    color: black !important;
}

.predict-btn > button {
    background: #00d2ff !important;
    color: #0f172a !important;
    border: none !important;
    border-radius: 12px !important;
    font-family: 'Outfit', sans-serif !important;
    font-size: 1rem !important;
    font-weight: 600 !important;
    width: 100% !important;
    padding: 0.75rem !important;
    transition: 0.2s;
}
.predict-btn > button:hover { background: #00b8e6 !important; transform: translateY(-2px); }

.result-card {
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
    margin-top: 1rem;
    height: 100%;
}
.grade-A { background: rgba(16,185,129,0.1); border: 2px solid #10b981; color:#fff; }
.grade-B { background: rgba(59,130,246,0.1); border: 2px solid #3b82f6; color:#fff; }
.grade-C { background: rgba(245,158,11,0.1); border: 2px solid #f59e0b; color:#fff; }
.grade-D { background: rgba(239,68,68,0.1); border: 2px solid #ef4444; color:#fff; }

.grade-badge {
    font-family: 'Playfair Display', serif;
    font-size: 4rem;
    line-height: 1;
}
.result-title { font-size: 1.3rem; font-weight: 600; margin: 0.5rem 0 0.2rem; }
.result-sub { font-size: 0.85rem; color: #cbd5e1; }

.score-row {
    display: flex; gap: 0.8rem; height: 100%;
}
.score-box {
    flex: 1; background: #1e293b;
    border: 1px solid #334155;
    border-radius: 10px; padding: 1.2rem 0.8rem;
    text-align: center;
    display: flex; flex-direction: column; justify-content: center;
}
.score-box .val {
    font-family: 'Playfair Display', serif;
    font-size: 1.8rem; color: #00d2ff;
}
.score-box .lbl { font-size: 0.75rem; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.08em; margin-top: 0.4rem; }

.model-compare {
    background: #1e293b; border: 1px solid #334155;
    border-radius: 12px; padding: 1.2rem; margin-top: 1.5rem;
    font-size: 0.82rem;
    color: #e2e8f0;
}
.model-compare table { width: 100%; border-collapse: collapse; }
.model-compare td, .model-compare th {
    padding: 0.6rem; text-align: left;
    border-bottom: 1px solid #334155;
    font-size: 0.85rem;
}
.model-compare th { color: #00d2ff; font-weight: 600; }
.best-row td { background: rgba(0, 210, 255, 0.05); color: #00d2ff; font-weight: 600; }

.insight-box {
    background: #1e293b; border-left: 3px solid #00d2ff;
    border-radius: 0 8px 8px 0; padding: 1.2rem;
    margin-top: 1rem; font-size: 0.9rem; color: #e2e8f0; line-height: 1.6;
}

.empty-state {
    background: #1e293b; border: 1.5px dashed #475569;
    border-radius: 16px; padding: 4rem; text-align: center; margin-top: 1rem;
}
</style>
""", unsafe_allow_html=True)

# ── Data & Model ──────────────────────────────────────────────
@st.cache_resource
def load_model():
    """Load dataset from UCI / generate synthetic fallback, train 4 models."""
    try:
        url = "https://raw.githubusercontent.com/dsrscientist/dataset1/master/StudentsPerformance.csv"
        df = pd.read_csv(url)
    except Exception:
        # Synthetic fallback matching dataset distributions
        np.random.seed(42)
        n = 1000
        gender = np.random.choice(['male','female'], n)
        race = np.random.choice(['group A','group B','group C','group D','group E'], n)
        parental = np.random.choice([
            "some high school","high school","some college",
            "associate's degree","bachelor's degree","master's degree"], n)
        lunch = np.random.choice(['standard','free/reduced'], n)
        test_prep = np.random.choice(['none','completed'], n)
        math = np.clip(np.random.normal(66, 15, n), 0, 100).astype(int)
        reading = np.clip(np.random.normal(69, 14, n), 0, 100).astype(int)
        writing = np.clip(np.random.normal(68, 15, n), 0, 100).astype(int)
        df = pd.DataFrame({
            'gender': gender, 'race/ethnicity': race,
            'parental level of education': parental,
            'lunch': lunch, 'test preparation course': test_prep,
            'math score': math, 'reading score': reading, 'writing score': writing
        })

    # Feature engineering
    df['average score'] = (df['math score'] + df['reading score'] + df['writing score']) / 3
    df['grade'] = pd.cut(df['average score'],
                         bins=[0, 50, 60, 70, 80, 101],
                         labels=['F', 'D', 'C', 'B', 'A'])
    df.dropna(subset=['grade'], inplace=True)

    # Encode categoricals
    le_dict = {}
    cat_cols = ['gender', 'race/ethnicity', 'parental level of education',
                'lunch', 'test preparation course']
    for col in cat_cols:
        le = LabelEncoder()
        df[col + '_enc'] = le.fit_transform(df[col])
        le_dict[col] = le

    features = [c + '_enc' for c in cat_cols] + ['math score', 'reading score', 'writing score']
    X = df[features].values
    y = LabelEncoder().fit_transform(df['grade'])  # A=0,B=1,C=2,D=3,F=4 (alphabetical)
    grade_le = LabelEncoder().fit(df['grade'])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    models = {
        'Logistic Regression': LogisticRegression(max_iter=500, random_state=42),
        'Decision Tree': DecisionTreeClassifier(max_depth=8, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
    }

    trained = {}
    accuracies = {}
    for name, m in models.items():
        m.fit(X_train_sc, y_train)
        acc = accuracy_score(y_test, m.predict(X_test_sc))
        trained[name] = m
        accuracies[name] = acc

    best_name = max(accuracies, key=accuracies.get)
    return trained, scaler, le_dict, accuracies, best_name, grade_le

trained_models, scaler, le_dict, accuracies, best_model_name, grade_le = load_model()

GRADE_META = {
    'A': {'class': 'grade-A', 'emoji': '🏆', 'label': 'Excellent', 'msg': 'Outstanding performance across all subjects.'},
    'B': {'class': 'grade-B', 'emoji': '⭐', 'label': 'Good', 'msg': 'Strong performance with room to push further.'},
    'C': {'class': 'grade-C', 'emoji': '📘', 'label': 'Average', 'msg': 'Meets expectations — focus on weak subjects.'},
    'D': {'class': 'grade-D', 'emoji': '⚠️', 'label': 'Below Average', 'msg': 'Needs improvement — consider extra support.'},
    'F': {'class': 'grade-D', 'emoji': '❌', 'label': 'Fail', 'msg': 'Significant intervention recommended.'},
}

# ── UI ────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <h1>Student Performance Predictor</h1>
    <p>Random Forest · Gradient Boosting · Logistic Regression · Decision Tree · Students Performance in Exams Dataset</p>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="section-label">👤 Student Details & Exam Scores</div>', unsafe_allow_html=True)

# ROW 1
c1, c2, c3, c4 = st.columns(4)
gender = c1.selectbox("Gender", ["female", "male"])
race = c2.selectbox("Race / Ethnicity", ["group A", "group B", "group C", "group D", "group E"])
parental_edu = c3.selectbox("Parental Education", [
    "some high school", "high school", "some college",
    "associate's degree", "bachelor's degree", "master's degree"
])
lunch = c4.selectbox("Lunch Type", ["standard", "free/reduced"])

# ROW 2
st.markdown('<div style="margin-top:0.5rem;"></div>', unsafe_allow_html=True)
c5, c6, c7, c8 = st.columns(4)
test_prep = c5.selectbox("Test Preparation", ["none", "completed"])
math_score = c6.slider("Math Score", 0, 100, 65)
reading_score = c7.slider("Reading Score", 0, 100, 70)
writing_score = c8.slider("Writing Score", 0, 100, 68)

st.markdown('<div class="section-label">⚙️ Configuration & Prediction</div>', unsafe_allow_html=True)
# ROW 3
c9, c10 = st.columns([3, 1])
model_choice = c9.selectbox("Choose Prediction Model", list(trained_models.keys()),
                             index=list(trained_models.keys()).index(best_model_name))

with c10:
    st.markdown("<div style='margin-top: 1.7rem;'></div>", unsafe_allow_html=True)
    st.markdown('<div class="predict-btn">', unsafe_allow_html=True)
    predict = st.button("🎓  Predict Grade", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("<hr style='border: 1px solid #334155; margin: 2rem 0;'>", unsafe_allow_html=True)

if predict:
    # Encode inputs
    def safe_encode(col, val):
        le = le_dict[col]
        if val in le.classes_:
            return le.transform([val])[0]
        return 0

    features_input = np.array([[
        safe_encode('gender', gender),
        safe_encode('race/ethnicity', race),
        safe_encode('parental level of education', parental_edu),
        safe_encode('lunch', lunch),
        safe_encode('test preparation course', test_prep),
        math_score, reading_score, writing_score
    ]])

    features_sc = scaler.transform(features_input)
    model = trained_models[model_choice]
    pred_enc = model.predict(features_sc)[0]

    # Map encoded prediction back to grade letter
    grade_map = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'F'}
    grade = grade_map.get(pred_enc, 'C')

    avg = (math_score + reading_score + writing_score) / 3
    meta = GRADE_META[grade]

    # Insights
    scores = {'Math': math_score, 'Reading': reading_score, 'Writing': writing_score}
    weakest = min(scores, key=scores.get)
    strongest = max(scores, key=scores.get)
    prep_note = "Test prep completed ✓" if test_prep == "completed" else "No test prep — completing it typically boosts scores by 5–10 pts"

    st.markdown('<div class="section-label" style="margin-top:0;">Prediction Results</div>', unsafe_allow_html=True)

    r1, r2 = st.columns([1, 2])
    with r1:
        st.markdown(f"""
        <div class="result-card {meta['class']}">
            <div class="grade-badge">{meta['emoji']}</div>
            <div class="result-title">Grade {grade}</div>
            <div class="result-sub">{meta['label']}</div>
        </div>
        """, unsafe_allow_html=True)

    with r2:
        st.markdown(f"""
        <div class="score-row">
            <div class="score-box">
                <div class="val">{avg:.1f}</div>
                <div class="lbl">Avg Score</div>
            </div>
            <div class="score-box">
                <div class="val">{math_score}</div>
                <div class="lbl">Math</div>
            </div>
            <div class="score-box">
                <div class="val">{reading_score}</div>
                <div class="lbl">Reading</div>
            </div>
            <div class="score-box">
                <div class="val">{writing_score}</div>
                <div class="lbl">Writing</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="insight-box">
        💡 <strong>Insights & Feedback:</strong><br>
        • {meta['msg']}<br>
        • Strongest subject: <strong>{strongest}</strong> ({scores[strongest]}/100)<br>
        • Needs attention: <strong>{weakest}</strong> ({scores[weakest]}/100)<br>
        • {prep_note}
    </div>
    """, unsafe_allow_html=True)

    # Model comparison table
    st.markdown('<div class="section-label" style="margin-top:2rem">Model Comparison</div>', unsafe_allow_html=True)
    rows = ""
    for name, acc in sorted(accuracies.items(), key=lambda x: -x[1]):
        best = "best-row" if name == best_model_name else ""
        star = " ★" if name == best_model_name else ""
        rows += f'<tr class="{best}"><td>{name}{star}</td><td>{acc:.1%}</td></tr>'

    st.markdown(f"""
    <div class="model-compare" style="margin-top:0.5rem">
        <table>
            <tr><th>Model</th><th>Test Accuracy</th></tr>
            {rows}
        </table>
    </div>
    """, unsafe_allow_html=True)

else:
    best_acc = accuracies[best_model_name]
    st.markdown(f"""
    <div class="empty-state">
        <div style="font-size:4rem; opacity:0.8">🎓</div>
        <div style="font-family:'Playfair Display',serif; font-size:1.5rem;
                    color:#00d2ff; margin-top:1rem;">
            Awaiting Input...
        </div>
        <div style="color:#94a3b8; font-size:0.95rem; margin-top:0.5rem;">
            Fill in the student details and scores above, then click Predict Grade.
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="model-compare" style="margin-top:1.2rem">
        <table>
            <tr><th>Model</th><th>Test Accuracy</th></tr>
            {"".join(f'<tr class="{"best-row" if n==best_model_name else ""}"><td>{n}{"  ★" if n==best_model_name else ""}</td><td>{a:.1%}</td></tr>'
                     for n, a in sorted(accuracies.items(), key=lambda x: -x[1]))}
        </table>
    </div>
    """, unsafe_allow_html=True)
