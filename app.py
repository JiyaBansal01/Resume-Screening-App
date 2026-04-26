from dotenv import load_dotenv
import os

load_dotenv()
import streamlit as st
import pickle
import docx
import PyPDF2
import re
import numpy as np
import pandas as pd
import io
from datetime import datetime

# ─── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ResumeIQ — AI Resume Screener",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Mono:wght@400;500&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

.stApp { background: #0f1117; }

[data-testid="stSidebar"] {
    background: #161b27 !important;
    border-right: 1px solid #2a2f3e;
}
[data-testid="stSidebar"] * { color: #c8d0e0 !important; }

.main .block-container { padding: 2rem 2.5rem; }

.iq-card {
    background: #161b27;
    border: 1px solid #2a2f3e;
    border-radius: 14px;
    padding: 1.5rem;
    margin-bottom: 1.2rem;
}
.iq-card-accent {
    background: linear-gradient(135deg, #1a2035 0%, #161b27 100%);
    border: 1px solid #3b4a6b;
    border-radius: 14px;
    padding: 1.5rem;
    margin-bottom: 1.2rem;
}

.cat-badge {
    display: inline-block;
    background: #1e3a5f;
    color: #60a5fa !important;
    border: 1px solid #2563eb55;
    border-radius: 8px;
    padding: 0.35rem 0.9rem;
    font-size: 0.8rem;
    font-weight: 500;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    margin: 0.2rem 0.2rem 0.2rem 0;
}
.cat-badge-top {
    background: #1a3a1a;
    color: #4ade80 !important;
    border-color: #16a34a55;
}

.conf-bar-wrap { margin: 0.5rem 0; }
.conf-bar-label {
    display: flex;
    justify-content: space-between;
    font-size: 0.82rem;
    color: #94a3b8;
    margin-bottom: 4px;
}
.conf-bar-bg {
    background: #1e2433;
    border-radius: 6px;
    height: 8px;
    overflow: hidden;
}
.conf-bar-fill {
    height: 100%;
    border-radius: 6px;
    background: linear-gradient(90deg, #2563eb, #60a5fa);
    transition: width 0.6s ease;
}

.kw-chip {
    display: inline-block;
    background: #1a2035;
    border: 1px solid #2a3a5a;
    border-radius: 20px;
    padding: 0.25rem 0.7rem;
    font-size: 0.78rem;
    color: #94a3b8;
    margin: 0.15rem;
}
.kw-chip-missing {
    background: #2a1515;
    border-color: #7f1d1d55;
    color: #f87171;
}
.kw-chip-match {
    background: #0f2a1a;
    border-color: #14532d55;
    color: #4ade80;
}

h1 { color: #f1f5f9 !important; font-weight: 600 !important; }
h2, h3 { color: #e2e8f0 !important; font-weight: 500 !important; }

[data-testid="stMetric"] {
    background: #161b27;
    border: 1px solid #2a2f3e;
    border-radius: 10px;
    padding: 1rem;
}
[data-testid="stMetricValue"] { color: #60a5fa !important; }
[data-testid="stMetricLabel"] { color: #94a3b8 !important; }

[data-baseweb="tab-list"] { background: #161b27; border-radius: 10px; }
[data-baseweb="tab"] { color: #94a3b8 !important; }
[aria-selected="true"] { color: #60a5fa !important; }

.stButton button {
    background: #2563eb;
    color: white;
    border: none;
    border-radius: 8px;
    font-weight: 500;
}
.stButton button:hover { background: #1d4ed8; }

[data-testid="stFileUploader"] {
    background: #161b27;
    border: 2px dashed #2a3a5a;
    border-radius: 12px;
}

textarea { background: #1e2433 !important; color: #e2e8f0 !important; border-color: #2a2f3e !important; }
hr { border-color: #2a2f3e; }
</style>
""", unsafe_allow_html=True)


# ─── Load models ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    clf   = pickle.load(open('clf.pkl', 'rb'))
    tfidf = pickle.load(open('tfidf.pkl', 'rb'))
    le    = pickle.load(open('encoder.pkl', 'rb'))
    return clf, tfidf, le

clf, tfidf, le = load_models()

# ─── Category keyword map ────────────────────────────────────────────────────
CATEGORY_KEYWORDS = {
    "Data Science":             ["python","machine learning","deep learning","tensorflow","pandas","numpy","sql","statistics","sklearn","nlp","data analysis","jupyter","matplotlib","seaborn","pytorch"],
    "Java Developer":           ["java","spring","maven","hibernate","microservices","junit","rest api","sql","oop","git","docker","kafka","jenkins"],
    "Python Developer":         ["python","django","flask","fastapi","rest","sql","nosql","git","docker","aws","pandas","numpy","linux","celery"],
    "Web Designing":            ["html","css","javascript","react","figma","photoshop","ux","ui","sass","tailwind","responsive","adobe xd","bootstrap","wordpress"],
    "DevOps Engineer":          ["docker","kubernetes","aws","ci/cd","jenkins","terraform","linux","ansible","git","monitoring","bash","azure","gcp","helm"],
    "HR":                       ["recruitment","talent acquisition","onboarding","payroll","hris","employee relations","performance management","training","compliance","excel"],
    "Testing":                  ["selenium","manual testing","automation","jira","test cases","qa","regression","performance testing","postman","api testing"],
    "Business Analyst":         ["requirements","stakeholders","sql","excel","visio","tableau","powerbi","brd","agile","scrum","process mapping","jira"],
    "Network Security Engineer":["firewall","vpn","siem","ids","ips","cisco","penetration testing","vulnerability","linux","wireshark","compliance","nist"],
    "Blockchain":               ["solidity","ethereum","smart contracts","web3","defi","nft","hyperledger","cryptography","truffle","metamask"],
    "Sales":                    ["crm","salesforce","b2b","b2c","lead generation","negotiation","pipeline","excel","communication","revenue","cold calling"],
    "Database":                 ["sql","mysql","postgresql","oracle","nosql","mongodb","database design","stored procedures","etl","performance tuning"],
    "Hadoop":                   ["hadoop","spark","hive","hdfs","pig","mapreduce","kafka","hbase","scala","python","yarn","data pipeline"],
    "ETL Developer":            ["etl","informatica","ssis","sql","data warehouse","python","talend","oracle","unix","data pipeline","ab initio"],
    "Operations Manager":       ["operations","supply chain","logistics","kpi","process improvement","lean","six sigma","budget","excel","team management"],
    "Mechanical Engineer":      ["autocad","solidworks","catia","manufacturing","thermodynamics","fea","ansys","production","cad","tolerance","machining"],
    "Civil Engineer":           ["autocad","staad","revit","construction","structural","concrete","surveying","project management","roads","bridges","estimation"],
    "Electrical Engineering":   ["plc","scada","electrical design","autocad","matlab","power systems","circuit","vfd","hmi","motors","wiring"],
    "SAP Developer":            ["sap","abap","s/4hana","fiori","basis","bw","sd","mm","fi","co","debug","transport management"],
    "Automation Testing":       ["selenium","appium","java","python","testng","cucumber","jenkins","rest assured","postman","git","jira","robot framework"],
    "DotNet Developer":         ["c#",".net","asp.net","mvc","entity framework","sql server","azure","wpf","rest api","visual studio","linq"],
    "Advocate":                 ["legal research","litigation","contracts","drafting","court","compliance","ipc","crpc","arbitration","intellectual property"],
    "Arts":                     ["photography","painting","illustration","adobe","photoshop","creativity","design","portfolio","exhibitions","art direction"],
    "Health and fitness":       ["nutrition","personal training","fitness","anatomy","physiology","wellness","rehabilitation","sports science","coaching"],
    "PMO":                      ["project management","pmo","pmp","agile","scrum","risk","stakeholders","ms project","governance","budget","milestones"],
}

CATEGORY_COLORS = {
    "Data Science": "#3b82f6", "Java Developer": "#f59e0b", "Python Developer": "#10b981",
    "Web Designing": "#8b5cf6", "DevOps Engineer": "#ef4444", "HR": "#ec4899",
    "Testing": "#06b6d4", "Business Analyst": "#84cc16", "Network Security Engineer": "#f97316",
    "Blockchain": "#6366f1", "Sales": "#14b8a6", "Database": "#a855f7",
    "Hadoop": "#eab308", "ETL Developer": "#64748b", "Operations Manager": "#0ea5e9",
    "Mechanical Engineer": "#78716c", "Civil Engineer": "#92400e", "Electrical Engineering": "#fbbf24",
    "SAP Developer": "#0d9488", "Automation Testing": "#7c3aed", "DotNet Developer": "#1d4ed8",
    "Advocate": "#be185d", "Arts": "#d97706", "Health and fitness": "#059669", "PMO": "#475569",
}


# ─── Utility functions ───────────────────────────────────────────────────────
def clean_resume(txt: str) -> str:
    txt = re.sub(r'http\S+\s', ' ', txt)
    txt = re.sub(r'RT|cc', ' ', txt)
    txt = re.sub(r'#\S+\s', ' ', txt)
    txt = re.sub(r'@\S+', ' ', txt)
    txt = re.sub(r'[%s]' % re.escape(r"""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', txt)
    txt = re.sub(r'[^\x00-\x7f]', ' ', txt)
    txt = re.sub(r'\s+', ' ', txt)
    return txt.strip()


def extract_text(file) -> str:
    ext = file.name.split('.')[-1].lower()
    if ext == 'pdf':
        reader = PyPDF2.PdfReader(file)
        return ' '.join(p.extract_text() or '' for p in reader.pages)
    elif ext == 'docx':
        doc = docx.Document(file)
        return '\n'.join(p.text for p in doc.paragraphs)
    elif ext == 'txt':
        raw = file.read()
        try:    return raw.decode('utf-8')
        except: return raw.decode('latin-1')
    raise ValueError(f"Unsupported file type: .{ext}")


def get_confidence_scores(text: str) -> list:
    cleaned   = clean_resume(text)
    vec       = tfidf.transform([cleaned]).toarray()
    scores_raw = clf.decision_function(vec)[0]
    exp_s     = np.exp(scores_raw - scores_raw.max())
    probs     = exp_s / exp_s.sum() * 100
    pairs     = sorted(zip(le.classes_, probs), key=lambda x: -x[1])
    return pairs


def predict_category(text: str) -> str:
    cleaned = clean_resume(text)
    vec     = tfidf.transform([cleaned]).toarray()
    pred    = clf.predict(vec)
    return le.inverse_transform(pred)[0]


def compute_resume_score(text: str, category: str) -> dict:
    lower = text.lower()
    words = lower.split()
    kws   = CATEGORY_KEYWORDS.get(category, [])

    matched  = [k for k in kws if k in lower]
    missing  = [k for k in kws if k not in lower]
    kw_score = (len(matched) / len(kws) * 40) if kws else 0

    wc = len(words)
    if   wc < 100:   len_score = 5
    elif wc < 300:   len_score = 12
    elif wc <= 800:  len_score = 20
    elif wc <= 1200: len_score = 15
    else:            len_score = 8

    sections = {
        "Education":    any(s in lower for s in ["education","degree","university","college","bachelor","master","phd"]),
        "Experience":   any(s in lower for s in ["experience","worked","employed","job","company","role","position"]),
        "Skills":       any(s in lower for s in ["skills","technologies","tools","proficient","expertise"]),
        "Projects":     any(s in lower for s in ["project","built","developed","created","implemented"]),
        "Contact info": any(s in lower for s in ["email","phone","linkedin","github","@"]),
    }
    section_score = sum(sections.values()) * 4

    verbs = ["managed","developed","led","designed","implemented","built","created","improved","increased",
             "reduced","analyzed","collaborated","delivered","launched","optimized"]
    verb_score = min(sum(1 for v in verbs if v in lower) * 2, 10)

    numbers = re.findall(r'\b\d+[%+]?\b', text)
    quant_score = min(len(numbers) * 2, 10)

    total = int(kw_score + len_score + section_score + verb_score + quant_score)

    return {
        "total": min(total, 100),
        "breakdown": {
            "Keyword match": int(kw_score),
            "Content length": int(len_score),
            "Sections present": int(section_score),
            "Action verbs": int(verb_score),
            "Quantified results": int(quant_score),
        },
        "matched_keywords":  matched,
        "missing_keywords":  missing,
        "sections_detected": sections,
        "word_count":        wc,
    }


def skills_gap_analysis(resume_text: str, job_description: str, category: str) -> dict:
    jd_lower  = job_description.lower()
    res_lower = resume_text.lower()
    kws       = CATEGORY_KEYWORDS.get(category, [])

    jd_required         = [k for k in kws if k in jd_lower]
    present_in_resume   = [k for k in jd_required if k in res_lower]
    missing_from_resume = [k for k in jd_required if k not in res_lower]
    match_pct = int(len(present_in_resume) / len(jd_required) * 100) if jd_required else 0

    return {
        "match_pct": match_pct,
        "jd_keywords": jd_required,
        "present":     present_in_resume,
        "missing":     missing_from_resume,
    }


def generate_csv(results: list) -> bytes:
    rows = []
    for r in results:
        rows.append({
            "File":           r["filename"],
            "Category":       r["category"],
            "Confidence (%)": f"{r['top_confidence']:.1f}",
            "Resume Score":   r["score"],
            "Word Count":     r["word_count"],
            "Matched Keywords": ", ".join(r["matched_kws"]),
            "Missing Keywords": ", ".join(r["missing_kws"]),
            "Timestamp":      r["timestamp"],
        })
    return pd.DataFrame(rows).to_csv(index=False).encode()


# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📄 ResumeIQ")
    st.markdown("*Smart Resume Analyzerc*")
    st.divider()

    mode = st.radio(
        "Mode",
        ["Single Resume", "Batch Screening", "Skills Gap Analysis"],
        label_visibility="collapsed",
    )
    st.divider()



    st.markdown("**Model info**")
    st.markdown(f"- `OneVsRest + SVC`")
    st.markdown(f"- `TF-IDF vectorizer`")
    st.markdown(f"- `{len(le.classes_)} job categories`")
    st.divider()

    st.markdown("**Supported formats**")
    st.markdown("PDF · DOCX · TXT")


# ─── SINGLE RESUME MODE ──────────────────────────────────────────────────────
if mode == "Single Resume":
    st.markdown("# Resume Analyzer")
    st.markdown("Upload a resume to get category prediction, confidence scores, and quality analysis.")
    st.markdown("")

    uploaded = st.file_uploader("Drop a resume here", type=["pdf","docx","txt"], label_visibility="collapsed")

    if uploaded:
        with st.spinner("Analyzing resume..."):
            try:
                text = extract_text(uploaded)
            except Exception as e:
                st.error(f"Could not read file: {e}")
                st.stop()

            category    = predict_category(text)
            conf_scores = get_confidence_scores(text)
            top_conf    = conf_scores[0][1]
            score_data  = compute_resume_score(text, category)

        col1, col2, col3, col4 = st.columns(4)
        with col1: st.metric("Predicted Category", category)
        with col2: st.metric("Confidence", f"{top_conf:.1f}%")
        with col3: st.metric("Resume Score", f"{score_data['total']}/100")
        with col4: st.metric("Word Count", score_data["word_count"])

        st.markdown("")
        tab1, tab2, tab3, tab4 = st.tabs(["🎯 Prediction", "📊 Quality Score", "🔑 Keywords", "📝 Raw Text"])

        # ── Tab 1: Prediction ────────────────────────────────────────────────
        with tab1:
            c1, c2 = st.columns([1, 1])
            with c1:
                color = CATEGORY_COLORS.get(category, "#60a5fa")
                st.markdown(f"""
                <div class="iq-card-accent">
                  <div style="font-size:0.8rem;color:#94a3b8;letter-spacing:.06em;text-transform:uppercase;margin-bottom:.5rem">Best match</div>
                  <div style="font-size:2rem;font-weight:600;color:{color}">{category}</div>
                  <div style="margin-top:.5rem;font-size:0.9rem;color:#64748b">{top_conf:.1f}% confidence</div>
                </div>
                """, unsafe_allow_html=True)

                st.markdown("**Other possible categories**")
                for cat, pct in conf_scores[1:6]:
                    c = CATEGORY_COLORS.get(cat, "#60a5fa")
                    st.markdown(f"""
                    <div class="conf-bar-wrap">
                      <div class="conf-bar-label"><span>{cat}</span><span>{pct:.1f}%</span></div>
                      <div class="conf-bar-bg"><div class="conf-bar-fill" style="width:{min(pct,100):.1f}%;background:linear-gradient(90deg,{c}88,{c})"></div></div>
                    </div>
                    """, unsafe_allow_html=True)

            with c2:
                st.markdown("**All 25 categories**")
                html = ""
                for cat, pct in conf_scores:
                    c = CATEGORY_COLORS.get(cat, "#60a5fa")
                    bold = "font-weight:600;" if cat == category else ""
                    html += f'<span class="cat-badge" style="{bold}border-color:{c}44;color:{c};background:{c}15">{cat} <span style="opacity:.7">{pct:.0f}%</span></span>'
                st.markdown(html, unsafe_allow_html=True)

        # ── Tab 2: Quality Score ─────────────────────────────────────────────
        with tab2:
            total = score_data["total"]
            color = "#4ade80" if total >= 70 else "#fbbf24" if total >= 45 else "#f87171"
            grade = "Excellent" if total >= 80 else "Good" if total >= 65 else "Average" if total >= 45 else "Needs work"

            c1, c2 = st.columns([1, 1.5])
            with c1:
                st.markdown(f"""
                <div class="iq-card" style="text-align:center;padding:2rem">
                  <div style="font-size:4rem;font-weight:700;font-family:'DM Mono',monospace;color:{color}">{total}</div>
                  <div style="font-size:1rem;color:{color};margin-top:.2rem">{grade}</div>
                  <div style="font-size:.8rem;color:#64748b;margin-top:.5rem">out of 100</div>
                </div>
                """, unsafe_allow_html=True)

                st.markdown("**Sections detected**")
                for sec, found in score_data["sections_detected"].items():
                    icon = "✅" if found else "❌"
                    st.markdown(f"{icon} {sec}")

            with c2:
                st.markdown("**Score breakdown**")
                for label, pts in score_data["breakdown"].items():
                    maxpts = {"Keyword match": 40, "Content length": 20, "Sections present": 20,
                              "Action verbs": 10, "Quantified results": 10}[label]
                    pct = pts / maxpts * 100
                    c = "#4ade80" if pct >= 70 else "#fbbf24" if pct >= 40 else "#f87171"
                    st.markdown(f"""
                    <div class="conf-bar-wrap" style="margin-bottom:.8rem">
                      <div class="conf-bar-label">
                        <span>{label}</span>
                        <span>{pts}/{maxpts} pts</span>
                      </div>
                      <div class="conf-bar-bg" style="height:10px">
                        <div class="conf-bar-fill" style="width:{pct:.0f}%;background:linear-gradient(90deg,{c}88,{c})"></div>
                      </div>
                    </div>
                    """, unsafe_allow_html=True)

        # ── Tab 3: Keywords ──────────────────────────────────────────────────
        with tab3:
            c1, c2 = st.columns(2)
            with c1:
                st.markdown(f"**✅ Found keywords** ({len(score_data['matched_keywords'])})")
                html = " ".join(f'<span class="kw-chip kw-chip-match">{k}</span>' for k in score_data['matched_keywords'])
                st.markdown(html or "<span style='color:#64748b'>None detected</span>", unsafe_allow_html=True)
            with c2:
                st.markdown(f"**❌ Missing keywords** ({len(score_data['missing_keywords'])})")
                html = " ".join(f'<span class="kw-chip kw-chip-missing">{k}</span>' for k in score_data['missing_keywords'])
                st.markdown(html or "<span style='color:#4ade80'>All key skills present!</span>", unsafe_allow_html=True)

            if score_data['missing_keywords']:
                st.markdown("")
                st.info(f"💡 Add these {len(score_data['missing_keywords'])} missing keywords to improve your score by up to **{len(score_data['missing_keywords']) * 3}** points.")

        # ── Tab 4: Raw Text ──────────────────────────────────────────────────
        with tab4:
            st.text_area("Extracted text", text, height=350, label_visibility="collapsed")


# ─── BATCH SCREENING MODE ────────────────────────────────────────────────────
elif mode == "Batch Screening":
    st.markdown("# Batch Resume Screening")
    st.markdown("Upload multiple resumes at once. Results shown in a ranked table and exportable as CSV.")
    st.markdown("")

    files = st.file_uploader(
        "Upload resumes (multiple allowed)",
        type=["pdf","docx","txt"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    filter_cat = st.selectbox("Filter by category (optional)", ["All"] + sorted(le.classes_))

    if files:
        results = []
        prog = st.progress(0, text="Processing resumes...")

        for i, f in enumerate(files):
            try:
                text  = extract_text(f)
                cat   = predict_category(text)
                confs = get_confidence_scores(text)
                top_c = confs[0][1]
                sc    = compute_resume_score(text, cat)
                results.append({
                    "filename":       f.name,
                    "category":       cat,
                    "top_confidence": top_c,
                    "score":          sc["total"],
                    "word_count":     sc["word_count"],
                    "matched_kws":    sc["matched_keywords"],
                    "missing_kws":    sc["missing_keywords"],
                    "timestamp":      datetime.now().strftime("%Y-%m-%d %H:%M"),
                })
            except Exception as e:
                results.append({
                    "filename": f.name, "category": "Error", "top_confidence": 0,
                    "score": 0, "word_count": 0, "matched_kws": [], "missing_kws": [],
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
                })
            prog.progress((i+1)/len(files), text=f"Processing {i+1}/{len(files)}...")

        prog.empty()

        shown = [r for r in results if filter_cat == "All" or r["category"] == filter_cat]
        shown.sort(key=lambda x: -x["score"])

        valid = [r for r in results if r["category"] != "Error"]
        c1, c2, c3, c4 = st.columns(4)
        with c1: st.metric("Total processed", len(results))
        with c2: st.metric("Unique categories", len(set(r["category"] for r in valid)))
        with c3: st.metric("Avg resume score", f"{np.mean([r['score'] for r in valid]):.0f}/100" if valid else "—")
        with c4: st.metric("Top score", f"{max(r['score'] for r in valid)}/100" if valid else "—")

        st.markdown("")
        st.markdown("### Results (sorted by score)")
        for r in shown:
            with st.expander(f"📄 {r['filename']}  —  {r['category']}  —  Score: {r['score']}/100"):
                cc1, cc2, cc3 = st.columns(3)
                with cc1: st.metric("Category", r["category"])
                with cc2: st.metric("Confidence", f"{r['top_confidence']:.1f}%")
                with cc3: st.metric("Score", f"{r['score']}/100")

                if r["matched_kws"]:
                    html = " ".join(f'<span class="kw-chip kw-chip-match">{k}</span>' for k in r["matched_kws"][:10])
                    st.markdown("**Matched keywords:** " + html, unsafe_allow_html=True)
                if r["missing_kws"]:
                    html = " ".join(f'<span class="kw-chip kw-chip-missing">{k}</span>' for k in r["missing_kws"][:8])
                    st.markdown("**Missing keywords:** " + html, unsafe_allow_html=True)

        st.markdown("")
        csv_bytes = generate_csv(results)
        st.download_button(
            label="⬇️  Export all results as CSV",
            data=csv_bytes,
            file_name=f"resume_screening_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv",
        )


# ─── SKILLS GAP ANALYSIS MODE ───────────────────────────────────────────────
elif mode == "Skills Gap Analysis":
    st.markdown("# Skills Gap Analysis")
    st.markdown("Compare a resume against a job description to see which skills are missing.")
    st.markdown("")

    c1, c2 = st.columns(2)
    with c1:
        resume_file = st.file_uploader("Upload resume", type=["pdf","docx","txt"])
    with c2:
        jd_text = st.text_area("Paste job description here", height=180, placeholder="Paste the full job description...")

    if resume_file and jd_text.strip():
        with st.spinner("Running gap analysis..."):
            try:
                resume_text = extract_text(resume_file)
            except Exception as e:
                st.error(str(e)); st.stop()

            category   = predict_category(resume_text)
            gap        = skills_gap_analysis(resume_text, jd_text, category)
            score_data = compute_resume_score(resume_text, category)

        match_color = "#4ade80" if gap["match_pct"] >= 70 else "#fbbf24" if gap["match_pct"] >= 40 else "#f87171"

        c1, c2, c3 = st.columns(3)
        with c1: st.metric("Detected category", category)
        with c2: st.metric("JD match", f"{gap['match_pct']}%")
        with c3: st.metric("Resume score", f"{score_data['total']}/100")

        st.markdown("")
        st.markdown(f"""
        <div class="conf-bar-wrap" style="margin-bottom:1.5rem">
          <div class="conf-bar-label">
            <span style="font-size:1rem;font-weight:500">Overall JD keyword match</span>
            <span style="font-size:1.1rem;font-weight:600;color:{match_color}">{gap['match_pct']}%</span>
          </div>
          <div class="conf-bar-bg" style="height:14px;border-radius:8px">
            <div class="conf-bar-fill" style="width:{gap['match_pct']}%;background:linear-gradient(90deg,{match_color}88,{match_color});border-radius:8px"></div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"### ✅ Skills you have ({len(gap['present'])})")
            if gap["present"]:
                html = " ".join(f'<span class="kw-chip kw-chip-match">{k}</span>' for k in gap["present"])
                st.markdown(html, unsafe_allow_html=True)
            else:
                st.markdown('<span style="color:#64748b">No matching JD keywords found</span>', unsafe_allow_html=True)

        with col2:
            st.markdown(f"### ❌ Skills to add ({len(gap['missing'])})")
            if gap["missing"]:
                html = " ".join(f'<span class="kw-chip kw-chip-missing">{k}</span>' for k in gap["missing"])
                st.markdown(html, unsafe_allow_html=True)
                st.markdown("")
                st.warning(f"💡 Adding **{len(gap['missing'])}** missing skills could increase your JD match to **100%**. Consider adding them to your skills section.")
            else:
                st.success("🎉 Your resume covers all detected JD keywords!")

        st.markdown("")
        st.markdown("### Resume quality for this role")
        for label, pts in score_data["breakdown"].items():
            maxpts = {"Keyword match": 40, "Content length": 20, "Sections present": 20,
                      "Action verbs": 10, "Quantified results": 10}[label]
            pct = pts / maxpts * 100
            c = "#4ade80" if pct >= 70 else "#fbbf24" if pct >= 40 else "#f87171"
            st.markdown(f"""
            <div class="conf-bar-wrap">
              <div class="conf-bar-label"><span>{label}</span><span>{pts}/{maxpts} pts</span></div>
              <div class="conf-bar-bg" style="height:10px">
                <div class="conf-bar-fill" style="width:{pct:.0f}%;background:linear-gradient(90deg,{c}88,{c})"></div>
              </div>
            </div>
            """, unsafe_allow_html=True)

    elif resume_file and not jd_text.strip():
        st.info("👆 Now paste a job description on the right to run the gap analysis.")
    elif not resume_file and jd_text.strip():
        st.info("👆 Now upload a resume on the left to run the gap analysis.")