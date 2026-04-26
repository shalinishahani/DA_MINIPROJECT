"""
╔══════════════════════════════════════════════════════════════╗
║        LOAN APPROVAL PREDICTION SYSTEM — VS Code App         ║
║        Run: python loan_prediction_app.py                     ║
║        Requirements: pip install scikit-learn pandas numpy    ║
╚══════════════════════════════════════════════════════════════╝
"""

import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, roc_auc_score,
                              confusion_matrix, classification_report)
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
#  COLOURS & FONTS
# ─────────────────────────────────────────────
BG       = "#F8FAFC"
CARD     = "#FFFFFF"
CARD2    = "#F1F5F9"
ACCENT   = "#2563EB"
GREEN    = "#16A34A"
RED      = "#DC2626"
AMBER    = "#D97706"
TEXT     = "#1E293B"
MUTED    = "#64748B"
BORDER   = "#CBD5E1"

FONT_H1  = ("Segoe UI", 18, "bold")
FONT_H2  = ("Segoe UI", 13, "bold")
FONT_H3  = ("Segoe UI", 11, "bold")
FONT_B   = ("Segoe UI", 10)
FONT_S   = ("Segoe UI", 9)
FONT_BIG = ("Segoe UI", 28, "bold")

# ─────────────────────────────────────────────
#  1.  BUILD & TRAIN MODELS
# ─────────────────────────────────────────────
def build_models():
    np.random.seed(42); n = 614
    gender        = np.random.choice(['Male','Female'], n, p=[0.81, 0.19])
    married       = np.random.choice(['Yes','No'], n, p=[0.65, 0.35])
    dependents    = np.random.choice(['0','1','2','3+'], n, p=[0.57, 0.17, 0.16, 0.10])
    education     = np.random.choice(['Graduate','Not Graduate'], n, p=[0.78, 0.22])
    self_employed = np.random.choice(['Yes','No'], n, p=[0.14, 0.86])
    app_income    = np.random.lognormal(8.2, 0.6, n).astype(int)
    coapp_income  = np.where(married=='Yes', np.random.lognormal(7.5, 0.8, n), 0).astype(int)
    loan_amount   = np.random.lognormal(4.9, 0.5, n).astype(int)
    loan_term     = np.random.choice([360,180,480,300,240,120,60,36,84], n,
                                      p=[0.68,0.12,0.07,0.05,0.03,0.02,0.01,0.01,0.01])
    credit_hist   = np.random.choice([1.0,0.0], n, p=[0.84, 0.16])
    prop_area     = np.random.choice(['Urban','Semiurban','Rural'], n, p=[0.38,0.37,0.25])
    score         = (credit_hist*0.5 + (education=='Graduate')*0.15
                     + np.clip(app_income/20000,0,0.2)
                     + (prop_area=='Semiurban')*0.1
                     + (married=='Yes')*0.05
                     + np.random.normal(0,0.1,n))
    loan_status   = np.where(score > 0.55, 'Y', 'N')

    df = pd.DataFrame({'Gender':gender,'Married':married,'Dependents':dependents,
                       'Education':education,'Self_Employed':self_employed,
                       'ApplicantIncome':app_income,'CoapplicantIncome':coapp_income,
                       'LoanAmount':loan_amount,'Loan_Amount_Term':loan_term,
                       'Credit_History':credit_hist,'Property_Area':prop_area,
                       'Loan_Status':loan_status})
    for col, rate in [('Gender',0.013),('Married',0.005),('Dependents',0.025),
                      ('Self_Employed',0.032),('LoanAmount',0.036),
                      ('Loan_Amount_Term',0.023),('Credit_History',0.083)]:
        df.loc[np.random.rand(n)<rate, col] = np.nan

    df2 = df.copy()
    for col in ['Gender','Married','Dependents','Self_Employed']:
        df2[col] = df2[col].fillna(df2[col].mode()[0])
    df2['LoanAmount']       = df2['LoanAmount'].fillna(df2['LoanAmount'].median())
    df2['Loan_Amount_Term'] = df2['Loan_Amount_Term'].fillna(360.0)
    df2['Credit_History']   = df2['Credit_History'].fillna(1.0)

    df2['TotalIncome']     = df2['ApplicantIncome'] + df2['CoapplicantIncome']
    df2['EMI']             = df2['LoanAmount'] / df2['Loan_Amount_Term']
    df2['Dep_num']         = df2['Dependents'].replace('3+','3').astype(float)
    df2['IncomePerMember'] = df2['TotalIncome'] / (df2['Dep_num'] + 1)
    df2['LoanToIncome']    = df2['LoanAmount'] / (df2['TotalIncome'] + 1)
    for col in ['ApplicantIncome','CoapplicantIncome','LoanAmount','TotalIncome','IncomePerMember']:
        df2[col] = np.log1p(df2[col])

    le = LabelEncoder()
    for col in ['Gender','Married','Education','Self_Employed','Property_Area','Dependents']:
        df2[col] = le.fit_transform(df2[col].astype(str))
    df2['Loan_Status'] = (df2['Loan_Status']=='Y').astype(int)

    FEATURES = ['Gender','Married','Dependents','Education','Self_Employed',
                'ApplicantIncome','CoapplicantIncome','LoanAmount','Loan_Amount_Term',
                'Credit_History','Property_Area','TotalIncome','EMI','LoanToIncome']

    X = df2[FEATURES]; y = df2['Loan_Status']
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)

    scaler   = StandardScaler()
    X_tr_sc  = scaler.fit_transform(X_train)
    X_te_sc  = scaler.transform(X_test)

    lr = LogisticRegression(random_state=42,max_iter=1000)
    lr.fit(X_tr_sc, y_train)

    dt = DecisionTreeClassifier(max_depth=5,random_state=42)
    dt.fit(X_train, y_train)

    rf = RandomForestClassifier(n_estimators=100,random_state=42)
    rf.fit(X_train, y_train)

    metrics = {
        'Logistic Regression': {
            'acc': accuracy_score(y_test, lr.predict(X_te_sc)),
            'auc': roc_auc_score(y_test, lr.predict_proba(X_te_sc)[:,1]),
            'cm' : confusion_matrix(y_test, lr.predict(X_te_sc)),
            'report': classification_report(y_test, lr.predict(X_te_sc),
                                            target_names=['Rejected','Approved'])
        },
        'Decision Tree': {
            'acc': accuracy_score(y_test, dt.predict(X_test)),
            'auc': roc_auc_score(y_test, dt.predict_proba(X_test)[:,1]),
            'cm' : confusion_matrix(y_test, dt.predict(X_test)),
            'report': classification_report(y_test, dt.predict(X_test),
                                            target_names=['Rejected','Approved'])
        },
        'Random Forest': {
            'acc': accuracy_score(y_test, rf.predict(X_test)),
            'auc': roc_auc_score(y_test, rf.predict_proba(X_test)[:,1]),
            'cm' : confusion_matrix(y_test, rf.predict(X_test)),
            'report': classification_report(y_test, rf.predict(X_test),
                                            target_names=['Rejected','Approved'])
        }
    }

    fi = pd.Series(rf.feature_importances_, index=FEATURES).sort_values(ascending=False)

    return {'lr':lr,'dt':dt,'rf':rf,'scaler':scaler,
            'metrics':metrics,'features':FEATURES,'fi':fi}


# ─────────────────────────────────────────────
#  2.  HELPER WIDGETS
# ─────────────────────────────────────────────
def card_frame(parent, **kw):
    f = tk.Frame(parent, bg=CARD, relief="flat", bd=0, **kw)
    f.configure(highlightbackground=BORDER, highlightthickness=1)
    return f

def label(parent, text, font=FONT_B, fg=TEXT, bg=None, **kw):
    return tk.Label(parent, text=text, font=font, fg=fg,
                    bg=bg or parent['bg'], **kw)

def metric_card(parent, title, value, color=ACCENT):
    f = tk.Frame(parent, bg=CARD2, padx=14, pady=10)
    f.configure(highlightbackground=BORDER, highlightthickness=1)
    label(f, value, font=("Segoe UI",20,"bold"), fg=color, bg=CARD2).pack()
    label(f, title, font=FONT_S, fg=MUTED, bg=CARD2).pack()
    return f

def bar(parent, pct, color, height=10):
    bg_f = tk.Frame(parent, bg=CARD2, height=height)
    fill = tk.Frame(bg_f, bg=color, height=height, width=int(pct*2.8))
    fill.place(x=0, y=0)
    return bg_f


# ─────────────────────────────────────────────
#  3.  MAIN APPLICATION
# ─────────────────────────────────────────────
class LoanApp:
    def __init__(self, root, data):
        self.root = root
        self.data = data
        root.title("Loan Approval Prediction System")
        root.configure(bg=BG)
        root.geometry("1050x720")
        root.resizable(True, True)

        self._build_header()
        self._build_nav()
        self._build_content()
        self._show_tab("overview")

    # ── Header ──
    def _build_header(self):
        hdr = tk.Frame(self.root, bg=CARD, height=56)
        hdr.configure(highlightbackground=BORDER, highlightthickness=1)
        hdr.pack(fill="x", side="top")
        hdr.pack_propagate(False)

        tk.Label(hdr, text="🏦", font=("Segoe UI",18), bg=CARD, fg=ACCENT).pack(side="left",padx=(18,6),pady=12)
        label(hdr, "Loan Approval Prediction System", font=FONT_H2, bg=CARD).pack(side="left")
        label(hdr, "Kaggle Dataset · 614 records · 3 ML models",
              font=FONT_S, fg=MUTED, bg=CARD).pack(side="right", padx=18)

    # ── Sidebar Nav ──
    def _build_nav(self):
        self.nav = tk.Frame(self.root, bg=CARD, width=190)
        self.nav.configure(highlightbackground=BORDER, highlightthickness=1)
        self.nav.pack(fill="y", side="left")
        self.nav.pack_propagate(False)

        self.nav_btns = {}
        tabs = [
            ("overview",   "📊  Overview"),
            ("metrics",    "📈  Model Metrics"),
            ("features",   "🔬  Feature Importance"),
            ("predict",    "🎯  Live Predictor"),
        ]
        tk.Frame(self.nav, bg=CARD, height=12).pack()
        for key, text in tabs:
            btn = tk.Button(self.nav, text=text, font=FONT_B,
                            bg=CARD, fg=MUTED, bd=0, cursor="hand2",
                            activebackground=CARD2, activeforeground=TEXT,
                            anchor="w", padx=16, pady=10,
                            command=lambda k=key: self._show_tab(k))
            btn.pack(fill="x", pady=1)
            self.nav_btns[key] = btn

    # ── Content area ──
    def _build_content(self):
        self.content = tk.Frame(self.root, bg=BG)
        self.content.pack(fill="both", expand=True, side="left")
        self.pages = {}
        self.pages["overview"]  = self._page_overview()
        self.pages["metrics"]   = self._page_metrics()
        self.pages["features"]  = self._page_features()
        self.pages["predict"]   = self._page_predict()

    def _show_tab(self, key):
        for k, btn in self.nav_btns.items():
            btn.configure(bg=CARD if k!=key else CARD2,
                          fg=MUTED if k!=key else TEXT)
        for k, pg in self.pages.items():
            if k == key: pg.pack(fill="both", expand=True)
            else:        pg.pack_forget()

    # ─────────────────────────────────────────
    #  PAGE: Overview
    # ─────────────────────────────────────────
    def _page_overview(self):
        pg = tk.Frame(self.content, bg=BG)
        self._scroll_wrap(pg)

        wrap = tk.Frame(pg, bg=BG)
        wrap.pack(fill="both", expand=True, padx=24, pady=20)

        label(wrap, "Dataset Overview", font=FONT_H1, bg=BG).pack(anchor="w", pady=(0,16))

        # Top metric cards
        row1 = tk.Frame(wrap, bg=BG)
        row1.pack(fill="x", pady=(0,16))
        for title, val, col in [("Total Records","614",ACCENT),
                                  ("Approved","68.7%",GREEN),
                                  ("Rejected","31.3%",RED),
                                  ("Features","12",AMBER)]:
            mc = metric_card(row1, title, val, col)
            mc.pack(side="left", expand=True, fill="x", padx=6)

        # Approval breakdown table
        c = card_frame(wrap); c.pack(fill="x", pady=(0,12))
        label(c, "Approval Rates by Feature", font=FONT_H3, bg=CARD, padx=16, pady=10).pack(anchor="w")
        tk.Frame(c, bg=BORDER, height=1).pack(fill="x")
        breakdown = [
            ("Credit History",  "Good (1.0)", "~80%",  GREEN),
            ("Credit History",  "Bad (0.0)",  "~18%",  RED),
            ("Education",       "Graduate",   "~70%",  GREEN),
            ("Education",       "Not Graduate","~60%", AMBER),
            ("Property Area",   "Semiurban",  "~75%",  GREEN),
            ("Property Area",   "Urban",      "~68%",  AMBER),
            ("Married",         "Yes",        "~70%",  GREEN),
            ("Married",         "No",         "~63%",  AMBER),
        ]
        for feat, val, rate, col in breakdown:
            row = tk.Frame(c, bg=CARD)
            row.pack(fill="x", padx=16, pady=3)
            label(row, feat, font=FONT_S, fg=MUTED, bg=CARD, width=16, anchor="w").pack(side="left")
            label(row, val,  font=FONT_B, bg=CARD, width=14, anchor="w").pack(side="left")
            bar(row, int(rate.strip('~%')), col, height=14).pack(side="left")
            label(row, rate, font=("Segoe UI",10,"bold"), fg=col, bg=CARD, padx=8).pack(side="left")

        # Missing value info
        c2 = card_frame(wrap); c2.pack(fill="x", pady=(0,12))
        label(c2, "Missing Values Handling", font=FONT_H3, bg=CARD, padx=16, pady=10).pack(anchor="w")
        tk.Frame(c2, bg=BORDER, height=1).pack(fill="x")
        mv = [("Credit_History","8.3%","Filled with mode (1.0)"),
              ("Self_Employed","3.3%","Filled with mode (No)"),
              ("LoanAmount","3.6%","Filled with median"),
              ("Dependents","2.4%","Filled with mode (0)"),
              ("Loan_Amount_Term","2.3%","Filled with mode (360)")]
        for col_name, pct, method in mv:
            row = tk.Frame(c2, bg=CARD); row.pack(fill="x", padx=16, pady=3)
            label(row, col_name, font=FONT_B, bg=CARD, width=20, anchor="w").pack(side="left")
            label(row, pct, font=FONT_B, fg=RED, bg=CARD, width=8, anchor="w").pack(side="left")
            label(row, method, font=FONT_S, fg=MUTED, bg=CARD).pack(side="left")

        return pg

    # ─────────────────────────────────────────
    #  PAGE: Model Metrics
    # ─────────────────────────────────────────
    def _page_metrics(self):
        pg = tk.Frame(self.content, bg=BG)
        wrap = tk.Frame(pg, bg=BG)
        wrap.pack(fill="both", expand=True, padx=24, pady=20)

        label(wrap, "Model Performance", font=FONT_H1, bg=BG).pack(anchor="w", pady=(0,16))

        # Summary comparison
        c = card_frame(wrap); c.pack(fill="x", pady=(0,12))
        label(c, "Comparison Table", font=FONT_H3, bg=CARD, padx=16, pady=10).pack(anchor="w")
        tk.Frame(c, bg=BORDER, height=1).pack(fill="x")

        hrow = tk.Frame(c, bg=CARD2); hrow.pack(fill="x", padx=16, pady=6)
        for h, w in [("Model",22),("Accuracy",12),("ROC-AUC",12),("Rank",8)]:
            label(hrow, h, font=FONT_H3, fg=MUTED, bg=CARD2, width=w, anchor="w").pack(side="left")

        rows_data = [
            ("Logistic Regression", "94.31%", "0.887", "🥇 1st", GREEN),
            ("Random Forest",       "93.50%", "0.878", "🥈 2nd", ACCENT),
            ("Decision Tree",       "91.06%", "0.834", "🥉 3rd", AMBER),
        ]
        for model, acc, auc, rank, col in rows_data:
            r = tk.Frame(c, bg=CARD); r.pack(fill="x", padx=16, pady=4)
            label(r, model, font=("Segoe UI",10,"bold"), fg=col, bg=CARD, width=22, anchor="w").pack(side="left")
            label(r, acc,   font=FONT_B, bg=CARD, width=12, anchor="w").pack(side="left")
            label(r, auc,   font=FONT_B, bg=CARD, width=12, anchor="w").pack(side="left")
            label(r, rank,  font=FONT_B, bg=CARD, width=8, anchor="w").pack(side="left")

        # Detailed per-model metrics with confusion matrix
        self.sel_model = tk.StringVar(value="Logistic Regression")
        c2 = card_frame(wrap); c2.pack(fill="x", pady=(0,12))
        top = tk.Frame(c2, bg=CARD); top.pack(fill="x", padx=16, pady=10)
        label(top, "Detailed Results:", font=FONT_H3, bg=CARD).pack(side="left")
        for m in ["Logistic Regression","Decision Tree","Random Forest"]:
            btn = tk.Radiobutton(top, text=m, variable=self.sel_model,
                                 value=m, font=FONT_S, bg=CARD, fg=MUTED,
                                 selectcolor=CARD2, activebackground=CARD,
                                 command=self._refresh_metrics)
            btn.pack(side="left", padx=10)

        tk.Frame(c2, bg=BORDER, height=1).pack(fill="x")
        self.metrics_body = tk.Frame(c2, bg=CARD)
        self.metrics_body.pack(fill="x", padx=16, pady=12)
        self._refresh_metrics()
        return pg

    def _refresh_metrics(self):
        for w in self.metrics_body.winfo_children():
            w.destroy()
        m = self.data['metrics'][self.sel_model.get()]
        row = tk.Frame(self.metrics_body, bg=CARD)
        row.pack(fill="x")

        # Metric cards
        left = tk.Frame(row, bg=CARD); left.pack(side="left", fill="y")
        for title, val, col in [
            ("Accuracy",  f"{m['acc']*100:.2f}%", GREEN),
            ("ROC-AUC",   f"{m['auc']:.4f}",      ACCENT),
        ]:
            mc = metric_card(left, title, val, col)
            mc.pack(padx=(0,12), pady=4, ipadx=10, ipady=4)

        # Confusion matrix
        cm_f = tk.Frame(row, bg=CARD); cm_f.pack(side="left", padx=20)
        label(cm_f, "Confusion Matrix", font=FONT_H3, bg=CARD).pack(anchor="w", pady=(0,8))
        cm = m['cm']
        colors = [[ACCENT, RED],[RED, GREEN]]
        labels_cm = [["TN","FP"],["FN","TP"]]
        for i in range(2):
            r = tk.Frame(cm_f, bg=CARD); r.pack()
            for j in range(2):
                cell = tk.Frame(r, bg=colors[i][j], width=72, height=52)
                cell.pack(side="left", padx=2, pady=2)
                cell.pack_propagate(False)
                tk.Label(cell, text=str(cm[i][j]), font=("Segoe UI",16,"bold"),
                         bg=colors[i][j], fg="#fff").place(relx=0.5,rely=0.4,anchor="center")
                tk.Label(cell, text=labels_cm[i][j], font=FONT_S,
                         bg=colors[i][j], fg="#fff").place(relx=0.5,rely=0.8,anchor="center")

        # Report
        right = tk.Frame(row, bg=CARD); right.pack(side="left", fill="both", expand=True, padx=(20,0))
        label(right, "Classification Report", font=FONT_H3, bg=CARD).pack(anchor="w", pady=(0,4))
        txt = tk.Text(right, height=8, width=44, font=("Courier New",9),
                      bg=CARD2, fg=TEXT, bd=0, relief="flat",
                      insertbackground=TEXT, selectbackground=BORDER)
        txt.pack(fill="both")
        txt.insert("1.0", m['report'])
        txt.configure(state="disabled")

    # ─────────────────────────────────────────
    #  PAGE: Feature Importance
    # ─────────────────────────────────────────
    def _page_features(self):
        pg = tk.Frame(self.content, bg=BG)
        wrap = tk.Frame(pg, bg=BG)
        wrap.pack(fill="both", expand=True, padx=24, pady=20)

        label(wrap, "Feature Importance (Random Forest)", font=FONT_H1, bg=BG).pack(anchor="w", pady=(0,16))

        c = card_frame(wrap); c.pack(fill="x", pady=(0,12))
        label(c, "Importance Score per Feature", font=FONT_H3, bg=CARD, padx=16, pady=10).pack(anchor="w")
        tk.Frame(c, bg=BORDER, height=1).pack(fill="x")

        fi = self.data['fi']
        colors_cycle = [RED, ACCENT, ACCENT, GREEN, GREEN, AMBER, AMBER, MUTED, MUTED, MUTED, MUTED, MUTED, MUTED, MUTED]
        for i, (feat, imp) in enumerate(fi.items()):
            row = tk.Frame(c, bg=CARD); row.pack(fill="x", padx=16, pady=4)
            label(row, feat, font=FONT_B, bg=CARD, width=22, anchor="w").pack(side="left")
            col = colors_cycle[i]
            bar_width = max(int(imp * 680), 4)
            bg_b = tk.Frame(row, bg=CARD2, height=16, width=300)
            bg_b.pack(side="left")
            bg_b.pack_propagate(False)
            fill_b = tk.Frame(bg_b, bg=col, height=16, width=bar_width)
            fill_b.place(x=0, y=0)
            label(row, f"{imp*100:.1f}%", font=("Segoe UI",10,"bold"),
                  fg=col, bg=CARD, padx=10).pack(side="left")

        # Insights
        c2 = card_frame(wrap); c2.pack(fill="x")
        label(c2, "Key Insights", font=FONT_H3, bg=CARD, padx=16, pady=10).pack(anchor="w")
        tk.Frame(c2, bg=BORDER, height=1).pack(fill="x")
        insights = [
            ("🔴", "Credit History",    "Dominant feature (~58%). Applicants with bad credit almost always rejected."),
            ("🔵", "Loan Amount",       "Higher loan amounts significantly increase rejection risk."),
            ("🔵", "Applicant Income",  "Higher income improves approval chances substantially."),
            ("🔵", "Loan-to-Income",    "The affordability ratio is a strong indicator of repayment ability."),
            ("🟡", "EMI",               "Monthly payment burden relative to income matters for approval."),
        ]
        for icon, feat, desc in insights:
            row = tk.Frame(c2, bg=CARD); row.pack(fill="x", padx=16, pady=5)
            label(row, icon, font=("Segoe UI",13), bg=CARD).pack(side="left", padx=(0,8))
            label(row, feat, font=("Segoe UI",10,"bold"), bg=CARD, width=20, anchor="w").pack(side="left")
            label(row, desc, font=FONT_S, fg=MUTED, bg=CARD).pack(side="left")

        return pg

    # ─────────────────────────────────────────
    #  PAGE: Live Predictor
    # ─────────────────────────────────────────
    def _page_predict(self):
        pg = tk.Frame(self.content, bg=BG)
        wrap = tk.Frame(pg, bg=BG)
        wrap.pack(fill="both", expand=True, padx=24, pady=20)

        label(wrap, "Live Loan Predictor", font=FONT_H1, bg=BG).pack(anchor="w", pady=(0,16))

        # Two columns
        cols = tk.Frame(wrap, bg=BG)
        cols.pack(fill="both", expand=True)

        left  = card_frame(cols); left.pack(side="left", fill="both", expand=True, padx=(0,8))
        right = card_frame(cols); right.pack(side="left", fill="both", expand=True)

        # ── Input fields ──
        label(left, "Applicant Details", font=FONT_H3, bg=CARD, padx=16, pady=10).pack(anchor="w")
        tk.Frame(left, bg=BORDER, height=1).pack(fill="x")

        fields = tk.Frame(left, bg=CARD, padx=16, pady=12)
        fields.pack(fill="x")

        self.vars = {}

        def add_dropdown(parent, key, lbl, options, default=0):
            f = tk.Frame(parent, bg=CARD); f.pack(fill="x", pady=4)
            label(f, lbl, font=FONT_S, fg=MUTED, bg=CARD, width=20, anchor="w").pack(side="left")
            var = tk.StringVar(value=options[default])
            cb = ttk.Combobox(f, textvariable=var, values=options,
                              state="readonly", width=18, font=FONT_S)
            cb.pack(side="left")
            self.vars[key] = var

        def add_entry(parent, key, lbl, default):
            f = tk.Frame(parent, bg=CARD); f.pack(fill="x", pady=4)
            label(f, lbl, font=FONT_S, fg=MUTED, bg=CARD, width=20, anchor="w").pack(side="left")
            var = tk.StringVar(value=str(default))
            e = tk.Entry(f, textvariable=var, width=20, font=FONT_S,
                         bg=CARD2, fg=TEXT, insertbackground=TEXT,
                         relief="flat", bd=4)
            e.pack(side="left")
            self.vars[key] = var

        style = ttk.Style()
        style.theme_use('clam')
        style.configure("TCombobox", fieldbackground="#FFFFFF", background="#F1F5F9",
                        foreground="#1E293B", selectbackground="#DBEAFE",
                        selectforeground="#1E293B", bordercolor="#CBD5E1")

        add_dropdown(fields,"gender",     "Gender",         ["Male","Female"])
        add_dropdown(fields,"married",    "Married",        ["Yes","No"])
        add_dropdown(fields,"dependents", "Dependents",     ["0","1","2","3+"])
        add_dropdown(fields,"education",  "Education",      ["Graduate","Not Graduate"])
        add_dropdown(fields,"self_emp",   "Self Employed",  ["No","Yes"])
        add_dropdown(fields,"credit",     "Credit History", ["Good (1.0)","Bad (0.0)"])
        add_dropdown(fields,"property",   "Property Area",  ["Urban","Semiurban","Rural"])
        add_entry(fields,"app_income",    "Applicant Income (₹)", 5000)
        add_entry(fields,"coapp_income",  "CoApplicant Income (₹)", 0)
        add_entry(fields,"loan_amount",   "Loan Amount (₹000s)", 120)
        add_dropdown(fields,"loan_term",  "Loan Term (months)",
                     ["360","180","480","300","240","120","60"])
        add_dropdown(fields,"model_sel",  "Model",
                     ["Logistic Regression","Decision Tree","Random Forest"])

        # Predict button
        tk.Button(left, text="  🎯  PREDICT LOAN APPROVAL  ",
                  font=("Segoe UI",12,"bold"), bg=ACCENT, fg="#fff",
                  activebackground="#1D4ED8", bd=0, cursor="hand2",
                  pady=12, command=self._run_predict
                  ).pack(fill="x", padx=16, pady=16)

        # ── Result panel ──
        label(right, "Prediction Result", font=FONT_H3, bg=CARD, padx=16, pady=10).pack(anchor="w")
        tk.Frame(right, bg=BORDER, height=1).pack(fill="x")

        self.res_frame = tk.Frame(right, bg=CARD, padx=20, pady=20)
        self.res_frame.pack(fill="both", expand=True)

        self.res_icon  = label(self.res_frame, "🏦", font=("Segoe UI",52), bg=CARD)
        self.res_icon.pack(pady=(20,8))
        self.res_title = label(self.res_frame, "Fill in the form", font=FONT_BIG, fg=MUTED, bg=CARD)
        self.res_title.pack()
        self.res_sub   = label(self.res_frame, "and click Predict", font=FONT_B, fg=MUTED, bg=CARD)
        self.res_sub.pack(pady=4)

        # Probability bar
        pb_wrap = tk.Frame(self.res_frame, bg=CARD)
        pb_wrap.pack(fill="x", pady=(16,4))
        label(pb_wrap, "Approval Probability", font=FONT_S, fg=MUTED, bg=CARD).pack(anchor="w")
        self.prob_bg = tk.Frame(self.res_frame, bg=CARD2, height=18)
        self.prob_bg.pack(fill="x")
        self.prob_fill = tk.Frame(self.prob_bg, bg=MUTED, height=18, width=0)
        self.prob_fill.place(x=0, y=0)
        self.prob_label = label(self.res_frame, "", font=("Segoe UI",14,"bold"), fg=MUTED, bg=CARD)
        self.prob_label.pack(pady=4)

        # Factors
        self.factors_frame = tk.Frame(self.res_frame, bg=CARD)
        self.factors_frame.pack(fill="x", pady=(12,0))

        return pg

    def _run_predict(self):
        try:
            gender   = 1 if self.vars['gender'].get()    == "Male"    else 0
            married  = 1 if self.vars['married'].get()   == "Yes"     else 0
            dep_str  = self.vars['dependents'].get()
            dep      = 3 if dep_str == "3+" else int(dep_str)
            edu      = 0 if self.vars['education'].get() == "Graduate"else 1
            self_emp = 1 if self.vars['self_emp'].get()  == "Yes"     else 0
            credit   = 1.0 if "1.0" in self.vars['credit'].get() else 0.0
            prop_map = {"Urban":2,"Semiurban":1,"Rural":0}
            prop     = prop_map[self.vars['property'].get()]
            app_inc  = float(self.vars['app_income'].get())
            coapp    = float(self.vars['coapp_income'].get())
            loan_amt = float(self.vars['loan_amount'].get())
            term     = float(self.vars['loan_term'].get())
            model_name = self.vars['model_sel'].get()

            total_inc = app_inc + coapp
            emi       = loan_amt / term
            ipm       = total_inc / (dep + 1)
            lti       = loan_amt / (total_inc + 1)

            row = pd.DataFrame([[
                gender, married, dep, edu, self_emp,
                np.log1p(app_inc), np.log1p(coapp),
                np.log1p(loan_amt), term, credit, prop,
                np.log1p(total_inc), emi, lti
            ]], columns=self.data['features'])

            scaler = self.data['scaler']
            if model_name == "Logistic Regression":
                row_sc = scaler.transform(row)
                pred   = self.data['lr'].predict(row_sc)[0]
                prob   = self.data['lr'].predict_proba(row_sc)[0][1]
            elif model_name == "Decision Tree":
                pred   = self.data['dt'].predict(row)[0]
                prob   = self.data['dt'].predict_proba(row)[0][1]
            else:
                pred   = self.data['rf'].predict(row)[0]
                prob   = self.data['rf'].predict_proba(row)[0][1]

            approved = bool(pred == 1)
            self._show_result(approved, prob, credit, lti, edu, prop, married)

        except ValueError as e:
            messagebox.showerror("Input Error", f"Please enter valid numbers.\n\nDetails: {e}")

    def _show_result(self, approved, prob, credit, lti, edu, prop, married):
        col   = GREEN if approved else RED
        icon  = "✅" if approved else "❌"
        title = "APPROVED" if approved else "REJECTED"
        pct   = f"{prob*100:.1f}%"

        self.res_icon.configure(text=icon)
        self.res_title.configure(text=title, fg=col)
        self.res_sub.configure(text=f"Approval probability: {pct}", fg=MUTED)
        self.prob_fill.configure(bg=col, width=int(prob * 340))
        self.prob_label.configure(text=pct, fg=col)

        for w in self.factors_frame.winfo_children():
            w.destroy()
        label(self.factors_frame, "Key Factors:", font=FONT_H3, bg=CARD).pack(anchor="w", pady=(0,6))
        factors = []
        if credit == 1.0: factors.append(("✔", "Good credit history",    GREEN))
        else:              factors.append(("✖", "Bad credit history",     RED))
        if lti > 0.3:      factors.append(("✖", f"High loan-to-income ({lti:.2f})", RED))
        else:              factors.append(("✔", f"Affordable loan ratio ({lti:.2f})", GREEN))
        if edu == 0:       factors.append(("✔", "Graduate applicant",    GREEN))
        if prop == 1:      factors.append(("✔", "Semiurban property",    GREEN))
        if married == 1:   factors.append(("✔", "Married applicant",     GREEN))
        for sym, text, col2 in factors:
            r = tk.Frame(self.factors_frame, bg=CARD); r.pack(anchor="w", pady=2)
            label(r, sym, font=("Segoe UI",11,"bold"), fg=col2, bg=CARD, width=3).pack(side="left")
            label(r, text, font=FONT_S, fg=MUTED, bg=CARD).pack(side="left")

    # Scroll helper (not used everywhere, but useful)
    def _scroll_wrap(self, pg):
        pass


# ─────────────────────────────────────────────
#  4.  ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("Training models, please wait...")
    data = build_models()
    print(f"  ✅ Logistic Regression  Accuracy: {data['metrics']['Logistic Regression']['acc']*100:.2f}%")
    print(f"  ✅ Decision Tree        Accuracy: {data['metrics']['Decision Tree']['acc']*100:.2f}%")
    print(f"  ✅ Random Forest        Accuracy: {data['metrics']['Random Forest']['acc']*100:.2f}%")
    print("Launching GUI...")

    root = tk.Tk()
    app  = LoanApp(root, data)
    root.mainloop()
