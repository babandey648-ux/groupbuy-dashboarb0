"""GroupBuy Business Analytics Dashboard.  Run from the project folder:  streamlit run Dashboard/app.py"""
from pathlib import Path
import numpy as np, pandas as pd, plotly.express as px, plotly.graph_objects as go, streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent if (HERE.parent / "Dataset").exists() else HERE
DATA = ROOT / "Dataset" if (ROOT / "Dataset").exists() else HERE   # works with folders or with all files uploaded loose
RES = ROOT / "Results" if (ROOT / "Results").exists() else HERE
st.set_page_config("GroupBuy Analytics", layout="wide")
BLUE, ORANGE, AQUA, YELLOW, PINK, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#8a8984"
SEG = {"Bulk Value Seekers": BLUE, "SME Procurement Buyers": ORANGE, "Social Online Shoppers": AQUA, "Cautious Sceptics": YELLOW, "Affluent Convenience Seekers": PINK}
AED = lambda v: f"AED {v/1e6:,.2f}M" if abs(v) >= 1e6 else f"AED {v:,.0f}"
pretty = lambda s: str(s).replace("_", " ")

@st.cache_data
def load():
    f = {n: pd.read_csv(DATA / f"{n}.csv") for n in ["customers", "products", "suppliers", "campaigns", "transactions"]}
    f |= {n: pd.read_csv(RES / f"{n}.csv") for n in ["Customer_Segments", "Cluster_Profiles", "Clustering_K_Selection", "Classification_Metrics", "Classification_Importance", "Predictions", "Lead_Scores", "Association_Rules", "Regression_Metrics", "Regression_Predictions", "Regression_Importance", "Model_Evaluation", "Data_Cleaning_Log", "Model_Validation", "Lead_Gain"]}
    f["campaigns"]["Month"] = pd.to_datetime(f["campaigns"].Start_Date).dt.to_period("M").dt.to_timestamp()
    f["transactions"]["Month"] = pd.to_datetime(f["transactions"].Campaign_Start).dt.to_period("M").dt.to_timestamp()
    return f
D = load(); C, CA, TX, SEGS, PROF = D["customers"], D["campaigns"], D["transactions"], D["Customer_Segments"], D["Cluster_Profiles"]
CLF, REG = D["Classification_Metrics"], D["Regression_Metrics"]
BEST_CLF = CLF[~CLF.Model.str.startswith("Baseline")].sort_values("CV_F1_Train", ascending=False).iloc[0]
BEST_REG = REG[~REG.Model.str.startswith("Baseline")].sort_values("CV_RMSE_Train").iloc[0]

def style(fig, h=340):
    fig.update_layout(height=h, margin=dict(l=8, r=8, t=44, b=8), plot_bgcolor="rgba(0,0,0,0)", legend_title_text="", font_size=13)
    fig.update_xaxes(showgrid=False); fig.update_yaxes(gridcolor="rgba(128,128,128,.18)")
    return fig
def show(fig, h=340): st.plotly_chart(style(fig, h), width="stretch")
def kpis(items):
    for col, (a, b, *h) in zip(st.columns(len(items)), items): col.metric(a, b, help=h[0] if h else None)
def table(df, **k): st.dataframe(df, width="stretch", hide_index=True, **k)
def hbar(s, title, x="Value", color=BLUE):
    d = s.sort_values().reset_index(); d.columns = ["Item", x]; d["Item"] = d.Item.map(pretty)
    return px.bar(d, x=x, y="Item", orientation="h", title=title, color_discrete_sequence=[color]).update_yaxes(title=None)

# ---- live what-if models, trained on project data using the notebook settings
LEAD_N = ["C1_Price_Consciousness", "C2_Willing_To_Wait_For_Deal", "C3_Trust_New_Platforms", "C4_Social_Influence", "C5_Convenience_Priority", "C7_Values_Price_Transparency", "F1_Min_Discount_To_Switch_Pct", "B1_Monthly_Spend_Eligible_AED", "Is_SME"]
LEAD_C = ["B4_Used_Group_Buying_Before", "B5_In_Community_WhatsApp_Buying_Group"]
@st.cache_resource
def whatif():
    pre = ColumnTransformer([("n", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), LEAD_N), ("c", OneHotEncoder(handle_unknown="ignore"), LEAD_C)])
    m1 = Pipeline([("p", pre), ("m", RandomForestClassifier(300, min_samples_leaf=5, class_weight="balanced", random_state=42, n_jobs=-1))]).fit(C[LEAD_N + LEAD_C], (C.Conversion_Segment == "High").astype(int))
    rn = ["Is_SME", "A6_Household_Size", "A7_Monthly_Income_AED", "B1_Monthly_Spend_Eligible_AED", "F1_Min_Discount_To_Switch_Pct", "F2_Max_Wait_Days"]
    m2 = make_pipeline(SimpleImputer(strategy="median"), RandomForestRegressor(300, min_samples_leaf=3, random_state=42, n_jobs=-1)).fit(C[rn], C.G4_Expected_Order_Value_AED)
    return m1, m2, rn

# ---- headline numbers shared across pages
done = CA[CA.Status == "Completed"]; GMV = TX.GMV_AED.sum(); COMM = TX.Commission_AED.sum(); VAR = TX.Variable_Cost_AED.sum(); FIX = 15000 * 12
paid = TX[TX.Paid == 1]; repeat = (paid.groupby("Customer_ID").size() > 1).mean()
m = TX.groupby("Month")[["GMV_AED", "Commission_AED", "Variable_Cost_AED"]].sum().reset_index(); m["Contribution"] = m.Commission_AED - m.Variable_Cost_AED; m["After_Fixed"] = m.Contribution - 15000
seg_c = C.merge(SEGS, left_on="Customer_ID", right_on="Customer_ID")
rules = D["Association_Rules"]; cat_rules = rules[rules.Rule_Type == "Category bundle"]; top_rule = cat_rules.sort_values("lift", ascending=False).iloc[0]
fe_rule = rules[rules.Rule_Type == "Feature pairing"].sort_values("lift", ascending=False).iloc[0]
PS = PROF.set_index("Segment"); core = PS.High_Intent_Share.idxmax()
BE_GMV = 15000 / .05

st.sidebar.title("GroupBuy")
st.sidebar.caption("Collective demand. Better deals.")
page = st.sidebar.radio("Section", ["Business performance", "Sales and revenue", "Customer analytics", "Correlation", "Clustering", "Classification", "Association rules", "Regression", "Insights and decisions", "Model evaluation", "Data and assumptions"], label_visibility="collapsed")
st.sidebar.warning("All data is mock or synthetic. Results show the method, not GroupBuy's real market.")

if page == "Business performance":
    st.title("Business performance")
    st.caption("Simulated Year 1 (Jan to Dec 2027), using the deck's 7% commission, 2% variable cost and AED 15,000 monthly fixed cost.")
    kpis([("GMV", AED(GMV)), ("Commission revenue", AED(COMM)), ("Contribution", AED(COMM - VAR)), ("After fixed costs", AED(COMM - VAR - FIX)), ("Campaign success", f"{len(done)/len(CA):.0%}", f"{len(done)} of {len(CA)} campaigns completed")])
    kpis([("Commitment to paid", f"{TX.Paid.mean():.0%}", "Deck pilot target: at least 50%"), ("Buyer savings", AED(TX.Savings_AED.sum())), ("On-time delivery", f"{done.On_Time.mean():.0%}"), ("Dispute rate", f"{paid.Dispute.mean():.1%}"), ("Repeat buyers", f"{repeat:.0%}")])
    a, b = st.columns([3, 2])
    with a:
        f = go.Figure(go.Bar(x=m.Month, y=m.After_Fixed, marker_color=[ORANGE if v < 0 else AQUA for v in m.After_Fixed], hovertemplate="%{x|%b %Y}<br>AED %{y:,.0f}<extra></extra>"))
        f.update_layout(title="Monthly contribution after fixed costs (AED)"); show(f)
    with b:
        s = CA.Status.value_counts().reset_index(); s.columns = ["Status", "Campaigns"]
        show(px.bar(s.sort_values("Campaigns"), x="Campaigns", y="Status", orientation="h", title="Campaign outcomes", color_discrete_sequence=[BLUE]).update_yaxes(title=None))
    first = m[m.After_Fixed > 0].Month.min()
    st.info(f"Break-even needs about {AED(BE_GMV)} GMV per month (AED 15,000 ÷ 5% contribution). The simulation first passes it in {first:%B %Y}." if pd.notna(first) else f"Break-even needs about {AED(BE_GMV)} GMV per month, which the simulation never reaches in Year 1.")
elif page == "Sales and revenue":
    st.title("Sales and revenue")
    cats = st.multiselect("Categories", sorted(TX.Category.unique()), format_func=pretty, placeholder="All categories")
    T, K = (TX[TX.Category.isin(cats)], CA[CA.Category.isin(cats)]) if cats else (TX, CA)
    kpis([("GMV", AED(T.GMV_AED.sum())), ("Paid orders", f"{T.Paid.sum():,}"), ("Average paid order", AED(T.GMV_AED.sum() / max(T.Paid.sum(), 1))), ("Average net discount", f"{K.Net_Discount_Pct.mean():.1f}%")])
    a, b = st.columns(2)
    with a: show(px.line(T.groupby("Month").GMV_AED.sum().reset_index(), x="Month", y="GMV_AED", markers=True, title="GMV by month (AED)", color_discrete_sequence=[BLUE]).update_traces(line_width=2).update_yaxes(title=None).update_xaxes(title=None))
    with b: show(hbar(T.groupby("Category").GMV_AED.sum(), "GMV by category (AED)", "GMV"), 400)
    K2 = K[K.Status == "Completed"].copy()
    if len(K2) > 3:
        K2["Paid share"] = K2.Paid_Buyers / K2.Committed_Buyers
        show(px.scatter(K2, x="Net_Discount_Pct", y="Paid share", size="GMV_AED", hover_data=["Campaign_ID", "Category"], title="Bigger net discounts convert more committed buyers (completed campaigns)", color_discrete_sequence=[BLUE], labels={"Net_Discount_Pct": "Net discount %"}).update_yaxes(tickformat=".0%"))
elif page == "Customer analytics":
    st.title("Customer analytics")
    st.caption("Mock survey after cleaning: one row per respondent.")
    a, b = st.columns(2); typ = a.multiselect("Respondent type", C.A1_Respondent_Type.unique(), placeholder="All"); area = b.multiselect("Area", sorted(C.A5_Area.unique()), placeholder="All")
    S = C[C.A1_Respondent_Type.isin(typ or C.A1_Respondent_Type.unique()) & C.A5_Area.isin(area or C.A5_Area.unique())]
    if S.empty: st.warning("No respondents match these filters."); st.stop()
    kpis([("Respondents", f"{len(S):,}"), ("Median eligible spend", AED(S.B1_Monthly_Spend_Eligible_AED.median())), ("Median minimum discount", f"{S.F1_Min_Discount_To_Switch_Pct.median():.0f}%"), ("High-intent share", f"{(S.Conversion_Segment=='High').mean():.0%}")])
    a, b = st.columns(2)
    with a: show(hbar(S.filter(like="CAT_").mean().rename(lambda x: x[4:]) * 100, "Category interest (% of respondents)", "%"), 420)
    with b: show(hbar(S.filter(like="FEAT_").mean().rename(lambda x: x[5:]) * 100, "Features wanted (% of respondents)", "%", AQUA), 420)
    a, b = st.columns(2)
    with a: show(hbar(S.H1_Biggest_Frustration_Or_Comment.value_counts().head(8), "Most common comments", "Mentions", GREY), 380)
    with b: show(px.histogram(S, x="F1_Min_Discount_To_Switch_Pct", nbins=30, title="Minimum discount needed to switch (%)", color_discrete_sequence=[BLUE]).update_yaxes(title="Respondents").update_xaxes(title=None), 380)
elif page == "Correlation":
    st.title("Correlation analysis")
    st.caption("Spearman rank correlation on the cleaned survey. Correlation shows association, not cause.")
    COLS = {"A7_Monthly_Income_AED": "Income", "A6_Household_Size": "Household", "B1_Monthly_Spend_Eligible_AED": "Spend", "B2_Online_Orders_Per_Month": "Online orders", "B6_Max_Delivery_Fee_AED": "Max delivery fee", "C1_Price_Consciousness": "Price conscious", "C2_Willing_To_Wait_For_Deal": "Will wait", "C3_Trust_New_Platforms": "Trust", "C4_Social_Influence": "Social influence", "C5_Convenience_Priority": "Convenience", "C7_Values_Price_Transparency": "Transparency", "F1_Min_Discount_To_Switch_Pct": "Min discount", "F2_Max_Wait_Days": "Max wait days", "F3_Membership_WTP_AED_Month": "Membership WTP", "G4_Expected_Order_Value_AED": "Order value", "Conversion_Score_0to10": "Conversion score"}
    typ = st.selectbox("Respondents", ["All", "Individual / household", "Business buyer (SME)"])
    S = C if typ == "All" else C[C.A1_Respondent_Type == typ]
    R = S[list(COLS)].rename(columns=COLS).corr(method="spearman").round(2)
    show(px.imshow(R, text_auto=True, color_continuous_scale="RdBu_r", zmin=-1, zmax=1, aspect="auto", title="Correlation heatmap"), 620)
    a, b = st.columns(2)
    with a:
        s = R["Conversion score"].drop("Conversion score").sort_values().reset_index(); s.columns = ["Variable", "Correlation"]
        show(px.bar(s, x="Correlation", y="Variable", orientation="h", title="What moves with conversion score", color=np.where(s.Correlation < 0, "Negative", "Positive"), color_discrete_map={"Negative": ORANGE, "Positive": BLUE}).update_yaxes(title=None), 470)
    with b:
        names = list(COLS.values()); rev = {v: k for k, v in COLS.items()}
        x = st.selectbox("X variable", names, names.index("Spend")); y = st.selectbox("Y variable", names, names.index("Order value"))
        if x == y: st.info("Choose two different variables.")
        else:
            st.plotly_chart(style(px.scatter(S, x=rev[x], y=rev[y], opacity=.55, labels={rev[x]: x, rev[y]: y}, color_discrete_sequence=[BLUE], title=f"{x} vs {y}: Spearman {R.loc[x, y]:.2f}"), 340), width="stretch")
    conv = R["Conversion score"].drop("Conversion score")
    st.markdown(f"**Reading this:** conversion falls most with **{conv.idxmin()}** ({conv.min():.2f}) and rises most with **{conv.idxmax()}** ({conv.max():.2f}). Spend has a correlation of {conv['Spend']:.2f}, so attitude matters more than purchasing power when targeting buyers.")
elif page == "Clustering":
    st.title("K-Means buyer segments")
    K = D["Clustering_K_Selection"]; a, b = st.columns(2)
    with a: show(px.line(K, x="k", y="inertia", markers=True, title="Elbow method", color_discrete_sequence=[BLUE]).update_traces(line_width=2), 280)
    with b: show(px.line(K, x="k", y="silhouette", markers=True, title="Silhouette score", color_discrete_sequence=[BLUE]).update_traces(line_width=2), 280)
    st.caption(f"k=5 chosen: the elbow, plus a local silhouette peak ({K.set_index('k').silhouette[5]:.3f}). k=2 scores higher but is too coarse to act on. Modest silhouette means the segments overlap.")
    show(px.scatter(SEGS, x="PC1", y="PC2", color="Segment", symbol="Segment", color_discrete_map=SEG, opacity=.75, title="Segments in two dimensions", hover_data=["Customer_ID"]).update_traces(marker_size=7), 460)
    v = PROF[["Segment", "Customers", "Spend", "Min_Discount", "Max_Wait", "Order_Value", "High_Intent_Share"]].sort_values("High_Intent_Share", ascending=False)
    st.subheader("Segment profile and action")
    table(v, column_config={"High_Intent_Share": st.column_config.NumberColumn("High intent", format="percent"), "Spend": st.column_config.NumberColumn("Median spend AED", format="%d"), "Order_Value": st.column_config.NumberColumn("Median order AED", format="%d"), "Min_Discount": st.column_config.NumberColumn("Min discount %", format="%.1f"), "Max_Wait": st.column_config.NumberColumn("Max wait days", format="%.1f")})
    st.markdown("\n".join(f"- **{r.Segment}:** {r.Strategy}" for r in PROF.sort_values("High_Intent_Share", ascending=False).itertuples()))
elif page == "Classification":
    st.title("Lead scoring: who is high intent?")
    table(CLF.drop(columns="CV_F1_Train").round(3))
    st.caption(f"{BEST_CLF.Model} selected by cross-validated F1 on training data. Results are for the 25% held-out test set.")
    P = D["Predictions"]; th = st.slider("Decision threshold: minimum probability to contact a lead", .1, .9, .5, .05)
    h = (P.Probability >= th).astype(int); tn, fp, fn, tp = confusion_matrix(P.Actual_High_Intent, h).ravel()
    pr, re = tp / max(tp + fp, 1), tp / max(tp + fn, 1); f1 = 2 * pr * re / max(pr + re, 1e-9)
    a, b = st.columns([2, 3])
    with a:
        f = px.imshow([[tn, fp], [fn, tp]], text_auto=True, x=["Predicted not high", "Predicted high"], y=["Actual not high", "Actual high"], color_continuous_scale="Blues", title="Confusion matrix").update_layout(coloraxis_showscale=False)
        show(f, 340)
    with b:
        kpis([("Accuracy", f"{(tp+tn)/len(P):.1%}"), ("Precision", f"{pr:.1%}"), ("Recall", f"{re:.1%}"), ("F1", f"{f1:.3f}")])
        st.markdown(f"- **True positive ({tp}):** high-intent lead contacted. Worth the call.\n- **False positive ({fp}):** low-intent lead contacted. Wasted founder time.\n- **False negative ({fn}):** high-intent lead skipped. A lost pilot buyer; costliest during cold start.\n- **True negative ({tn}):** low-intent lead left in low-cost nurture.\n\nPrecision: when we call, how often it is worthwhile. Recall: of all good leads, how many we reach.")
    a, b = st.columns(2)
    with a: show(hbar(D["Classification_Importance"].head(8).set_index("Feature").Importance, "What drives the score (permutation importance)"), 360)
    with b:
        st.subheader("Score a new lead")
        st.caption("Simplified live model trained on all survey rows, using a subset of questions; other answers are held at typical values.")
        m1, _, _ = whatif()
        x = dict(C1_Price_Consciousness=st.slider("Price consciousness", 1, 5, 4), C2_Willing_To_Wait_For_Deal=st.slider("Willing to wait", 1, 5, 3), C3_Trust_New_Platforms=st.slider("Trust in new platforms", 1, 5, 3), C4_Social_Influence=3, C5_Convenience_Priority=st.slider("Convenience priority", 1, 5, 3), C7_Values_Price_Transparency=4, F1_Min_Discount_To_Switch_Pct=st.slider("Minimum discount to switch (%)", 2, 38, 15), B1_Monthly_Spend_Eligible_AED=2500, Is_SME=0, B4_Used_Group_Buying_Before=st.selectbox("Group buying before", C.B4_Used_Group_Buying_Before.unique()), B5_In_Community_WhatsApp_Buying_Group=st.selectbox("In a WhatsApp buying group", ["No", "Yes"]))
        p = m1.predict_proba(pd.DataFrame([x]))[0, 1]; st.metric("Probability of high intent", f"{p:.0%}")
        st.markdown("**Automated action:** " + ("send to founder call list" if p >= th else "add to WhatsApp nurture flow"))
elif page == "Association rules":
    st.title("Apriori: bundles and feature pairings")
    a, b, c_, d = st.columns(4); t = a.selectbox("Rule type", rules.Rule_Type.unique()); s = b.slider("Minimum support", .08, .25, .08, .01); co = c_.slider("Minimum confidence", .4, .9, .5, .05); l = d.slider("Minimum lift", 1.0, 3.0, 1.3, .1)
    R = rules[(rules.Rule_Type == t) & (rules.support >= s) & (rules.confidence >= co) & (rules.lift >= l)].sort_values("lift", ascending=False)
    if R.empty: st.warning("No rules meet these thresholds. Lower one of the sliders.")
    else:
        kpis([("Rules", len(R)), ("Highest lift", f"{R.lift.max():.2f}"), ("Median confidence", f"{R.confidence.median():.0%}")])
        show(px.scatter(R, x="support", y="confidence", size="lift", hover_data=["Antecedent", "Consequent", "lift"], title="Rules by support and confidence (size shows lift)", color_discrete_sequence=[BLUE]))
        table(R[["Antecedent", "Consequent", "support", "confidence", "lift"]].round(3))
        st.caption("Support: share of respondents with the whole combination. Confidence: chance of the right side given the left. Lift above 1: they occur together more than chance.")
elif page == "Regression":
    st.title("Expected order value")
    table(REG.drop(columns="CV_RMSE_Train").round(2))
    st.caption(f"{BEST_REG.Model} selected on training cross-validation RMSE; metrics are from the 20% test set. Order values are in AED.")
    P = D["Regression_Predictions"].assign(Buyer=lambda d: d.Is_SME.map({0: "Individual", 1: "SME"}))
    a, b = st.columns([3, 2])
    with a:
        f = px.scatter(P, x="Actual_AED", y="Predicted_AED", color="Buyer", symbol="Buyer", color_discrete_map={"Individual": BLUE, "SME": ORANGE}, title="Actual vs predicted (the line is a perfect prediction)", hover_data=["Customer_ID"])
        mx = P[["Actual_AED", "Predicted_AED"]].max().max(); f.add_shape(type="line", x0=0, y0=0, x1=mx, y1=mx, line=dict(color=GREY, width=1)); show(f, 420)
    with b:
        show(hbar(D["Regression_Importance"].head(6).set_index("Feature").Importance, "Main drivers"), 260)
        for g, d in P.groupby("Buyer"): st.metric(f"{g} mean absolute error", AED(d.Error_AED.abs().mean()))
    st.subheader("Plan a campaign")
    _, m2, rn = whatif(); a, b, c_, d = st.columns(4)
    sp = a.number_input("Buyer's eligible monthly spend (AED)", 300, 15000, 2500, 100); inc = b.number_input("Monthly income (AED)", 3000, 150000, 18000, 1000); n = c_.number_input("Expected paying buyers", 5, 300, 40); sme = d.selectbox("Buyer type", ["Individual", "SME"])
    v = m2.predict(pd.DataFrame([[int(sme == "SME"), 4, inc, sp, 15, 6]], columns=rn))[0]
    kpis([("Predicted order value", AED(v)), ("Projected campaign GMV", AED(v * n)), ("Commission at 7%", AED(v * n * .07)), ("Contribution at 5%", AED(v * n * .05))])
elif page == "Insights and decisions":
    st.title("From analysis to decisions")
    cwa = C.merge(SEGS)
    t = st.tabs(["Clustering", "Classification", "Association rules", "Regression", "Recommendations"])
    def q(rows):
        for k, v in rows: st.markdown(f"**{k}** {v}")
    with t[0]: q([("Problem.", "One message cannot suit everyone."), ("Why this algorithm.", "There are no segment labels to learn from, and K-Means is simple to interpret."), ("Data.", "13 behavioural variables from 1,200 survey respondents; intent answers excluded."), ("How it works.", "Each respondent joins the nearest of five centres; centres move until stable."), ("Output.", "A segment for each customer."), ("What we learned.", f"{core} has the highest high-intent share ({PS.High_Intent_Share[core]:.0%}). Affluent Convenience Seekers have just {PS.High_Intent_Share['Affluent Convenience Seekers']:.0%} and want a {PS.Min_Discount['Affluent Convenience Seekers']:.0f}% average discount."), ("Decision.", f"Launch to {core} first, and set SME campaigns apart: their median order is {AED(PS.Order_Value['SME Procurement Buyers'])}."), ("Automation.", "Assign new sign-ups to the nearest centre, then route them to that segment's WhatsApp flow."), ("Limitations.", "Silhouette of about 0.24 shows overlap; mock survey answers may not match behaviour.")])
    with t[1]: q([("Problem.", "A small team cannot follow up every lead; the deck targets 30 pilot commitments."), ("Why this algorithm.", "Tree ensembles capture interactions; logistic regression and a single tree serve as simpler comparisons."), ("Data.", "52 survey fields; intent answers, the score, comments, gender and nationality excluded."), ("How it works.", "Many decision trees vote on the probability of High intent."), ("Output.", "A probability and yes/no decision for each lead."), ("What we learned.", f"{BEST_CLF.Model} found {BEST_CLF.TP} of {BEST_CLF.TP+BEST_CLF.FN} high-intent test leads (recall {BEST_CLF.Recall:.0%}) at {BEST_CLF.Precision:.0%} precision. Baseline accuracy is already {CLF.Accuracy[0]:.0%}, so accuracy alone misleads."), ("Decision.", "Call scored leads first; keep the threshold low during cold start because a missed buyer costs more than a wasted message."), ("Automation.", "Sign-up form → score → founder call list or nurture flow."), ("Limitations.", "Predicts stated intent, not payment. Retrain on real pilot outcomes.")])
    with t[2]: q([("Problem.", "Delivery costs erode savings; combining categories could spread them."), ("Why this algorithm.", "Apriori is built for finding co-occurring items in yes/no baskets."), ("Data.", "12 category and 13 feature selections per respondent."), ("How it works.", "It keeps combinations above minimum support, then derives rules."), ("Output.", "Rules with support, confidence and lift."), ("What we learned.", f"{pretty(top_rule.Antecedent)} → {pretty(top_rule.Consequent)} has lift {top_rule.lift:.2f}. In features, {pretty(fe_rule.Antecedent)} → {pretty(fe_rule.Consequent)} has lift {fe_rule.lift:.2f}."), ("Decision.", "Run family grocery bundles on a shared delivery; package building pooling, consolidated delivery and WhatsApp updates as one offer."), ("Automation.", "When a buyer commits to one category, suggest the highest-lift open campaign."), ("Limitations.", "Shows stated interest overlap, not purchases or causation.")])
    with t[3]: q([("Problem.", "Target quantities depend on how much each buyer will order."), ("Why this algorithm.", "Linear regression gives a transparent benchmark; a random forest handles the different SME and individual patterns."), ("Data.", "17 survey variables; intent excluded."), ("How it works.", "Averages many trees' estimates of order value."), ("Output.", "Predicted AED order value."), ("What we learned.", f"{BEST_REG.Model}: R² {BEST_REG.R2:.2f}, MAE {AED(BEST_REG.MAE)}. Eligible spend is the leading driver. SME errors are far larger."), ("Decision.", "Forecast individual and SME demand separately when setting campaign targets."), ("Automation.", "Expected paying buyers × predicted value → suggested target quantity sent to suppliers."), ("Limitations.", "Stated order values, and few SMEs in the test set.")])
    with t[4]:
        st.markdown(f"""1. **Launch with {core}.** Highest high-intent share among segments; suit staples and household campaigns.
2. **Treat SMEs as a separate track.** Median order {AED(PS.Order_Value['SME Procurement Buyers'])}, and they accept a {PS.Min_Discount['SME Procurement Buyers']:.0f}% minimum discount.
3. **Design for a net discount at or above the typical {C.F1_Min_Discount_To_Switch_Pct.median():.0f}% minimum.** Simulated offers average {CA.Net_Discount_Pct.mean():.1f}%, so many buyers would decline.
4. **Lead with trust features.** {pretty(C.E2_Most_Important_Feature.mode()[0])} is the feature most often named as most important in the survey.
5. **Revisit the financial plan.** The deck's break-even is about 15 orders of AED 20,000, or {AED(BE_GMV*12)} GMV a year. Its Year 1 GMV is AED 1.2M, which would not cover fixed costs at these rates.""")
elif page == "Model evaluation":
    st.title("Model evaluation")
    table(D["Model_Evaluation"])
    V = D["Model_Validation"]
    st.subheader("Is it right? Repeated validation and stability checks")
    table(V)
    show(px.line(D["Lead_Gain"].assign(Scored=lambda d: d.Scored * 100, Random=lambda d: d.Random * 100, Contacted=lambda d: d.Share_Contacted * 100), x="Contacted", y=["Scored", "Random"], color_discrete_map={"Scored": BLUE, "Random": GREY}, title="Is it useful? High-intent leads found (%) by leads contacted (%)", labels={"value": "Found (%)", "Contacted": "Contacted (%)"}).update_traces(line_width=2), 360)
    st.markdown(f"""**How reliable are these results?**
- **Classification:** F1 of {BEST_CLF.F1:.2f} is well above the baseline's zero, yet logistic regression scored higher on this particular test set. The two are close, so the choice between them is not clear-cut.
- **Label check:** with shuffled labels, ROC-AUC falls to about 0.49, which supports the absence of target leakage in the pipeline.
- **Regression:** R² of {BEST_REG.R2:.2f} comes mostly from individual buyers; SME predictions are much weaker.
- **Clustering:** silhouette is modest, so treat segments as working groups.
- **Overall:** models learn patterns built into mock data. High scores do not prove real-world accuracy.

**Improvements:** retrain on actual pilot payments, add time-based validation, tune the threshold using real costs, and collect more SME responses.""")
else:
    st.title("Data and assumptions")
    st.warning("The survey is labelled mock data. Campaigns, transactions, products and suppliers are synthetic and generated in Notebook 01. Do not present them as market evidence.")
    table(pd.DataFrame([dict(Table=n, Rows=len(D[n]), Columns=D[n].shape[1], Source=s) for n, s in [("customers", "Cleaned mock survey"), ("products", "Synthetic"), ("suppliers", "Synthetic, fictional names"), ("campaigns", "Synthetic, driven by survey"), ("transactions", "Synthetic, driven by survey")]]))
    st.subheader("Cleaning log"); table(D["Data_Cleaning_Log"])
    st.subheader("Data dictionary: why each column is required"); table(pd.read_csv(DATA / "Data_Dictionary.csv"))
    st.markdown("**From the deck:** 7% commission, 2% variable cost, AED 15,000 monthly fixed cost, cancellation rule.\n\n**Assumed for the simulation:** 120 campaigns, SKU prices, bulk-discount ceilings, delivery at 1.2–4% of price, supplier reliability, and buyer payment behaviour.")
