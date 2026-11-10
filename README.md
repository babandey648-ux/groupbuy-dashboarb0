# GroupBuy Dubai: Startup Data Analytics Project

GroupBuy pools buyers' demand, makes suppliers bid against it, and lets buyers approve the final price. This project applies four algorithms to decide whom to target, what to bundle, how large orders will be, and what can be automated.

## Run the dashboard
Requires Python 3.10 or later. From this folder:
```
python -m venv .venv
.venv\Scripts\activate        (Windows)   |   source .venv/bin/activate   (Mac/Linux)
pip install -r requirements.txt
streamlit run Dashboard/app.py
```
It opens at http://localhost:8501. To open the notebooks as well, run `pip install notebook`. Run the command from this folder, or the theme settings are not applied.

## Business problems and algorithms
| Algorithm | Business question | Data | Automated decision |
|---|---|---|---|
| K-Means | Which buyer groups exist? | 13 survey behaviours | Route each sign-up to a segment flow |
| Random Forest, Logistic Regression, Decision Tree | Which leads are high intent? | 52 non-intent fields | Founder call list or nurture flow |
| Apriori | What should be pooled together? | Category and feature selections | Suggest a related open campaign |
| Random Forest, Linear Regression | How large will an order be? | 17 non-intent fields | Suggest campaign target quantity |

## Contents
- `Dataset/`: raw mock survey, cleaned `customers.csv`, synthetic `products`, `suppliers`, `campaigns`, `transactions`
- `Notebooks/`: seven executed notebooks; run them in order to regenerate everything
- `Dashboard/app.py`: Streamlit dashboard
- `Results/`: metrics, predictions, rules, segments, figures
- `Screenshots/`: genuine captures of the running dashboard
- `Reports/`: draft report in Word and PDF

## Data honesty
The survey file is labelled mock, and its dates (Nov–Dec 2026) are illustrative. Campaigns, transactions, products and suppliers are synthetic, generated in Notebook 01 with stated assumptions. The deck supplies 7% commission, 2% variable cost and AED 15,000 fixed cost. Results demonstrate method and do not validate the market.

## Leakage controls
`Conversion_Score` is almost entirely a weighted sum of G1–G3 (R² about 0.99), so those fields, the score and comments are excluded from predictors. Imputation and scaling are fitted on training data only.

The dashboard reads saved results. Its "Score a new lead" and "Plan a campaign" tools train simplified live models when first opened, so they may take a few seconds.
