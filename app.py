# ============================================
# AI Investment System (Fixed Final Version)
# ============================================

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="Mutual Fund Investment with AI Reasoning", layout="wide")

st.title("🧠 Mutual Fund Investment with AI Reasoning")
# -----------------------------
# DISCLAIMER
# -----------------------------
st.warning(
    "⚠️ Mutual Funds are subject to market risks. "
    "This AI system provides generated insights and should not be considered financial advice. "
    "Please consult your financial advisor before making investment decisions."
)
# -----------------------------
# USER INPUTS
# -----------------------------
st.sidebar.header("User Inputs")

initial_capital = st.sidebar.number_input("Initial Investment (₹)", value=100000)
monthly_expense = st.sidebar.number_input("Monthly Withdrawal (₹)", value=2000)
years = st.sidebar.slider("Investment Duration (Years)", 1, 20, 10)

risk_profile = st.sidebar.selectbox(
    "Risk Profile",
    ["Conservative", "Balanced", "Aggressive"]
)

# -----------------------------
# LOAD DATA
# -----------------------------
@st.cache_data
def load_data():
    df = yf.download("^NSEI", period="10y", interval="1d")

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df[['Close']].copy()
    df['Close'] = df['Close'].astype(float)
    df.dropna(inplace=True)
    return df
    if df.empty:
        st.error("⚠️ Failed to fetch market data. Please try again.")
        st.stop()
data = load_data()

# -----------------------------
# FEATURES
# -----------------------------
data['Return'] = data['Close'].pct_change()
data['MA50'] = data['Close'].rolling(50).mean()
data['MA200'] = data['Close'].rolling(200).mean()
data['Volatility'] = data['Return'].rolling(20).std()

delta = data['Close'].diff()
gain = (delta.where(delta > 0, 0)).rolling(14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
rs = gain / loss
data['RSI'] = 100 - (100 / (1 + rs))

data.dropna(inplace=True)

# -----------------------------
# VISUALS
# -----------------------------
st.subheader("📈 NIFTY Trend")
st.line_chart(data['Close'])

st.subheader("📊 Price vs MA50 vs MA200")
st.line_chart(data[['Close', 'MA50', 'MA200']].tail(200))

st.subheader("⚡ RSI Indicator")
st.line_chart(data['RSI'].tail(200))

# -----------------------------
# ML MODEL
# -----------------------------
# -----------------------------
# ML MODEL (SAFE VERSION)
# -----------------------------
features = ['Return','MA50','MA200','Volatility','RSI']

# Safety check
if data.empty or len(data) < 250:
    st.error("⚠️ Not enough data to train model. Please refresh or try later.")
    st.stop()

X = data[features]
y = (data['Close'] > data['MA50']).astype(int)

# Drop any remaining NaN safely
X = X.dropna()
y = y.loc[X.index]

# Final safety check
if len(X) < 10:
    st.error("⚠️ Insufficient cleaned data for ML model.")
    st.stop()

split = int(len(X)*0.8)

X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

accuracy = accuracy_score(y_test, model.predict(X_test))

prediction = model.predict(X.iloc[-1:].values)[0]
market_condition = "Bull" if prediction == 1 else "Bear"

# -----------------------------
# SIGNALS
# -----------------------------
latest = data.iloc[-1]

st.subheader("📌 Current Market Signals")

c1,c2,c3,c4 = st.columns(4)
c1.metric("RSI", round(latest['RSI'],2))
c2.metric("Volatility", round(latest['Volatility'],4))
c3.metric("MA50", round(latest['MA50'],2))
c4.metric("MA200", round(latest['MA200'],2))

# -----------------------------
# PREDICTION
# -----------------------------
st.subheader("📡 AI Market Prediction")

if market_condition == "Bull":
    st.success("🚀 Bull Market Detected")
else:
    st.warning("⚠️ Bear Market Detected")

st.write(f"Model Accuracy: {round(accuracy*100,2)}%")

# -----------------------------
# ALLOCATION
# -----------------------------
def allocation_agent(risk, market):
    if risk=="Conservative":
        base={"Corporate":40,"Balanced":25,"Flexi":15,"Index":10,"MidSmall":10}
    elif risk=="Balanced":
        base={"Corporate":25,"Balanced":20,"Flexi":20,"Index":20,"MidSmall":15}
    else:
        base={"Corporate":10,"Balanced":10,"Flexi":25,"Index":25,"MidSmall":30}

    if market=="Bear":
        base["Corporate"]+=10
        base["Flexi"]-=5
        base["MidSmall"]-=5
    else:
        base["MidSmall"]+=10
        base["Corporate"]-=5
        base["Balanced"]-=5

    total=sum(base.values())
    return {k:v/total for k,v in base.items()}

ai_allocation = allocation_agent(risk_profile, market_condition)

# -----------------------------
# SIDEBAR ALLOCATION
# -----------------------------
st.sidebar.subheader("🤖 AI Suggested Allocation")
for k,v in ai_allocation.items():
    st.sidebar.write(f"{k}: {round(v*100,1)}%")

# -----------------------------
# MANUAL OVERRIDE
# -----------------------------
st.sidebar.subheader("⚙️ Manual Override")
use_manual = st.sidebar.checkbox("Enable Manual Allocation")

if use_manual:
    corp = st.sidebar.slider("Corporate %",0,100,int(ai_allocation["Corporate"]*100))
    bal = st.sidebar.slider("Balanced %",0,100,int(ai_allocation["Balanced"]*100))
    flex = st.sidebar.slider("Flexi %",0,100,int(ai_allocation["Flexi"]*100))
    ind = st.sidebar.slider("Index %",0,100,int(ai_allocation["Index"]*100))
    mid = st.sidebar.slider("MidSmall %",0,100,int(ai_allocation["MidSmall"]*100))

    total = corp+bal+flex+ind+mid

    if total != 100:
        st.sidebar.error(f"Total must be 100% (Current: {total}%)")
        st.stop()

    final_allocation = {
        "Corporate":corp/100,
        "Balanced":bal/100,
        "Flexi":flex/100,
        "Index":ind/100,
        "MidSmall":mid/100
    }
else:
    final_allocation = ai_allocation

st.sidebar.subheader("✅ Final Allocation")
for k,v in final_allocation.items():
    st.sidebar.write(f"{k}: {round(v*100,1)}%")

# -----------------------------
# RISK + DEVIATION (NOW FIXED)
# -----------------------------
st.subheader("⚖️ Portfolio Risk Meter")

equity_exposure = (
    final_allocation["Flexi"] +
    final_allocation["Index"] +
    final_allocation["MidSmall"]
)

equity_percent = equity_exposure * 100

if equity_percent < 40:
    risk_level = "🟢 Low"
elif equity_percent < 70:
    risk_level = "🟡 Medium"
else:
    risk_level = "🔴 High"

c1,c2 = st.columns(2)
c1.metric("Equity Exposure", f"{round(equity_percent,1)}%")
c2.metric("Risk Level", risk_level)

# Deviation
st.subheader("⚠️ AI Deviation")

dev_data=[]
for f in ai_allocation:
    dev_data.append({
        "Fund":f,
        "AI %":round(ai_allocation[f]*100,1),
        "Your %":round(final_allocation[f]*100,1),
        "Diff":round((final_allocation[f]-ai_allocation[f])*100,1)
    })

dev_df=pd.DataFrame(dev_data)
st.dataframe(dev_df)

# -----------------------------
# SIMULATION (UNCHANGED)
# -----------------------------
returns = {"Corporate":6,"Balanced":8,"Flexi":12,"Index":11,"MidSmall":15}

portfolio = {k:initial_capital*v for k,v in final_allocation.items()}
initial_portfolio = portfolio.copy()

history=[]
depletion_tracker = {k:None for k in portfolio}

months = years*12

for m in range(months):

    for k in portfolio:
        portfolio[k]*=(1+returns[k]/100/12)

    remaining = monthly_expense

    for k in ["Corporate","Balanced","Flexi","Index","MidSmall"]:
        if remaining<=0: break

        if portfolio[k]>=remaining:
            portfolio[k]-=remaining
            remaining=0
        else:
            remaining-=portfolio[k]
            portfolio[k]=0

        if portfolio[k]==0 and depletion_tracker[k] is None:
            depletion_tracker[k]=m+1

    history.append({"Month":m+1,**portfolio,"Total":sum(portfolio.values())})

df = pd.DataFrame(history)

# -----------------------------
# RESULTS
# -----------------------------
st.subheader("📈 Portfolio Growth")
st.line_chart(df.set_index("Month")["Total"])

#st.subheader("📦 Final Portfolio")


# -----------------------------
# CLEAN FINAL PORTFOLIO VIEW
# -----------------------------
st.subheader("📦 Final Portfolio")

final = df.iloc[-1]

portfolio_df = pd.DataFrame({
    "Fund": ["Corporate","Balanced","Flexi","Index","MidSmall"],
    "Final Value (₹)": [
        final["Corporate"],
        final["Balanced"],
        final["Flexi"],
        final["Index"],
        final["MidSmall"]
    ]
})

st.dataframe(portfolio_df.round(2))

# Show Total separately (better UI)
st.metric("💰 Total Portfolio Value", f"₹{round(final['Total'],2)}")

# -----------------------------
# MONTHLY FUND-WISE TRACKING
# -----------------------------
st.subheader("📊 Monthly Fund Tracking")

# Optional: limit rows for performance
st.dataframe(df.reset_index(drop=True), hide_index=True)

# ============================================
# END
# ============================================

st.markdown("---")
st.caption(
    "⚠️ Disclaimer: Mutual Funds are subject to market risks. "
    "This AI-generated system is for educational purposes only. "
    "Consult a financial advisor before investing."
)
