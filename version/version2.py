import streamlit as st
import joblib
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
from dateutil.relativedelta import relativedelta

# 1. 페이지 설정
st.set_page_config(page_title="Adidas Strategic Simulator v2", layout="wide")

# 2. 자산 로드
@st.cache_resource
def load_assets():
    # 모델 패키지 로드 (VSC 내 models/ 폴더 확인)
    return joblib.load('models/adidas_web_model.pkl')

pkg = load_assets()

# 3. 매장 및 제품 카테고리 정의
store_types = {
    "🏬 대형 플래그십": {"cluster": 2, "color": "#1f77b4", "price": 55},
    "💰 프리미엄 매장": {"cluster": 3, "color": "#2ca02c", "price": 46},
    "🛒 박리다매 매장": {"cluster": 0, "color": "#ff7f0e", "price": 37},
    "📍 지역 표준 매장": {"cluster": 1, "color": "#d62728", "price": 31}
}

product_categories = list(pkg['le_product'].classes_)

# --- 사이드바 ---
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/2/20/Adidas_Logo.svg", width=80)
st.sidebar.title("Partner Portal")

selected_type = st.sidebar.selectbox("🏠 매장 유형", options=list(store_types.keys()))
selected_prod_cat = st.sidebar.selectbox("👟 제품 카테고리", options=product_categories)

# 1개월 단위 촘촘한 설정 (1~12개월)
forecast_horizon = st.sidebar.slider("📅 예측 기간 (개월)", 1, 12, 6)
growth_input = st.sidebar.slider("📈 목표 성장률 설정 (%)", -50, 50, 0)

# --- 데이터 계산 엔진 ---
def get_simulation_data(cluster, product, horizon, growth):
    current_date = datetime.now()
    data = []
    
    # 모델 입력용 인코딩 (대표값 사용)
    ret_enc = pkg['le_retailer'].transform([pkg['le_retailer'].classes_[0]])[0]
    prod_enc = pkg['le_product'].transform([product])[0]
    
    # [1] 현재 달 데이터 (Base)
    curr_feat = pd.DataFrame([[ret_enc, prod_enc, current_date.month, cluster]], columns=pkg['features'])
    curr_pred = pkg['model'].predict(curr_feat)[0]
    data.append({
        "Month": "현재 (" + current_date.strftime("%m월") + ")",
        "Units": int(curr_pred),
        "Type": "Actual"
    })
    
    # [2] 미래 예측 데이터 (1개월 단위)
    for i in range(1, horizon + 1):
        future = current_date + relativedelta(months=i)
        feat = pd.DataFrame([[ret_enc, prod_enc, future.month, cluster]], columns=pkg['features'])
        pred = pkg['model'].predict(feat)[0] * (1 + growth / 100)
        data.append({
            "Month": future.strftime("%Y-%m"),
            "Units": int(pred),
            "Type": "Forecast"
        })
    return pd.DataFrame(data)

# --- 메인 화면 레이아웃 ---
tab1, tab2 = st.tabs(["🚀 시뮬레이션 엔진", "📰 마켓 인텔리전스 뉴스"])

with tab1:
    df = get_simulation_data(store_types[selected_type]['cluster'], selected_prod_cat, forecast_horizon, growth_input)
    st.subheader(f"📊 {selected_type} 시나리오 분석: {selected_prod_cat}")
    
    # [수정] Plotly 인터랙티브 그래프 (막대 + 꺾은선)
    fig = go.Figure()
    
    # 눈금 고정 기준값 (성장률 0% 기준의 2.5배)
    y_max = get_simulation_data(store_types[selected_type]['cluster'], selected_prod_cat, forecast_horizon, 0)['Units'].max() * 2.5

    # 1. 현재 달 (회색 막대)
    curr_df = df[df['Type'] == "Actual"]
    fig.add_trace(go.Bar(
        x=curr_df['Month'], y=curr_df['Units'],
        name='현재 기준', marker_color='lightgrey',
        hovertemplate='%{x}<br>판매량: %{y} Pcs<extra></extra>'
    ))

    # 2. 미래 예측 (색상 막대)
    fore_df = df[df['Type'] == "Forecast"]
    fig.add_trace(go.Bar(
        x=fore_df['Month'], y=fore_df['Units'],
        name='미래 예측(수량)', marker_color=store_types[selected_type]['color'],
        hovertemplate='%{x}<br>예측 수량: %{y} Pcs<extra></extra>'
    ))

    # 3. 꺾은선 그래프 추가 (추세 강조)
    fig.add_trace(go.Scatter(
        x=df['Month'], y=df['Units'],
        mode='lines+markers', name='판매 추세',
        line=dict(color='black', width=2),
        marker=dict(size=8),
        hovertemplate='%{x}<br>수치: %{y} Pcs<extra></extra>'
    ))

    # 레이아웃 설정 (눈금 고정)
    fig.update_layout(
        yaxis=dict(range=[0, y_max], title="판매량 (Pcs)"),
        xaxis=dict(title="타임라인 (월 단위)"),
        hovermode="x unified",
        template="plotly_white",
        height=500,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    st.plotly_chart(fig, use_container_width=True)

    # 지표 요약
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("타겟 제품", selected_prod_cat)
    with c2:
        total_units = df[df['Type']=="Forecast"]['Units'].sum()
        st.metric("총 예상 판매량", f"{total_units:,} Pcs")
    with c3:
        total_rev = total_units * store_types[selected_type]['price']
        st.metric("총 예상 매출", f"${total_rev:,}")

with tab2:
    st.subheader("🇺🇸 US Market Intelligence News")
    
    # 향후 크롤링 연동을 위한 뉴스 데이터 셋
    news_feed = [
        {"title": "Adidas North America Revenue Jumps 15% in Q1", "date": "2026-02-24", "sentiment": 0.8},
        {"title": "Potential Supply Chain Slowdown due to Port Congestion", "date": "2026-02-20", "sentiment": -0.5},
        {"title": "Rising Demand for 'Terrace' Footwear Collection in US Metro Areas", "date": "2026-02-15", "sentiment": 0.6},
        {"title": "US Consumer Spending Shows Resilience Despite Inflation", "date": "2026-02-10", "sentiment": 0.3}
    ]

    # 시장 지수 계산 (NLP 연동 뼈대)
    avg_score = sum(n['sentiment'] for n in news_feed) / len(news_feed)
    
    col_n1, col_n2 = st.columns([2, 1])
    
    with col_n1:
        st.markdown("### 📣 주요 실시간 뉴스")
        for news in news_feed:
            icon = "📈" if news['sentiment'] > 0 else "📉"
            st.write(f"{icon} **{news['title']}**")
            st.caption(f"발행일: {news['date']} | 영향도: {'긍정' if news['sentiment'] > 0 else '부정'}")
            st.divider()

    with col_n2:
        st.markdown("### 📊 AI 시장 감성 분석")
        sentiment_label = "낙관적 (Bullish)" if avg_score > 0.2 else "주의 (Cautious)"
        st.progress(int((avg_score + 1) / 2 * 100), text=f"시장 분위기: {sentiment_label}")
        st.info(f"현재 시장 감성 점수는 **{avg_score:.2f}**입니다. 0.2 이상일 경우 적극적인 목표 상향을 권장합니다.")
        
        st.divider()
        st.markdown("#### 📝 전략 비고")
        st.text_area("기사를 참고하여 발주 및 마케팅 메모를 남기세요.")