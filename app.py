import streamlit as st
import joblib
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
from dateutil.relativedelta import relativedelta

# 1. 페이지 설정 및 디자인 커스텀 (상단 여백 최소화)
st.set_page_config(page_title="Adidas Intelligence Dashboard", layout="wide")

st.markdown("""
    <style>
    /* 상단 여백 및 불필요한 요소 제거 */
    .block-container { padding-top: 1rem !important; padding-bottom: 0rem !important; }
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    header { visibility: hidden; }
    
    /* 사이드바 너비 조절 */
    [data-testid="stSidebar"] { width: 220px !important; }
    
    /* 뉴스 폰트 조절 */
    .small-font { font-size: 13.5px !important; line-height: 1.4; margin-bottom: 8px; }
    </style>
    """, unsafe_allow_html=True)

# 2. 자산 로드
@st.cache_resource
def load_assets():
    # 모델 파일 경로가 정확한지 확인하세요 (models/ 폴더 내)
    return joblib.load('models/adidas_web_model.pkl')

pkg = load_assets()

# 3. 매장 및 제품 정의
store_types = {
    "🏬 대형 플래그십": {"cluster": 2, "color": "#1f77b4", "price": 55},
    "💰 프리미엄 매장": {"cluster": 3, "color": "#2ca02c", "price": 46},
    "🛒 박리다매 매장": {"cluster": 0, "color": "#ff7f0e", "price": 37},
    "📍 지역 표준 매장": {"cluster": 1, "color": "#d62728", "price": 31}
}
product_categories = list(pkg['le_product'].classes_)

# --- 가상 뉴스 데이터베이스 (최신순 3개 필터링용) ---
news_db = [
    {"date": "2026-02-24", "title": "뉴욕/LA 플래그십 매장 '삼바' 재고 품귀", "sentiment": 0.8, "tags": [2, "SF"]},
    {"date": "2026-02-23", "title": "아디다스 의류(AP) 북미 매출 20% 급등", "sentiment": 0.9, "tags": ["AP"]},
    {"date": "2026-02-22", "title": "물류 대란으로 인한 운동화 입고 2주 지연", "sentiment": -0.6, "tags": [1, 0, "AF", "SF"]},
    {"date": "2026-02-21", "title": "고금리 영향으로 프리미엄 소비 심리 위축", "sentiment": -0.4, "tags": [3]},
    {"date": "2026-02-20", "title": "러닝화 신제품 'Adizero' 마케팅 캠페인 시작", "sentiment": 0.7, "tags": ["SF"]}
]

# --- 사이드바 제어판 ---
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/2/20/Adidas_Logo.svg", width=50)
    st.header("Control Panel")
    selected_type = st.selectbox("🏠 매장 유형", options=list(store_types.keys()))
    selected_prod_cat = st.sidebar.selectbox("👟 제품군", options=product_categories)
    forecast_horizon = st.sidebar.slider("📅 예측(월)", 1, 12, 6)
    growth_input = st.sidebar.slider("📈 추가 성장률(%)", -50, 50, 0)

# --- 데이터 엔진 (증감율 계산 로직 포함) ---
def get_simulation_data(cluster, product, horizon, growth):
    current_date = datetime.now()
    data = []
    ret_enc = pkg['le_retailer'].transform([pkg['le_retailer'].classes_[0]])[0]
    prod_enc = pkg['le_product'].transform([product])[0]
    
    prev_units = None
    for i in range(0, horizon + 1):
        target = current_date + relativedelta(months=i)
        feat = pd.DataFrame([[ret_enc, prod_enc, target.month, cluster]], columns=pkg['features'])
        pred = pkg['model'].predict(feat)[0]
        if i > 0: pred *= (1 + growth / 100)
        
        current_units = int(pred)
        
        # [증감율 계산] 소수점 둘째자리에서 반올림하여 첫째자리까지 표시
        change_pct = 0.0
        if prev_units is not None and prev_units != 0:
            change_pct = round(((current_units - prev_units) / prev_units) * 100, 1)
        
        data.append({
            "Month": "현재" if i == 0 else target.strftime("%y년 %m월"),
            "Units": current_units,
            "Change": change_pct,
            "Type": "Actual" if i == 0 else "Forecast"
        })
        prev_units = current_units
    return pd.DataFrame(data)

df = get_simulation_data(store_types[selected_type]['cluster'], selected_prod_cat, forecast_horizon, growth_input)

# --- 본문 레이아웃 ---
tab1, tab2 = st.tabs(["🚀 시뮬레이션 대시보드", "📰 전체 뉴스 관리"])

with tab1:
    st.subheader(f"📊 {selected_type} 시나리오 분석: {selected_prod_cat}")
    
    col_left, col_right = st.columns([1.6, 1])


    with col_left:
        # 그래프 영역
        y_max = df['Units'].max() * 2.2
        fig = go.Figure()

        # 막대 그래프 (호버 시 매출량 + 증감율 표시)
        fig.add_trace(go.Bar(
            x=df['Month'], 
            y=df['Units'], 
            customdata=df['Change'],
            marker_color=[store_types[selected_type]['color'] if t=="Forecast" else "lightgrey" for t in df['Type']], 
            name="판매량",
            hovertemplate="<b>%{x}</b><br>매출량: %{y:,} Pcs<br>전달대비: %{customdata:+.1f}%<extra></extra>"
        ))
        
        # 추세선 (선 그래프)
        fig.add_trace(go.Scatter(
            x=df['Month'], y=df['Units'], 
            mode='lines+markers', 
            line=dict(color='black', width=3),
            hoverinfo="skip" # 선은 호버 생략하여 막대에 집중
        ))

        fig.update_layout(
            height=460, 
            margin=dict(l=0, r=0, t=10, b=0),
            yaxis=dict(range=[0, y_max], tickfont=dict(size=14), title="수량 (Pcs)"),
            xaxis=dict(tickfont=dict(size=14)),
            hoverlabel=dict(bgcolor="white", font_size=20, font_family="Malgun Gothic"),
            hovermode="closest", 
            template="plotly_white", 
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

    with col_right:
        # 1. 주요 지표 요약
        st.markdown("##### 📈 핵심 예측 지표")
        t_units = df[df['Type']=="Forecast"]['Units'].sum()
        t_rev = t_units * store_types[selected_type]['price']
        
        c1, c2 = st.columns(2)
        c1.metric("📦 총 예상 판매", f"{t_units:,} Pcs")
        c2.metric("💵 총 예상 매출", f"${t_rev:,}")
        st.divider()

        # 2. 관련 뉴스 자동 매칭 (최신순 최대 3개)
        st.markdown("##### 🔍 맞춤 타겟 뉴스 (최신 3)")
        rel_news = [n for n in news_db if store_types[selected_type]['cluster'] in n['tags'] or selected_prod_cat in n['tags']]
        display_news = rel_news[:3] # 최신 3개만 필터링
        
        if not display_news:
            st.caption("현재 조건과 관련된 최신 뉴스가 없습니다.")
        for n in display_news:
            color = "#1E88E5" if n['sentiment'] > 0 else "#E53935"
            st.markdown(f"<p class='small-font'>• <b style='color:{color};'>[{'호재' if n['sentiment']>0 else '악재'}]</b> {n['title']}</p>", unsafe_allow_html=True)

        # 3. AI 추천 전략
        st.markdown("##### 🤖 AI 비즈니스 가이드")
        with st.container(border=True):
            if growth_input > 15: 
                st.warning(f"**공격적 확장:** {selected_prod_cat} 인벤토리를 평소보다 30% 추가 확보하십시오.")
            elif growth_input < 0: 
                st.error(f"**재고 리스크:** 수요 감소가 예상됩니다. 할인 프로모션을 통해 회전율을 높이십시오.")
            else: 
                st.success(f"**안정 유지:** 현재 시장 흐름과 일치합니다. 기존 마케팅 비용을 유지하십시오.")

with tab2:
    st.subheader("📰 Adidas USA Market Intelligence Center")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.info("🟢 긍정적 소식")
        for n in [x for x in news_db if x['sentiment'] > 0]:
            st.write(f"✅ {n['title']}")
    with col2:
        st.error("🔴 주의 요망")
        for n in [x for x in news_db if x['sentiment'] < 0]:
            st.write(f"🚨 {n['title']}")
    with col3:
        st.success("📊 인사이트 요약")
        st.write(f"- 분석 품목: {selected_prod_cat}")
        st.write(f"- 클러스터 {store_types[selected_type]['cluster']} 영향도 점검 완료")