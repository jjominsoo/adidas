import streamlit as st
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import font_manager, rc

# [수정] 한글 폰트 설정 (윈도우 기준)
try:
    font_path = "C:/Windows/Fonts/malgun.ttf"
    font_name = font_manager.FontProperties(fname=font_path).get_name()
    rc('font', family=font_name)
except:
    pass # 맥/리눅스 환경일 경우 기본 폰트 사용

st.set_page_config(page_title="Adidas USA 2026 AI", layout="wide")
st.title("👟 Adidas USA 2026 Sales Strategy")

@st.cache_resource
def load_all_assets():
    pkg = joblib.load('models/adidas_web_model.pkl')
    return pkg

pkg = load_all_assets()

# 사이드바
st.sidebar.header("🔍 설정")
selected_retailer = st.sidebar.selectbox("리테일러", pkg['le_retailer'].classes_)
selected_product = st.sidebar.selectbox("품목", pkg['le_product'].classes_)
target_growth = st.sidebar.slider("목표 성장률 (%)", -100, 100, 10) # 범위를 넓게 잡아서 테스트

# 예측 함수
def get_prediction(retailer, product, growth):
    ret_enc = pkg['le_retailer'].transform([retailer])[0]
    prod_enc = pkg['le_product'].transform([product])[0]
    
    predict_df = pd.DataFrame({
        'Retailer_ID_Enc': [ret_enc] * 12,
        'Product_Type_Enc': [prod_enc] * 12,
        'Month': list(range(1, 13)),
        'Cluster': [1] * 12
    })
    
    preds = pkg['model'].predict(predict_df[pkg['features']])
    
    # [중요] 성장률 적용
    adjusted_preds = preds * (1 + growth / 100)
    
    avg_price = pkg['price_map'].get((retailer, product), 50)
    avg_sf = pkg['sf_map'].get((retailer, product), 1.0)
    monthly_sales = adjusted_preds * avg_price * avg_sf
    
    return adjusted_preds, monthly_sales

units, sales = get_prediction(selected_retailer, selected_product, target_growth)
# [추가] 눈금 고정을 위한 기준값 계산 
# 성장률이 0%일 때를 기준으로 약 2배 정도의 여유 공간을 둡니다.
# 이렇게 해야 슬라이더를 올릴 때 그래프가 천장으로 솟구치는 효과가 납니다.
base_units, base_sales = get_prediction(selected_retailer, selected_product, 0)
max_unit_limit = base_units.max() * 2.5  # 고정 눈금 (수량)
max_sales_limit = base_sales.max() * 2.5 # 고정 눈금 (매출)
# 화면 구성
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader(f"📅 2026 {selected_retailer} 실시간 시뮬레이션")
    
    chart_data = pd.DataFrame({
        "월": [f"{m}월" for m in range(1, 13)],
        "수량": units,
        "매출액": sales
    })
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # 1. 막대 그래프 (수량)
    sns.barplot(x="월", y="수량", data=chart_data, ax=ax1, color="skyblue", alpha=0.8)
    ax1.set_ylabel("예상 판매 수량 (Pcs)", color="blue", fontsize=12)
    
    # [핵심] Y축 눈금 고정!
    ax1.set_ylim(0, max_unit_limit) 

    # 2. 꺾은선 그래프 (매출) - 오른쪽 축
    ax2 = ax1.twinx()
    sns.lineplot(x="월", y="매출액", data=chart_data, ax=ax2, marker='o', color='red', linewidth=3)
    ax2.set_ylabel("예상 매출액 ($)", color="red", fontsize=12)
    
    # [핵심] Y축 눈금 고정!
    ax2.set_ylim(0, max_sales_limit)

    # 그래프 격자 추가 (변화를 더 잘 보이게 함)
    ax1.grid(True, axis='y', linestyle='--', alpha=0.5)
    
    plt.title(f"{selected_product} 성장률 {target_growth}% 적용 결과", fontsize=15)
    st.pyplot(fig)

with col2:
    st.subheader("🤖 AI 리포트")
    total_sales = sales.sum()
    
    # 변화를 확인하기 위한 지표
    st.metric("총 매출액", f"${total_sales:,.0f}", f"{target_growth}%")
    st.metric("총 판매수량", f"{units.sum():,.0f} Pcs")
    
    if target_growth > 20:
        st.error("🚨 공격적 목표: 재고 확보 필수")
    elif target_growth < -10:
        st.warning("⚠️ 수요 감소: 할인 행사 준비")
    else:
        st.success("✅ 적정 목표: 현재 공급망 유지")