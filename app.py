import numpy as np
from PIL import Image
import streamlit as st

from coin_detector import DENOMINATIONS, calculate_total, detect_coins

st.set_page_config(
    page_title="新台幣硬幣金額計算系統",
    page_icon="🪙",
    layout="wide",
)

st.title("🪙 新台幣硬幣金額計算系統（離線版）")
st.caption("完全離線運作 · 支援 1元、5元、10元、20元、50元 硬幣辨識與手動計算器")

# Initialize session state for manual calculator
for denom in DENOMINATIONS:
    key = f"qty_{denom}"
    if key not in st.session_state:
        st.session_state[key] = 0


def _apply_counts(counts: dict[int, int], add_mode: bool = False) -> None:
    for denom in DENOMINATIONS:
        key = f"qty_{denom}"
        if add_mode:
            st.session_state[key] += counts.get(denom, 0)
        else:
            st.session_state[key] = counts.get(denom, 0)


def _reset_counts() -> None:
    for denom in DENOMINATIONS:
        st.session_state[f"qty_{denom}"] = 0


col_vision, col_calc = st.columns([1.15, 1], gap="large")

with col_vision:
    st.subheader("📷 硬幣影像辨識")
    input_method = st.radio(
        "選擇輸入方式",
        options=["上傳圖片", "使用相機拍照"],
        horizontal=True,
    )

    uploaded_file = None
    if input_method == "上傳圖片":
        uploaded_file = st.file_uploader(
            "請上傳包含新台幣硬幣的照片 (JPG / PNG)",
            type=["jpg", "jpeg", "png"],
        )
    else:
        uploaded_file = st.camera_input("拍攝硬幣照片")

    with st.expander("⚙️ 辨識參數微調（若硬幣過大或過小可調整）"):
        param2 = st.slider("圓形偵測靈敏度 (param2，越小越敏感)", 20, 80, 38)
        min_radius = st.slider("最小硬幣半徑 (px)", 10, 80, 18)
        max_radius = st.slider("最大硬幣半徑 (px)", 50, 300, 140)

    if uploaded_file is not None:
        pil_img = Image.open(uploaded_file).convert("RGB")
        img_rgb = np.array(pil_img)

        result = detect_coins(
            img_rgb,
            param2=param2,
            min_radius=min_radius,
            max_radius=max_radius,
        )

        st.image(
            result.annotated_image,
            caption=f"辨識結果：共偵測到 {len(result.coins)} 枚硬幣，小計 NT$ {result.total_amount:,}",
            use_container_width=True,
        )

        st.markdown("**各面額辨識數量：**")
        badge_cols = st.columns(len(DENOMINATIONS))
        for idx, denom in enumerate(DENOMINATIONS):
            badge_cols[idx].metric(f"{denom} 元", f"{result.counts[denom]} 枚")

        btn_col1, btn_col2 = st.columns(2)
        with btn_col1:
            st.button(
                "📥 套用辨識結果到計算器",
                use_container_width=True,
                on_click=_apply_counts,
                args=(result.counts, False),
            )
        with btn_col2:
            st.button(
                "➕ 累加到計算器",
                use_container_width=True,
                on_click=_apply_counts,
                args=(result.counts, True),
            )
    else:
        st.info("請上傳硬幣照片或使用相機拍照，系統會自動離線辨識 1、5、10、20、50 元硬幣。")

with col_calc:
    st.subheader("🧮 手動輸入金額計算器")

    current_counts: dict[int, int] = {}
    for denom in DENOMINATIONS:
        c1, c2, c3 = st.columns([2.2, 1, 1])
        with c1:
            qty = st.number_input(
                f"{denom} 元硬幣 (枚)",
                min_value=0,
                step=1,
                key=f"qty_{denom}",
            )
            current_counts[denom] = int(qty)
        with c2:
            st.write("")
            st.write("")
            if st.button("+1", key=f"add1_{denom}", use_container_width=True):
                st.session_state[f"qty_{denom}"] += 1
                st.rerun()
        with c3:
            st.write("")
            st.write("")
            if st.button("+5", key=f"add5_{denom}", use_container_width=True):
                st.session_state[f"qty_{denom}"] += 5
                st.rerun()

    st.divider()

    total_coins = sum(current_counts.values())
    total_amount = calculate_total(current_counts)

    m1, m2 = st.columns(2)
    m1.metric("總枚數", f"{total_coins:,} 枚")
    m2.metric("總金額", f"NT$ {total_amount:,}")

    st.button("🗑️ 全部歸零", on_click=_reset_counts, use_container_width=True)
