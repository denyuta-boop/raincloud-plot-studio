import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns  # ✅ これを追加しました
from scipy.stats import mannwhitneyu
import io

# ページ設定
st.set_page_config(layout="wide", page_title="Scientific Plot Generator Pro")

st.title("📈 Scientific Plot Generator Pro")
st.markdown("左右背中合わせのハーフバイオリン図を作成します。")

# --- サイドバー：データと設定 ---
with st.sidebar:
    st.header("1. データ設定")
    use_demo = st.checkbox("サンプルデータを使う (Tips Dataset)")
    
    if use_demo:
        df = sns.load_dataset("tips") # ここでsnsを使用します
        group_col = "time"
        val_col = "total_bill"
        st.success("サンプルデータ(Tips)をロードしました")
    else:
        uploaded_file = st.file_uploader("CSVアップロード", type="csv")
        if uploaded_file:
            df = pd.read_csv(uploaded_file)
            group_col = st.selectbox("グループ列 (X軸)", df.columns)
            val_col = st.selectbox("値の列 (Y軸)", df.columns)
        else:
            df = None

    if df is not None:
        st.header("2. デザイン設定")
        color1 = st.color_picker("左グループの色", "#6FA9E5")
        color2 = st.color_picker("右グループの色", "#FA8F7C")
        plot_title = st.text_input("タイトル", "Distribution Comparison")

# --- メイン処理 ---
if df is not None:
    groups = df[group_col].unique()
    if len(groups) != 2:
        st.error("グループ列は必ず2つのカテゴリを含んでください。")
    else:
        g1, g2 = groups[0], groups[1]
        d1 = df[df[group_col] == g1][val_col].values
        d2 = df[df[group_col] == g2][val_col].values
        
        # 統計検定
        _, p_val = mannwhitneyu(d1, d2)
        
        # 描画開始
        fig, ax = plt.subplots(figsize=(7, 6))
        
        # --- 1. ハーフバイオリン描画（Seabornの機能を活用） ---
        # sideを指定することで確実にハーフにします
        sns.violinplot(data=df, x=[0]*len(d1), y=d1, color=color1, ax=ax, 
                       inner=None, cut=0, side='left', width=1.0, alpha=0.6)
        sns.violinplot(data=df, x=[0]*len(d2), y=d2, color=color2, ax=ax, 
                       inner=None, cut=0, side='right', width=1.0, alpha=0.6)

        # --- 2. 散布図とボックスプロット ---
        # ストリッププロット（各サイドに配置）
        ax.scatter(np.random.normal(0, 0.05, size=len(d1)) - 0.2, d1, color=color1, alpha=0.4, s=20)
        ax.scatter(np.random.normal(0, 0.05, size=len(d2)) + 0.2, d2, color=color2, alpha=0.4, s=20)
        
        # ボックスプロット（左右にオフセット）
        ax.boxplot([d1, d2], positions=[-0.2, 0.2], widths=0.15, showfliers=False, 
                   patch_artist=True, boxprops={'facecolor':'none', 'edgecolor':'black'})

        # --- 装飾 ---
        ax.axvline(0, color="gray", linestyle="--", alpha=0.5)
        ax.set_xticks([0])
        ax.set_xticklabels([f"{g1} vs {g2}\n(p={p_val:.4f})"], fontsize=12, fontweight='bold')
        ax.set_xlim(-0.8, 0.8)
        ax.set_title(plot_title, fontsize=14)
        
        st.pyplot(fig)
        
        # ダウンロード
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=300, bbox_inches='tight')
        st.download_button("PNGダウンロード", buf, "publication_plot.png", "image/png")
else:
    st.info("サイドバーからサンプルを選択するか、CSVをアップロードしてください。")
