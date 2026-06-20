import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mannwhitneyu
import io

st.set_page_config(layout="wide", page_title="Scientific Plot Generator Pro")
st.title("📈 Scientific Plot Generator Pro")
st.markdown("**左右背中合わせのハーフバイオリン図**を作成します。")

# --- サイドバー ---
with st.sidebar:
    st.header("1. データ設定")
    use_demo = st.checkbox("サンプルデータを使う (Tips Dataset)", value=True)
  
    if use_demo:
        df = sns.load_dataset("tips")
        group_col = "time"
        val_col = "total_bill"
        st.success("✅ サンプルデータ(Tips)をロードしました")
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

        # === 新規：調整スライダー ===
        st.header("3. グラフ調整")
        fig_width = st.slider("画像横幅 (inch)", 6.0, 14.0, 9.0, 0.5)
        violin_width = st.slider("バイオリン幅", 0.5, 1.2, 0.9, 0.05)
        offset = st.slider("オフセット位置（左右のずれ）", 0.10, 0.40, 0.22, 0.01)
        jitter = st.slider("散布点のJitter量", 0.02, 0.15, 0.06, 0.01)
        point_size = st.slider("散布点サイズ", 10, 50, 24, 2)
        point_alpha = st.slider("散布点透明度", 0.2, 1.0, 0.65, 0.05)

# --- メイン処理 ---
if df is not None:
    groups = df[group_col].unique()
    if len(groups) != 2:
        st.error("グループ列は**必ず2つのカテゴリ**を含んでください。")
    else:
        g1, g2 = sorted(groups)
        d1 = df[df[group_col] == g1][val_col].dropna().values
        d2 = df[df[group_col] == g2][val_col].dropna().values
      
        _, p_val = mannwhitneyu(d1, d2, alternative='two-sided')
      
        # ==================== 描画 ====================
        fig, ax = plt.subplots(figsize=(fig_width, 7), dpi=300)
      
        plot_df = pd.DataFrame({
            'Value': np.concatenate([d1, d2]),
            'Group': [g1] * len(d1) + [g2] * len(d2)
        })
       
        sns.violinplot(
            data=plot_df,
            x="Group",
            y="Value",
            hue="Group",
            split=True,
            inner=None,
            cut=0,
            density_norm='count',
            width=violin_width,
            palette=[color1, color2],
            ax=ax,
            linewidth=1.2
        )
       
        # 散布点
        left_x = np.random.normal(-offset, jitter, len(d1))
        right_x = np.random.normal(offset, jitter, len(d2))
        ax.scatter(left_x, d1, color=color1, alpha=point_alpha, s=point_size,
                  edgecolor='white', linewidth=0.4, zorder=3)
        ax.scatter(right_x, d2, color=color2, alpha=point_alpha, s=point_size,
                  edgecolor='white', linewidth=0.4, zorder=3)
       
        # ボックスプロット
        bp = ax.boxplot([d1, d2], positions=[-offset, offset], widths=0.18,
                        showfliers=False, patch_artist=True,
                        medianprops={'color': 'darkorange', 'linewidth': 2.5},
                        zorder=4)
       
        for patch, color in zip(bp['boxes'], [color1, color2]):
            patch.set_facecolor('none')
            patch.set_edgecolor('black')
            patch.set_alpha(0.95)
       
        # 装飾
        ax.set_xlabel("")
        ax.set_title(plot_title, fontsize=16, pad=20)
        ax.set_ylabel(val_col)
        ax.set_xticks([0, 1])
        ax.set_xticklabels([g1, g2], fontsize=13)
        ax.text(0.5, ax.get_ylim()[1] * 0.96, f"p = {p_val:.4f}",
                ha='center', fontsize=12.5, fontweight='bold')
       
        sns.despine(left=False, bottom=False, ax=ax)
       
        st.pyplot(fig)
      
        # ダウンロード
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=300, bbox_inches='tight')
        st.download_button(
            label="📥 PNGダウンロード (Publication Quality)",
            data=buf,
            file_name="half_violin_plot.png",
            mime="image/png"
        )
else:
    st.info("👈 サイドバーからデータを選んでください")
