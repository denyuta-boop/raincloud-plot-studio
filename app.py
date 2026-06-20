import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
from scipy.stats import mannwhitneyu, wilcoxon, kruskal, friedmanchisquare
from itertools import combinations
import io
import os
import urllib.request

st.set_page_config(layout="wide", page_title="Raincloud Plot Studio")
st.title("🌧️ Raincloud Plot Studio")
st.markdown(
    "ハーフバイオリン＋散布図＋箱ひげ図を重ねた **Raincloud Plot** を、"
    "系列数・並び順・色を自由に変えながら作れます。"
)


# ============================================================
# ヘルパー関数
# ============================================================
def stars(p):
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    else:
        return "ns"


def to_mathtext(label):
    if "_" in label:
        main, sub = label.split("_", 1)
        return f"${main}_{{\\mathrm{{{sub}}}}}$"
    return label


@st.cache_resource
def load_google_font(url, cache_path):
    try:
        if not os.path.exists(cache_path):
            urllib.request.urlretrieve(url, cache_path)
        fm.fontManager.addfont(cache_path)
        return fm.FontProperties(fname=cache_path).get_name()
    except Exception:
        return None


@st.cache_data
def load_demo_data():
    return sns.load_dataset("tips")


# ============================================================
# サイドバー：データ取り込み
# ============================================================
with st.sidebar:
    st.header("0. フォント")
    font_choice = st.radio(
        "グラフ内のフォント",
        ["デフォルト", "Times New Roman（欧文・論文向け）", "Noto Sans JP（日本語タイトル等を使う場合）"],
        index=0,
    )
    font_family = "sans-serif"
    if font_choice.startswith("Times"):
        loaded = load_google_font(
            "https://github.com/google/fonts/raw/main/apache/timesnewroman/Times%20New%20Roman.ttf",
            "/tmp/TimesNewRoman.ttf",
        )
        if loaded:
            font_family = loaded
        else:
            st.warning("フォントのダウンロードに失敗しました。デフォルトフォントを使用します。")
    elif font_choice.startswith("Noto"):
        loaded = load_google_font(
            "https://github.com/google/fonts/raw/main/ofl/notosansjp/NotoSansJP%5Bwght%5D.ttf",
            "/tmp/NotoSansJP.ttf",
        )
        if loaded:
            font_family = loaded
        else:
            st.warning("フォントのダウンロードに失敗しました。日本語タイトル等は文字化けする可能性があります。")
    plt.rcParams["font.family"] = font_family
    plt.rcParams["mathtext.fontset"] = "stix"
    plt.rcParams["axes.unicode_minus"] = False

    st.header("1. データ")
    uploaded = st.file_uploader("CSVをアップロード", type="csv")
    use_demo = st.checkbox("サンプルデータを使う (Tips Dataset)", value=(uploaded is None))

    if uploaded is not None:
        df = pd.read_csv(uploaded)
    elif use_demo:
        df = load_demo_data()
    else:
        df = None

    data_mode = None
    selected = []
    data_by_cat = {}
    val_label_default = "Value"
    is_paired_mode = False

    if df is not None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) == 0:
            st.error("数値列が見つかりません。CSVを確認してください。")
            df = None

    if df is not None:
        data_mode = st.radio(
            "データの形式",
            ["ロング形式（グループ列1つ＋値1列）", "ワイド形式（基準値列＋比較列を複数選択→誤差を計算）"],
        )

        if data_mode.startswith("ロング"):
            cat_cols = [c for c in df.columns if c not in numeric_cols] or df.columns.tolist()
            group_col = st.selectbox(
                "グループ列（系列を分ける列）", cat_cols,
                index=cat_cols.index("day") if "day" in cat_cols else 0,
            )
            val_col = st.selectbox(
                "値の列（数値）", numeric_cols,
                index=numeric_cols.index("total_bill") if "total_bill" in numeric_cols else 0,
            )
            all_categories = sorted(df[group_col].dropna().unique().tolist(), key=str)
            selected = st.multiselect("含める系列（2つ以上）", all_categories, default=all_categories)
            for cat in selected:
                vals = df[df[group_col] == cat][val_col].dropna().values
                if len(vals) > 0:
                    data_by_cat[cat] = vals
            selected = [c for c in selected if c in data_by_cat]
            val_label_default = val_col
            is_paired_mode = False

        else:
            st.caption("例：meanSSDE（基準値）, SSDE_RAI, SSDE_center… のような列構成のCSV")
            ref_col = st.selectbox("基準値列", numeric_cols)
            method_candidates = [c for c in numeric_cols if c != ref_col]
            default_methods = method_candidates[: min(2, len(method_candidates))]
            selected = st.multiselect("比較したい列（誤差＝基準値－この列、2つ以上）", method_candidates, default=default_methods)
            if len(selected) >= 1:
                sub = df[[ref_col] + selected].dropna()
                for m in selected:
                    data_by_cat[m] = (sub[ref_col] - sub[m]).values
            val_label_default = f"{ref_col} - Method"
            is_paired_mode = True

# ============================================================
# サイドバー：表示・グラフオプション（系列が2つ以上揃ってから表示）
# ============================================================
plot_df = None
ordered_cats = []

if df is not None and len(selected) >= 2:
    with st.sidebar:
        st.header("2. 表示オプション")
        plot_title = st.text_input("タイトル", "Distribution Comparison")
        y_label = st.text_input("Y軸ラベル", val_label_default)
        if font_choice != "Noto Sans JP（日本語タイトル等を使う場合）" and any(
            ord(ch) > 0x3000 for ch in (plot_title + y_label)
        ):
            st.caption("⚠️ 日本語を含む文字列が検出されました。グラフ内で文字化けする場合は「Noto Sans JP」を選択してください。")
        use_mathtext = st.checkbox("X軸ラベルを下付き文字風にする（例: SSDE_RAI）", value=False)
        show_zero_line = st.checkbox("y = 0 の破線を表示", value=is_paired_mode)
        show_pairwise = st.checkbox("ペアごとの比較表を表示（3系列以上の場合）", value=True)

        st.header("3. グラフ調整")
        fig_width = st.slider("画像横幅 (inch)", 6.0, 22.0, float(min(16, max(7, len(selected) * 2.3))), 0.5)
        fig_height = st.slider("画像縦幅 (inch)", 5.0, 12.0, 7.5, 0.5)
        violin_width = st.slider("バイオリン幅", 0.5, 1.2, 0.9, 0.05)
        box_width = st.slider("ボックス幅", 0.1, 0.4, 0.2, 0.02)
        jitter = st.slider("散布点のJitter量", 0.05, 0.4, 0.15, 0.01)
        point_size = st.slider("散布点サイズ", 2, 15, 6, 1)
        point_alpha = st.slider("散布点透明度", 0.2, 1.0, 0.7, 0.05)
        point_color_mode = st.radio("散布点の色", ["グレー（統一）", "系列色"], index=0)

        st.header("4. フォントサイズ")
        fs_title = st.slider("タイトル", 10, 40, 20)
        fs_axis = st.slider("軸ラベル", 10, 40, 18)
        fs_tick = st.slider("目盛りラベル", 10, 40, 16)
        fs_sig = st.slider("有意差ラベル", 10, 40, 16)

    # --------------------------------------------------------
    # メインエリア：系列ごとの設定（順番・ラベル・色）
    # --------------------------------------------------------
    st.subheader("🎨 系列の設定（順番・ラベル・色）")
    preset = st.selectbox("プリセットカラーパレット", ["Set2", "Pastel1", "muted", "husl", "viridis", "coolwarm"])
    default_hex = sns.color_palette(preset, len(selected)).as_hex()

    series_config = {}
    n_cols = 3
    for row_start in range(0, len(selected), n_cols):
        row_cats = selected[row_start: row_start + n_cols]
        cols = st.columns(len(row_cats))
        for col, cat in zip(cols, row_cats):
            idx = selected.index(cat)
            with col:
                st.markdown(f"**{cat}**")
                order_val = st.number_input(
                    "順番", min_value=1, max_value=len(selected), value=idx + 1, key=f"order_{cat}"
                )
                label_val = st.text_input("ラベル", str(cat), key=f"label_{cat}")
                color_val = st.color_picker("色", default_hex[idx], key=f"color_{cat}_{preset}")
                series_config[cat] = {"order": order_val, "label": label_val, "color": color_val}

    ordered_cats = sorted(selected, key=lambda c: (series_config[c]["order"], selected.index(c)))
    labels = [series_config[c]["label"] for c in ordered_cats]
    colors = [series_config[c]["color"] for c in ordered_cats]
    data_list = [data_by_cat[c] for c in ordered_cats]

    if len(set(labels)) != len(labels):
        st.error("ラベルが重複しています。系列ごとに異なるラベルを設定してください。")
        plot_df = None
    else:
        plot_df = pd.DataFrame(
            {
                "Group": sum([[lab] * len(d) for lab, d in zip(labels, data_list)], []),
                "Value": np.concatenate(data_list),
            }
        )

# ============================================================
# メインエリア：統計検定 ＋ 描画
# ============================================================
if plot_df is not None:
    N = len(ordered_cats)
    sig_text = None
    bracket_pair = None  # (x1, x2) for N==2 only
    pairwise_df = None

    if N == 2:
        d1, d2 = data_list[0], data_list[1]
        st.subheader("📐 統計検定")
        test_options = ["Mann-Whitney U検定（対応なし）", "Wilcoxon符号順位検定（対応あり）"]
        default_index = 1 if is_paired_mode else 0
        test_choice = st.radio("検定方法", test_options, index=default_index, horizontal=True)
        if test_choice.startswith("Wilcoxon") and len(d1) != len(d2):
            st.caption(
                f"⚠️ サンプル数が異なります（{len(d1)} 件 / {len(d2)} 件）。"
                "選択した場合は自動的にMann-Whitney Uにフォールバックします。"
            )

        is_wilcoxon = test_choice.startswith("Wilcoxon")
        fell_back = False
        if is_wilcoxon and len(d1) == len(d2):
            _, p_val = wilcoxon(d1, d2)
            test_name_used = "Wilcoxon符号順位検定（対応あり）"
        else:
            if is_wilcoxon:
                fell_back = True
            _, p_val = mannwhitneyu(d1, d2, alternative="two-sided")
            test_name_used = "Mann-Whitney U検定（対応なし）"

        sig_text = f"{stars(p_val)} (p = {p_val:.4g})"
        bracket_pair = (0, 1)

        if fell_back:
            st.warning(
                f"サンプル数が異なるため（{len(d1)}件 / {len(d2)}件）、Wilcoxon検定は実行できません。"
                "代わりにMann-Whitney U検定の結果を表示しています。"
            )
        st.caption(f"使用した検定：{test_name_used}　/　p = {p_val:.4g}")

    else:
        st.subheader("📐 統計検定（3系列以上）")
        omnibus_fell_back = False
        if is_paired_mode:
            try:
                _, p_val = friedmanchisquare(*data_list)
                test_name_used = "Friedman検定（対応あり・N群）"
                test_name_en = "Friedman test"
            except Exception:
                omnibus_fell_back = True
                _, p_val = kruskal(*data_list)
                test_name_used = "Kruskal-Wallis検定（フォールバック・対応なし）"
                test_name_en = "Kruskal-Wallis test"
        else:
            _, p_val = kruskal(*data_list)
            test_name_used = "Kruskal-Wallis検定（対応なし・N群）"
            test_name_en = "Kruskal-Wallis test"

        if omnibus_fell_back:
            st.warning("Friedman検定が実行できなかったため、Kruskal-Wallis検定にフォールバックしました。")

        st.caption(f"使用した検定：{test_name_used}　/　p = {p_val:.4g}　{stars(p_val)}")
        # グラフ内に直接描画する注釈は英数字のみ（フォント未対応による文字化け防止）
        sig_text = f"{test_name_en}: {stars(p_val)} (p = {p_val:.4g})"

        if show_pairwise:
            pairs = list(combinations(range(N), 2))
            rows = []
            for i, j in pairs:
                di, dj = data_list[i], data_list[j]
                if is_paired_mode and len(di) == len(dj):
                    _, p_raw = wilcoxon(di, dj)
                    method_used = "Wilcoxon"
                else:
                    _, p_raw = mannwhitneyu(di, dj, alternative="two-sided")
                    method_used = "Mann-Whitney U"
                p_corr = min(p_raw * len(pairs), 1.0)
                rows.append(
                    {
                        "系列1": labels[i],
                        "系列2": labels[j],
                        "検定": method_used,
                        "p (raw)": round(p_raw, 4),
                        "p (Bonferroni補正)": round(p_corr, 4),
                        "有意差": stars(p_corr),
                    }
                )
            pairwise_df = pd.DataFrame(rows)

    # --------------------------------------------------------
    # 描画
    # --------------------------------------------------------
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=300)

    sns.violinplot(
        data=plot_df, x="Group", y="Value", hue="Group", order=labels, hue_order=labels,
        split=True, inner=None, cut=0, density_norm="count",
        width=violin_width, palette=colors, linewidth=1.2,
        ax=ax, legend=False,
    )

    use_group_colors = point_color_mode.startswith("系列")
    sns.stripplot(
        data=plot_df, x="Group", y="Value", order=labels,
        hue="Group" if use_group_colors else None,
        hue_order=labels if use_group_colors else None,
        palette=colors if use_group_colors else None,
        color=None if use_group_colors else "gray",
        size=point_size, jitter=jitter, alpha=point_alpha, ax=ax, legend=False,
        edgecolor="white", linewidth=0.3,
    )

    sns.boxplot(
        data=plot_df, x="Group", y="Value", order=labels, width=box_width,
        showcaps=True,
        boxprops={"facecolor": "none", "zorder": 10, "linewidth": 1.5},
        medianprops={"color": "black", "zorder": 11, "linewidth": 1.5},
        whiskerprops={"linewidth": 1.5, "zorder": 10},
        capprops={"linewidth": 1.5, "zorder": 10},
        showfliers=False, ax=ax,
    )

    ax.set_title(plot_title, fontsize=fs_title)
    ax.set_ylabel(y_label, fontsize=fs_axis)
    ax.set_xlabel("")

    ax.set_xticks(list(range(N)))
    if use_mathtext:
        ax.set_xticklabels([to_mathtext(lab) for lab in labels], fontsize=fs_tick)
    else:
        ax.set_xticklabels(labels, fontsize=fs_tick)

    ax.tick_params(axis="y", labelsize=fs_tick)
    ax.tick_params(axis="x", length=0)

    if show_zero_line:
        ax.axhline(0, color="black", linestyle="--", linewidth=1.2)

    y_max = plot_df["Value"].max()
    y_min = plot_df["Value"].min()
    data_range = y_max - y_min if y_max > y_min else 1.0
    top_padding = data_range * 0.25
    ax.set_ylim(y_min - data_range * 0.1, y_max + top_padding)

    y = y_max + top_padding * 0.2
    h = data_range * 0.03
    if N == 2 and bracket_pair is not None:
        x1, x2 = bracket_pair
        ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1.5, color="black")
        ax.text((x1 + x2) / 2, y + h, sig_text, ha="center", va="bottom", fontsize=fs_sig)
    elif sig_text is not None:
        ax.text((N - 1) / 2, y, sig_text, ha="center", va="bottom", fontsize=fs_sig)

    sns.despine(ax=ax)
    st.pyplot(fig)

    if pairwise_df is not None:
        st.markdown("**ペアごとの比較（Bonferroni補正済み）**")
        st.dataframe(pairwise_df, use_container_width=True)

    # --------------------------------------------------------
    # ダウンロード
    # --------------------------------------------------------
    c1, c2 = st.columns(2)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=300, bbox_inches="tight")
    c1.download_button(
        "📥 PNGダウンロード (Publication Quality)", data=buf,
        file_name="raincloud_plot.png", mime="image/png",
    )
    c2.download_button(
        "📄 描画データをCSVダウンロード", data=plot_df.to_csv(index=False).encode("utf-8"),
        file_name="raincloud_plot_data.csv", mime="text/csv",
    )

elif df is not None:
    st.info("👈 系列を2つ以上選択してください。")
else:
    st.info("👈 サイドバーからCSVをアップロードするか、サンプルデータを使ってください。")
