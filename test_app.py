import pytest
from streamlit.testing.v1 import AppTest
# app.py から単体テストしたい関数をインポート
from app import stars, to_mathtext

# ============================================================
# 1. 独立した関数の単体テスト (Unit Test)
# ============================================================
def test_stars_function():
    """p値に応じた有意差マークが正しく返ってくるかテスト"""
    assert stars(0.0005) == "***"
    assert stars(0.005) == "**"
    assert stars(0.03) == "*"
    assert stars(0.12) == "ns"

def test_to_mathtext_function():
    """アンダーバーを含むラベルがLaTeX数式風に変換されるかテスト"""
    assert to_mathtext("Sample_1") == "$Sample_{\\mathrm{1}}$"
    assert to_mathtext("NormalLabel") == "NormalLabel"


# ============================================================
# 2. Streamlit画面の統合テスト (Integration Test via AppTest)
# ============================================================
def test_streamlit_app_load():
    """アプリがエラーなしで初期起動し、デフォルトの要素が存在するかテスト"""
    # app.py をシミュレータに読み込む
    at = AppTest.from_file("app.py").run()
    
    # 起動時に例外エラー（Crash）が発生していないことを確認
    assert not at.exception
    
    # タイトルが正しく表示されているか確認
    assert at.title[0].value == "🌧️ Raincloud Plot Studio"
    
    # 初期状態では「サンプルデータを使う」のチェックボックスがONになっているか確認
    # (sidebarの中のcheckboxの0番目)
    assert at.sidebar.checkbox[0].value is True


def test_streamlit_app_font_and_toggle():
    """サイドバーの操作や設定変更が正常に反映されるかテスト"""
    at = AppTest.from_file("app.py").run()
    
    # フォントのラジオボタンを「Times New Roman」に切り替えて再実行してみる
    at.sidebar.radio[0].set_value("Times New Roman（欧文・論文向け）").run()
    assert not at.exception

    # グラフの向きを「横 (Horizontal)」に切り替えてみる
    # グラフ調整エリアのラジオボタンを探して操作
    at.sidebar.radio[2].set_value("横 (Horizontal)").run()
    assert not at.exception
    
    # 有意差のマークやデータプレビューのサブヘッダーが画面に登場しているか確認
    assert any("データプレビュー" in sh.value for sh in at.subheader)