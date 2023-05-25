
import streamlit as st

st.set_page_config(
    page_title="AIIOT LAB",
    page_icon="👋",
)

def main():
    # st.write("# AIIOT LAB 방문을 환영합니다.")
    # st.markdown("---")
    # st.write("# Welcome to Streamlit! 👋")
    # st.sidebar.success("데모를 보려면 위 메뉴를 클릭하세요.")

    st.markdown(
        """
        # 데모 설명

        1. 화면에 등장하는 보행자 숫자 세기 
        2. 골프 카트 주행 추적
        3. 난반사되는 강에서의 보트 움직임 추적

        **데모의 실행은 👈 왼쪽 메뉴를 이용하세요.**
        """
    )

    hide_streamlit_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
    """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True)

if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        pass
