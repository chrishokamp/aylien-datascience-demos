import streamlit as st
# importing tools from your library into the demo:
from aylien_datascience_demos.example_module import Counter, TextReverser


page_config = st.set_page_config(
    page_title="Demo",
)


def get_session_state():
    # Initialize session state
    if not st.session_state.get('INIT', False):
        st.session_state['counter'] = Counter()

    st.session_state['INIT'] = True
    return st.session_state


def main():
    st.write("# Simple Streamlit Demo")


if __name__ == '__main__':
    main()
