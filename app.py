import json

import streamlit as st

from serving.predictor import NewsScorerPredictor


EXAMPLES = [
    'Company cuts full-year guidance amid weak demand',
    'CEO resigns effective immediately after accounting probe',
    'Analysts discuss whether tech may outperform next year',
    'Drugmaker wins FDA approval for new cancer treatment',
]


@st.cache_resource
def get_predictor() -> NewsScorerPredictor:
    return NewsScorerPredictor()


def apply_example() -> None:
    selected_example = st.session_state.get('selected_example')
    if selected_example and selected_example != 'Custom input':
        st.session_state['input_text'] = selected_example


st.set_page_config(page_title='News-to-Trade Relevance Scorer', page_icon='📈', layout='wide')

st.title('News-to-Trade Relevance Scorer')
st.caption('Compact ONNX-powered financial NLP demo for sentiment, actionability, event type, and heuristic trading horizon.')
st.info('This is a decision-support demo, not investment advice.')

st.sidebar.header('Examples')
st.sidebar.selectbox('Load example', options=['Custom input', *EXAMPLES],
                     key='selected_example', on_change=apply_example)

text = st.text_area('Financial text', key='input_text', height=140,
                    placeholder='Paste a headline, tweet, short news item, or press release excerpt...')

score_clicked = st.button('Score', type='primary', use_container_width=True)

if score_clicked:
    try:
        result = get_predictor().predict(text)
    except ValueError as error:
        st.error(str(error))
    except Exception as error:
        st.exception(error)
    else:
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric('Sentiment', result['sentiment']['label'])
            st.caption(f'Confidence: {result['sentiment']['confidence']:.1%}')

        with col2:
            st.metric('Actionability', result['actionability']['label'])
            st.caption(f'Confidence: {result['actionability']['confidence']:.1%}')

        with col3:
            event_type = (
                'not_applicable'
                if result['event_type'] is None
                else result['event_type']['label']
            )
            st.metric('Event Type', event_type)
            if result['event_type'] is not None:
                st.caption(f'Confidence: {result['event_type']['confidence']:.1%}')

        with col4:
            st.metric('Trading Horizon', result['horizon'])
            st.caption(f'Backend: {result['meta']['backend']}')

        st.subheader('Rationale')
        st.write(result['rationale'])

        st.subheader('Raw JSON')
        st.code(json.dumps(result, indent=2, ensure_ascii=False), language='json')
