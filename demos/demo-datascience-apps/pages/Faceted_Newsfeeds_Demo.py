import streamlit as st
from pathlib import Path
from PIL import Image
import os
import copy
import arrow
import calendar
import hashlib
from collections import OrderedDict, defaultdict, Counter
from datetime import datetime
import json

import spacy

import aylien_datascience_demos.schema_population as schema_population
import aylien_datascience_demos.newsapi as newsapi
from aylien_datascience_demos.components import download_button
import streamlit.components.v1 as components
from collections import namedtuple

import pandas as pd


FACET_MAP = {
  'LOC_FACET': 'locations',
  'PEOPLE_FACET': 'people',
  'DATE_FACET': 'dates',
  'ORG_FACET': 'organisations',
  'CATEGORY_FACET': 'categories'
}
FACETS = namedtuple(
    'Facets',
    FACET_MAP.keys()
)(**FACET_MAP)

path_to_file = Path(os.path.dirname(os.path.abspath(__file__)))
img = Image.open(path_to_file / '../favicon.ico')

st.set_page_config(
    page_title='Aylien Feed Facets Demo',
    page_icon=img,
    layout="wide",
    initial_sidebar_state='auto')


@st.cache
def get_example_feeds():
    EXAMPLE_FEED_PATHS = [
        p for p in (path_to_file / 'example_faceted_newsfeeds').glob('*.json')
    ]
    feeds = []
    for p in EXAMPLE_FEED_PATHS:
        feed = json.load(open(p))
        for e in feed['feed_items'].values():
            # custom deserialization :-)
            e['start_date'] = arrow.get(e['start_date']).datetime
        feed['feed_name'] = p.stem
        feeds.append(feed)
    return feeds


class Newsapi:
    def __init__(self):
        self.stories_cache = {}

    def retrieve_stories(self, params, **kwargs):
        key = json.dumps(params)
        if key in self.stories_cache:
            return self.stories_cache
        else:
            data = newsapi.retrieve_stories(params, **kwargs)
            self.stories_cache[key] = data
        return self.stories_cache[key]


def hide_menu_and_footer():
    hide_streamlit_style = '''
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
    '''
    st.markdown(hide_streamlit_style, unsafe_allow_html=True)


def thick_vertical_line(width):
    components.html(
        """<hr style="height:5px;border:none;color:#333;background-color:#4293f5;margin-bottom:0;padding:0" /> """,
        width=width
    )


def story_from_text(text: str):
    story = {
        "title": '',
        "body": text,
    }
    return story


def get_session_state():
    state = st.session_state

    if not state.get('INIT_FACETED_NEWSFEEDS_DEMO', False):
        # load previously cached feeds

        # Initialize user state if session doesn't exist
        # Create a Tokenizer with the default settings for English
        # including punctuation rules and exceptions
        nlp = spacy.load("en_core_web_sm")
        # TODO: for online views, let user add more stories
        # incrementally, simulating streaming usecase (tracking how
        # events evolve over time)
        query_template = {
          "text": "Ukraine",
          "per_page": 50,
          "published_at.start": "NOW-7DAYS",
          "sort_by": "relevance",
          "language": "en"
        }
        state['local_nlp'] = nlp
        state['current_newsapi_query'] = copy.deepcopy(query_template)
        state['feed_items'] = OrderedDict()
        state['stories'] = OrderedDict()
        state['newsapi'] = Newsapi()
        state['example_feeds'] = get_example_feeds()
    state['INIT_FACETED_NEWSFEEDS_DEMO'] = True

    return state


def item_id(story):
    """Document ID from text"""
    return hashlib.md5(story['body'][:100].encode()).hexdigest()


def render():
    """
    UI for rendering views of Aylien item collections
    """
    session_state = get_session_state()
    render_sidebar(session_state)
    render_main(session_state)


def render_sidebar(session_state):
    ###########
    # SIDEBAR #
    ###########

    ###################################
    # Configure Input Streams/Sources #
    ###################################
    # user can load or paste generated queries from file
    # st.sidebar.markdown('### Input Sources')
    # if st.sidebar.button('Clear Feed'):
    #     session_state['feed_items'] = OrderedDict()
    #     session_state['id_to_geolocs'] = None
    #     st.experimental_rerun()
    st.sidebar.markdown('----')

    st.sidebar.markdown('### Select a Sample Feed')

    selected_feed = st.sidebar.radio(
        'Select an Example Feed',
        options=session_state['example_feeds'],
        format_func=lambda o: o['feed_name'],
        key='select-example-feed-radio'
    )
    session_state['current_newsapi_query'] = selected_feed['current_newsapi_query']
    session_state['feed_items'] = selected_feed['feed_items']
    session_state['stories'] = selected_feed['stories']
    session_state["id_to_geolocs"] = selected_feed['id_to_geolocs']
    session_state["sf_to_geoloc"] = selected_feed['sf_to_geoloc']
    # st.experimental_rerun()

    st.sidebar.markdown('----')

    # Newsapi Query
    # TODO: query cache with some data loaded up on startup
    query = st.sidebar.text_area(
        'NewsAPI Query',
        json.dumps(session_state['current_newsapi_query'], indent=2),
        height=350
    )
    session_state['current_newsapi_query'] = json.loads(query)

    if st.sidebar.button('Populate Feed from Query', key='populate_feed_from_query'):
        query = session_state['current_newsapi_query']
        st.sidebar.info('Retrieving Feed Stories ... ')
        stories = session_state['newsapi'].retrieve_stories(
            params=query
        )
        for s in stories:
            s["title_doc"] = session_state["local_nlp"](s["title"])
            s["body_doc"] = session_state["local_nlp"](s["body"])
        st.sidebar.info('Extracting clusters from feed ... ')
        ###############################################
        # Map items into views (Events, Entities, ... #
        # ("Event" is another type of transient Item) #
        ###############################################
        clusters = schema_population.cluster_items(
            stories,
            get_text=(lambda x: f"{str(x['title_doc'])} {str(list(x['body_doc'].sents)[:3])}"),
            min_samples=2,
            eps=0.75
        )
        st.sidebar.info('Extracting Events from Feed ... ')
        events = [schema_population.stories_to_event(c) for c in clusters]
        st.sidebar.info('Extracting Geolocations... ')
        id_to_geolocs, sf_to_geoloc =\
            schema_population.extract_geolocations(events)
        # TODO: cache events for query, don't recompute
        # TODO: assert events are json serializable
        for e in events:
            session_state['feed_items'][e['id']] = e
        for s in stories:
            session_state['stories'][s['id']] = s

        session_state["id_to_geolocs"] = id_to_geolocs
        session_state["sf_to_geoloc"] = sf_to_geoloc
        st.experimental_rerun()

    st.sidebar.markdown('----')

    ##################
    # DOWNLOAD STATE #
    ##################
    # User can download their state
    
    if len(session_state['feed_items']) > 0:
        st.sidebar.markdown('### Download State')
        downloadable_state = {
            'current_newsapi_query': session_state['current_newsapi_query'],
            'feed_items': session_state['feed_items'],
            'stories': session_state['stories'],
            'id_to_geolocs': session_state['id_to_geolocs'],
            'sf_to_geoloc': session_state['sf_to_geoloc']
        }
        download_state_str = \
            download_button(
                json.dumps(downloadable_state, indent=2, cls=DateTimeEncoder),
                download_filename=f'state.json',
                button_text='Download Current State',
                pickle_it=False
            )
        st.sidebar.markdown(download_state_str, unsafe_allow_html=True)

    ###############
    # END SIDEBAR #
    ###############


class DateTimeEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, datetime):
            return o.isoformat()
        if isinstance(o, spacy.tokens.Doc):
            return None
        return json.JSONEncoder.default(self, o)


def render_main(session_state):
    #############
    # Main Area #
    #############
    if len(session_state['feed_items']) > 0:
        st.write("# Overview")
        facet, selected = create_overview(session_state)

        if facet == FACETS.DATE_FACET:
            render_chronological_view(session_state, selected)
        elif facet == FACETS.LOC_FACET:
            render_location_view(session_state, selected)
        elif facet == FACETS.PEOPLE_FACET:
            render_people_view(session_state, selected)
        elif facet == FACETS.ORG_FACET:
            render_organisations_view(session_state, selected)
        elif facet == FACETS.CATEGORY_FACET:
            render_categories_view(session_state, selected)
        else:
            pass
    else:
        st.write("# Faceted Newsfeeds")
        st.write("### Please populate your feed in the sidebar")
        st.write("### Or choose from example feeds")


def create_overview(session_state):
    event_summary_cols = st.columns(6)

    event_summary_cols[1].metric('Stories in Feed', len(session_state['stories']))
    event_summary_cols[0].metric('Events in Feed', len(session_state['feed_items']))

    events = session_state["feed_items"].values()
    date_to_events = defaultdict(list)
    for e in events:
        date_to_events[e["start_date"].date()].append(e)

    country_to_events = group_by_country(events, session_state['id_to_geolocs'])

    people_to_events = group_by_entity(events, 'people')
    org_to_events = group_by_entity(events, 'organisations')

    # TODO: group by category (i.e. category in story)
    # TODO: session_state['visible_events']
    # TODO: updated every time a checkbox is clicked
    # TODO: don't make user click "Render"(?)
    # TODO: remove facet-specific displays(?)
    category_to_events = group_by_category(events)

    facet_to_selected = defaultdict(list)
    date_col, loc_col, people_col, org_col, category_col = st.columns((1, 1, 1, 1, 1))

    date_col.write("### Dates")
    all_dates = date_col.checkbox("all / clear", key="all-dates")
    idx = 0
    for d in sorted(date_to_events):
        d_events = date_to_events[d]
        selected = date_col.checkbox(
            format_date(d) + f" ({len(d_events)} events)",
            value=all_dates,
            key=str(idx)
        )
        idx += 1
        if selected:
            facet_to_selected[FACETS.DATE_FACET].append(d)

    loc_col.write("### Locations")
    all_locs = loc_col.checkbox("all / clear", key="all-locs")
    # TODO: configurable cutoff from sidebar for UX
    for c, c_events in sorted(country_to_events.items(), key=lambda x: len(x[1]), reverse=True):
        selected = loc_col.checkbox(
            f"{c} ({len(c_events)} events)",
            value=all_locs,
            key=f'locations-checkbox-{idx}'
        )
        idx += 1
        if selected:
            facet_to_selected[FACETS.LOC_FACET].append(c)

    people_col.write("### People")
    all_people = people_col.checkbox("all / clear", key="all-people")
    for e, e_events in sorted(
            people_to_events.items(),
            key=lambda x: len(x[1]),
            reverse=True
    ):
        sf = e[1]
        selected = people_col.checkbox(
            f"{sf} ({len(e_events)} events)",
            value=all_people,
            key=f'people-checkbox-{idx}'
        )
        idx += 1
        if selected:
            facet_to_selected[FACETS.PEOPLE_FACET].append(e)

    org_col.write("### Organisations")
    all_orgs = org_col.checkbox("all / clear", key="all-orgs")
    for e, e_events in sorted(
            org_to_events.items(), key=lambda x: len(x[1]), reverse=True):
        sf = e[1]
        selected = org_col.checkbox(
            f"{sf} ({len(e_events)} events)",
            value=all_orgs,
            key=f'org-checkbox-{idx}'
        )
        idx += 1
        if selected:
            facet_to_selected[FACETS.ORG_FACET].append(e)

    category_col.write("### Categories")
    all_categories = category_col.checkbox("all / clear", key="all-categories")
    for e, e_events in sorted(
            category_to_events.items(),
            key=lambda x: len(x[1]),
            reverse=True
    ):
        selected = category_col.checkbox(
            f"{e} ({len(e_events)} events)",
            value=all_categories,
            key=f'categories-checkbox-{idx}'
        )
        idx += 1
        if selected:
            facet_to_selected[FACETS.CATEGORY_FACET].append(e)

    # date_col, loc_col, people_col, org_col = st.columns((1, 1, 1, 1))
    facet = None
    if date_col.button("View by date"):
        facet = FACETS.DATE_FACET
    if loc_col.button("View by country"):
        facet = FACETS.LOC_FACET
    if people_col.button("View by person"):
        facet = FACETS.PEOPLE_FACET
    if org_col.button("View by organisation"):
        facet = FACETS.ORG_FACET
    if category_col.button("View by Category"):
        facet = FACETS.CATEGORY_FACET

    st.write("---")
    return facet, facet_to_selected[facet]


def most_common(items):
    return Counter(items).most_common(1)[0][0]


def format_date(d, abbr=False):
    if abbr:
        return f"{d.day} {calendar.month_abbr[d.month]} {d.year}"
    else:
        return f"{d.day} {calendar.month_name[d.month]} {d.year}"


def render_chronological_view(session_state, selected_dates):
    events = session_state["feed_items"].values()
    date_to_events = defaultdict(list)
    for e in events:
        date_to_events[e["start_date"].date()].append(e)
    for d in selected_dates:
        st.write(f"## Events on {format_date(d)}")
        for i, e in enumerate(date_to_events[d]):
            render_event_card(e, session_state)
        st.markdown("----")


def render_location_view(session_state, selected_countries):
    events = session_state["feed_items"].values()
    geolocs = session_state["sf_to_geoloc"].values()
    geolocs = [g for g in geolocs if g["address"]["country"] in selected_countries]

    df = pd.DataFrame({
        'lat': [float(g["lat"]) for g in geolocs],
        'lon': [float(g["lon"]) for g in geolocs]
    })
    st.map(df)
    country_to_events = group_by_country(events, session_state["id_to_geolocs"])
    for c in selected_countries:
        c_events = country_to_events[c]
        st.write(f"## Events in {c}")
        for i, e in enumerate(c_events):
            render_event_card(e, session_state)
        st.markdown("----")


def render_people_view(session_state, selected_people):
    events = session_state["feed_items"].values()
    entity_to_events = group_by_entity(events, "people")
    for (eid, sf) in selected_people:
        e_events = entity_to_events[(eid, sf)]
        st.write(f"## Events involving {sf}")
        st.write(f"#### Entity: {eid}")
        for i, e in enumerate(e_events):
            render_event_card(e, session_state)


def render_organisations_view(session_state, selected_orgs):
    events = session_state["feed_items"].values()
    entity_to_events = group_by_entity(events, "organisations")
    for (eid, sf) in selected_orgs:
        e_events = entity_to_events[(eid, sf)]
        st.write(f"## Events involving {sf}")
        st.write(f"#### {eid}")
        for i, e in enumerate(e_events):
            render_event_card(e, session_state)


def render_categories_view(session_state, selected_categories):
    events = session_state["feed_items"].values()
    category_to_events = group_by_category(events)
    for category_string in selected_categories:
        e_events = category_to_events[category_string]
        st.write(f"## Events involving {category_string}")
        st.write(f"#### {category_string}")
        for i, e in enumerate(e_events):
            render_event_card(e, session_state)


def group_by_country(items, id_to_geolocs):
    country_to_items = defaultdict(list)
    for item in items:
        geolocs = id_to_geolocs[item["id"]]
        item_countries = [g["address"]["country"] for g in geolocs]
        if len(item_countries) > 0:
            country = most_common(item_countries)
            country_to_items[country].append(item)
    return country_to_items


def hash_category(category):
    return f'id: {category["id"]} - Taxonomy: {category["taxonomy"]}'


def group_by_category(items):
    category_to_items = defaultdict(list)

    for item in items:
        for category in item['categories']:
            category_to_items[hash_category(category)].append(item)
    return category_to_items


def group_by_entity(items, entity_field):
    entity_to_items = defaultdict(list)
    for item in items:
        ents = item[entity_field]
        for e in ents:
            key = (e["id"], e["surface_form"])
            entity_to_items[key].append(item)
    return entity_to_items


def render_event_card(event, session_state):
    col1, col2 = st.columns([4, 2])
    col1.write(f'#### {event["title"]}')
    col1.write(f'{event["description"]}')
    col1.metric(
        'Stories', len(event['stories'])
    )
    cols = st.columns(6)
    cols[0].write('### People')
    for entity in event['people']:
        cols[0].write(entity['surface_form'])
    cols[1].write('### Locations')
    for entity in event['locations']:
        cols[1].write(entity['surface_form'])
    cols[2].write('### Organisations')
    for entity in event['organisations']:
        cols[2].write(entity['surface_form'])
    cols[3].write('### Categories')
    for category in event['categories']:
        cols[3].write(hash_category(category))

    col1, col2 = st.columns(2)
    with col1.expander('Stories'):
        for story in event['stories']:
            st.markdown(f'[{story["title"]}]({story["links"]["permalink"]})')

    col1.markdown('-----')


if __name__ == '__main__':
    render()
