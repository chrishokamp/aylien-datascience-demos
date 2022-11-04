import streamlit as st
from pathlib import Path
import json
from PIL import Image
import os
import copy
import time
import calendar
import hashlib
from collections import OrderedDict, defaultdict, Counter

import spacy

import aylien_datascience_demos.schema_population as schema_population
import aylien_datascience_demos.newsapi as newsapi
import streamlit.components.v1 as components
from collections import namedtuple

import pandas as pd


FACET_MAP = {
  'LOC_FACET': 'locations',
  'PEOPLE_FACET': 'people',
  'DATE_FACET': 'dates',
  'ORG_FACET': 'organisations'
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


# TODO: cache resolved events, processing is expensive


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
        # Initialize user state if session doesn't exist
        # Create a Tokenizer with the default settings for English
        # including punctuation rules and exceptions
        nlp = spacy.load("en_core_web_sm")
        # TODO: for online views, let user get more stories
        query_template = {
          "text": "Takeoff",
          "per_page": 25,
          "published_at.start": "NOW-7DAYS",
          "sort_by": "relevance",
          "language": "en"
        }
        # TODO: let user specify the number of _items_ they want to get,
        # this is important in the context of online event extraction

        state['local_nlp'] = nlp
        state['current_newsapi_query'] = copy.deepcopy(query_template)
        state['feed_items'] = OrderedDict()

        state['newsapi'] = Newsapi()
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

    ###########
    # SIDEBAR #
    ###########
    # st.sidebar.markdown('#### Dynamic Event Views')

    ###################################
    # Configure Input Streams/Sources #
    ###################################
    # user can load or paste generated queries from file
    st.sidebar.markdown('### Input Sources')
    if st.sidebar.button('Clear Feed'):
        session_state['feed_items'] = OrderedDict()
        session_state['events'] = OrderedDict()
        session_state['id_to_geolocs'] = None
        st.experimental_rerun()
    st.sidebar.markdown('----')

    # Newsapi Query
    # TODO: query cache with some data loaded up on startup
    query = st.sidebar.text_area(
        'NewsAPI Query',
        json.dumps(session_state['current_newsapi_query'], indent=2),
        height=350
    )
    if st.sidebar.button('Populate Feed from Query', key='populate_feed_from_query'):
        query = session_state['current_newsapi_query']
        stories = session_state['newsapi'].retrieve_stories(
            params=query
        )
        for s in stories:
            s["title_doc"] = session_state["local_nlp"](s["title"])
            s["body_doc"] = session_state["local_nlp"](s["body"])
        clusters = schema_population.cluster_items(
            stories,
            get_text=(lambda x: f"{str(x['title_doc'])} {str(list(x['body_doc'].sents)[:3])}"),
            min_samples=2,
            eps=0.75
        )
        events = [schema_population.stories_to_event(c) for c in clusters]
        id_to_geolocs, sf_to_geoloc =\
            schema_population.extract_geolocations(events)
        # TODO: cache events for query, don't recompute
        # TODO: assert events are json serializable
        for e in events:
            session_state["feed_items"][e["id"]] = e
        session_state["id_to_geolocs"] = id_to_geolocs
        session_state["sf_to_geoloc"] = sf_to_geoloc
        st.experimental_rerun()

    st.sidebar.markdown('----')

    session_state['current_newsapi_query'] = json.loads(query)

    ###############
    # END SIDEBAR #
    ###############

    #############
    # Main Area #
    #############

    ###############################################
    # Map items into views (Events, Entities, ... #
    # ("Event" is another type of transient Item) #
    ###############################################

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
        else:
            pass
    else:
        st.write("# Breaking a newsfeed into facets: time, location, entities")

    #################
    # End View Area #
    #################


def create_overview(session_state):
    events = session_state["feed_items"].values()
    date_to_events = defaultdict(list)
    for e in events:
        date_to_events[e["start_date"].date()].append(e)

    geolocs = session_state["sf_to_geoloc"].values()
    country_to_events = group_by_country(events, session_state["id_to_geolocs"])

    people_to_events = group_by_entity(events, "people")
    org_to_events = group_by_entity(events, "organisations")
    # TODO: group by category (i.e. category in story)

    facet_to_selected = defaultdict(list)
    date_col, loc_col, people_col, org_col = st.columns((1, 1, 1, 1))

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

    geolocs = session_state["sf_to_geoloc"].values()
    df = pd.DataFrame({
        'lat' : [float(g["lat"]) for g in geolocs],
        'lon' : [float(g["lon"]) for g in geolocs]
    })

    loc_col.write("### Locations")
    all_locs = loc_col.checkbox("all / clear", key="all-locs")
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
    for e, e_events in sorted(people_to_events.items(), key=lambda x: len(x[1]), reverse=True):
        sf = e[1]
        # if people_col.button(f"{sf} ({len(d_events)} events)", key=f"per-{sf}"):
        #     pass
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
    for e, e_events in sorted(org_to_events.items(), key=lambda x: len(x[1]), reverse=True):
        sf = e[1]
        selected = org_col.checkbox(
            f"{sf} ({len(e_events)} events)",
            value=all_orgs,
            key=f'org-checkbox-{idx}'
        )
        idx += 1
        if selected:
            facet_to_selected[FACETS.ORG_FACET].append(e)

    date_col, loc_col, people_col, org_col = st.columns((1, 1, 1, 1))
    facet = None
    if date_col.button("View by date"):
        facet = FACETS.DATE_FACET
    if loc_col.button("View by country"):
        facet = FACETS.LOC_FACET
    if people_col.button("View by person"):
        facet = FACETS.PEOPLE_FACET
    if org_col.button("View by organisation"):
        facet = FACETS.ORG_FACET

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
        st.write(f"Entity: {eid}")
        for i, e in enumerate(e_events):
            render_event_card(e, session_state)


def render_organisations_view(session_state, selected_orgs):
    events = session_state["feed_items"].values()
    entity_to_events = group_by_entity(events, "organisations")
    for (eid, sf) in selected_orgs:
        e_events = entity_to_events[(eid, sf)]
        st.write(f"## Events involving {sf}")
        st.write(f"{eid}")
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


def group_by_entity(items, entity_field):
    entity_to_items = defaultdict(list)
    for item in items:
        ents = item[entity_field]
        for e in ents:
            key = (e["id"], e["surface_form"])
            entity_to_items[key].append(item)
    return entity_to_items


def render_event_card(event, session_state):
    col1, col2 = st.columns(2)
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

    col1, col2 = st.columns(2)
    with col1.expander('Stories'):
        for story in event['stories']:
            st.markdown(f'[{story["title"]}]({story["links"]["permalink"]})')

    col1.markdown('-----')


def render_event(
        item,
        session_state,
    ):
    stories = item["stories"]
    # if item.get('start_date', None) is not None:
    #     date_col.markdown(f'Date: `{item["start_date"].date()}`')
    # num_stories_col.markdown(f"Number of stories: `{len(stories)}`")
    #
    # st.markdown(f'#### {item["title"]}')
    # st.write(item["description"])

    people_string = 'PEOPLE: ' + ", ".join([str(e["surface_form"]) for e in item["people"]])
    loc_string = 'LOCATIONS: ' + ", ".join([str(e["surface_form"]) for e in item["locations"]])
    org_string = 'ORGS: ' + ", ".join([str(e["surface_form"]) for e in item["organisations"]])
    st.write(people_string)
    st.write(loc_string)
    st.write(org_string)

    dedup_stories = []
    seen = set()
    for s in stories:
        # if s["links"]["permalink"]:
        if s["title"] not in seen:
            dedup_stories.append(s)
            seen.add(s["title"])

    titles_string = "\n".join([
        f'<p><a style="color: #ffab03 !important;" href={s["links"]["permalink"]}>{s["title"]}</a></p>'
        for s in dedup_stories[:3]
    ])
    # attr_style = 'style="color: #7d7d7d"'
    attr_style = 'style="color: black"'
    components.html(
        f"""
        <div style="border: 3px solid #ffab03; color: white; border-radius: 15px; margin-bottom: 10px;">
        <div style="padding: 15px 15px 0px 15px; font-family: sans-serif">
        <h3 style="text-align: center">{item["title"]}</h3>
        <p style="text-align: center">{item["description"]}</p>
        <hr style="width:100%; text-align:left; margin-left:0">
        <p><b {attr_style}>Stories:</b> {len(stories)}</p>
        <p><b {attr_style}>People:</b> {people_string}</p>
        <p><b {attr_style}>Locations:</b> {loc_string}</p>
        <p><b {attr_style}>Organisations:</b> {org_string}</p>
        <hr style="width:100%; text-align:left; margin-left:0">
        {titles_string}
        </div>
        </div>
        """,
        height=430,
        width=800,
    )
    # st.markdown('---------------------')


def render_aylien_event(
        item,
        session_state,
        _st=None
    ):
    id = item_id(item)

    if _st is None:
        _st = st

    # visualizer config
    # these are the same ents in the Aylien EL config
    # get all local user NER labels on the fly by looking at all of user's patterns
    # global_labels = [r['label'] for r in session_state['global_rules'].values()]
    # local_labels = [r['label'] for r in session_state['local_rules'][id].values()]
    # user_labels = list(set(global_labels + local_labels))

    # extract attributes from this item
    date_col, article_col, source_col, link_col = st.columns([2, 2, 2, 1])
    # if this is an article from a newsAPI source, show the id
    if item.get('published_at', None) is not None:
        date_col.markdown(f'Published at: {item["published_at"]}')
    if item.get('published_at', None) is not None:
        article_col.markdown(f'`story_id: {item["id"]}`')
    if item.get('source', None) is not None:
        source_col.markdown(f'Source: [{item["source"]["name"]}]({item["source"]["home_page_url"]})')
    if item.get('links', {}).get('permalink', None) is not None:
        link_col.markdown(f'[Permalink]({item["links"]["permalink"]})')

    _st.markdown(f'#### {item["title"]}')
    _st.markdown('#### Entities:')
    _st.markdown(", ".join([str(e) for e in item["ents"]]))
    _st.markdown('#### Sub-events:')
    body_md = '\n'.join([f'- {s}' for s in item['sents']])
    _st.markdown(body_md)

    _st.write('\n')
    col1, col2, _, _, col3 = st.columns([2, 2, 1, 1, 1])
    if col1.button('Remove Document', key=f'skip_doc_button_{id}'):
        col3.write('Document Removed')
        del session_state['feed_items'][id]
        time.sleep(0.4)
        st.experimental_rerun()

    _st.markdown('---------------------')


if __name__ == '__main__':
    render()
