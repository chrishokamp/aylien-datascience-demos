from collections import defaultdict, Counter
import arrow
from collections import defaultdict
import networkx as nx
import uuid
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import DBSCAN
from geopy.geocoders import Nominatim
from time import sleep


geolocator = Nominatim(timeout=2, user_agent="story_locations")


# TODO: switch to HDBSCAN
def cluster_items(items, get_text, eps=0.99, min_samples=2):
    vectorizer = TfidfVectorizer(stop_words="english")
    texts = [get_text(item) for item in items]
    X = vectorizer.fit_transform(texts)
    clusterer = DBSCAN(eps=eps, min_samples=min_samples)
    labels = clusterer.fit_predict(X)

    label_to_items = defaultdict(list)
    noise_clusters = []
    for item, l in zip(items, labels):
        if l == -1:
            noise_clusters.append([item])
        else:
            label_to_items[l].append(item)
    clusters = list(label_to_items.values()) + noise_clusters
    return sorted(clusters, key=len, reverse=True)


def pick_event_title(stories):
    texts = []
    labels = []
    for s in stories:
        if s["title"].strip() != "":
            labels.append(1)
            texts.append(s["title"])
        if s["body"].strip() != "":
            labels.append(0)
            texts.append(s["body"])

    vectorizer = TfidfVectorizer(stop_words="english", lowercase=True)
    X = vectorizer.fit_transform(texts)
    ranked_indices = textrank(X)
    cluster_title = None
    for i in ranked_indices:
        if labels[i] == 1:
            cluster_title = texts[i]
    return cluster_title


def textrank(vectors):
    S = cosine_similarity(vectors, vectors)
    nodes = list(range(S.shape[0]))
    graph = nx.from_numpy_matrix(S)
    pagerank = nx.pagerank(graph, weight='weight')
    scores = [pagerank[i] for i in nodes]
    rank_enum = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
    ranked_indices = [x[0] for x in rank_enum]
    return ranked_indices


def pick_event_description(stories, num_body_sents=2):
    sents = []
    for s in stories:
        sents += [str(sent) for sent in list(s["body_doc"].sents)[:num_body_sents]]
    labels = [(1 if len(s.split()) < 60 else 0) for s in sents]
    vectorizer = TfidfVectorizer(stop_words="english", lowercase=True)
    X = vectorizer.fit_transform(sents)
    ranked_indices = textrank(X)
    summary = None
    for i in ranked_indices:
        if labels[i] == 1:
            summary = sents[i]
    return summary


def extract_start_date(stories):
    times = [arrow.get(s["published_at"]).datetime for s in stories]
    return min(times)


def extract_frequent_entities(
        stories,
        allowed_types=[],
        max_num=5,
        min_ratio=0.2,
        unique_sfs=True,
    ):
    id_to_sf_counts = defaultdict(lambda: defaultdict(int))
    entity_ids = set()
    all_types = set()
    for s in stories:
        # note implicit schema match
        for text_field in ("title", "body"):
            # doc = s[f"{text_field}_doc"]
            # span_to_type = {}
            # for e in doc.ents:
            #     span = (e.start_char, e.end_char)
            #     span_to_type[span] = e.label_
            for e in s['entities']:
                all_types.update(e['types'])
                if 'links' in e:
                    eid = e['links']['wikidata']
                    for sf in e[text_field]['surface_forms']:
                        for m in sf['mentions']:
                            start = m['index']['start']
                            end = m['index']['end']
                            # e_type = span_to_type.get(span)
                            e_types = set(e['types'])
                            if len(e_types.intersection(set(allowed_types))):
                                # sf_text = str(s[text_field][start: end + 1])
                                sf_text = str(s[text_field][start: end])
                                id_to_sf_counts[eid][sf_text] += 1
                                entity_ids.add(eid)

    # aggregate entity frequencies and canonical surface forms
    if len(entity_ids) == 0:
        return []
    entity_items = []
    for eid in entity_ids:
        sf_counts = id_to_sf_counts[eid]
        frequency = sum(sf_counts.values())
        canonical_sf = max(sf_counts, key=sf_counts.get)
        entity_items.append({"id": eid, "surface_form": canonical_sf, "count": frequency})
    entity_items.sort(key=lambda x: x["count"], reverse=True)

    # only keep entities passing minimum ratio of occurrences compared to most frequent
    max_count = max([x["count"] for x in entity_items])
    entity_items = [x for x in entity_items if x["count"] / max_count >= min_ratio]

    # only keep most frequent of entities if they share canonical surface forms
    if unique_sfs:
        entity_items_filtered = []
        seen_sfs = set()
        for x in entity_items:
            if x["surface_form"] not in seen_sfs:
                entity_items_filtered.append(x)
                seen_sfs.add(x["surface_form"])
        entity_items = entity_items_filtered

    return entity_items[:max_num]


def extract_geolocations(events):

    event_id_to_geolocs = defaultdict(list)
    name_to_geoloc = {}

    for e in events:
        for loc_ent in e["locations"]:
            loc_name = loc_ent["surface_form"]
            if loc_name in name_to_geoloc:
                geoloc = name_to_geoloc[loc_name]
            else:
                coords = geolocator.geocode(loc_ent["surface_form"])
                sleep(0.25)
                if coords is not None:
                    lat, long = coords.latitude, coords.longitude
                    geoloc = geolocator.reverse(f"{lat}, {long}", language='en').raw
                    event_id_to_geolocs[e["id"]].append(geoloc)
                    name_to_geoloc[loc_name] = geoloc

    return event_id_to_geolocs, name_to_geoloc


def stories_to_event(stories):
    """
    * title
    * description
    * locations
    * people
    * organisations
    """
    title = pick_event_title(stories)
    desc = pick_event_description(stories)
    start_date = extract_start_date(stories)

    people = extract_frequent_entities(
        stories,
        allowed_types=['Human'], max_num=5, min_ratio=0.4)
    # allowed_types = ['Human'], max_num = 5, min_ratio = 0.4)

    locations = extract_frequent_entities(
        stories,
        allowed_types=['Sovereign_state', 'Community', 'State_(polity)', 'U.S._state', 'Location', 'Country', 'Island_country', 'City'],
        max_num=5, min_ratio=0.4
    )
    # {'Product_(business)', 'Software', 'Sovereign_state', 'Community', 'State_(polity)', 'Location', 'Country',
    #
    #  'Educational_organization'}
    # {'Government', 'Sovereign_state', 'Community', 'State_(polity)', 'Location', 'U.S._state', 'Country', 'Human',
    #  'Political_organisation', 'Island_country', 'Company', 'City', 'Organization', 'Corporation', 'Local_government'}

    organisations = extract_frequent_entities(
        stories,
        allowed_types=[
            'Government', 'Local_government', 'Nonprofit_organization', 'Political_organisation'
            'Public_company', 'Company', 'Business', 'Organization', 'Brick_and_mortar'
        ],
        max_num=5, min_ratio=0.4)

    event = {
        'id': str(uuid.uuid4()),
        'title': title,
        'description': desc,
        'start_date': start_date,
        'people': people,
        'locations': locations,
        'organisations': organisations,
        'stories': stories,
    }
    return event
