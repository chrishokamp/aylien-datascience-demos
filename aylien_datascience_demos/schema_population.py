from collections import defaultdict, Counter
import json
import arrow
from collections import defaultdict
import networkx as nx
import uuid

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import DBSCAN
from geopy.geocoders import Nominatim
from time import sleep


geolocator = Nominatim(timeout=2, user_agent="story_locations")


def cluster_items(items, get_text, eps, min_samples):
    vectorizer = TfidfVectorizer(stop_words="english")
    texts = [get_text(item) for item in items]
    X = vectorizer.fit_transform(texts)
    clusterer = DBSCAN(eps=eps, min_samples=min_samples, metric='cosine')
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
        # we add first body sentence to improve textrank,
        # but it can't be the best title
        if len(s["body"].strip()):
            labels.append(0)
            texts.append(str(list(s["body_doc"].sents)[0]))

    vectorizer = TfidfVectorizer(stop_words="english")
    X = vectorizer.fit_transform(texts)
    ranked_indices, ranked_scores = textrank(X)
    cluster_title = None
    for i in ranked_indices:
        if labels[i] == 1:
            cluster_title = texts[i]
    return cluster_title


def textrank(vectors, min_sim=0.3):
    S = cosine_similarity(vectors, vectors)
    np.fill_diagonal(S, 0.)
    S[S < min_sim] = 0.
    nodes = list(range(S.shape[0]))
    graph = nx.from_numpy_matrix(S)

    pagerank = nx.pagerank(graph, weight='weight')
    scores = [pagerank[i] for i in nodes]
    return zip(*sorted(enumerate(scores), key=lambda x: x[1], reverse=True))


def pick_event_description(stories, num_body_sents=2):
    sents = []
    for s in stories:
        sents += [str(sent) for sent in list(s["body_doc"].sents)[:num_body_sents]]
    labels = [(1 if len(s.split()) < 60 else 0) for s in sents]
    vectorizer = TfidfVectorizer(stop_words="english", lowercase=True)
    X = vectorizer.fit_transform(sents)
    ranked_indices, ranked_scores = textrank(X)
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
            for e in s['entities']:
                all_types.update(e['types'])
                if 'links' in e:
                    eid = e['links']['wikidata']
                    for sf in e[text_field]['surface_forms']:
                        for m in sf['mentions']:
                            start = m['index']['start']
                            end = m['index']['end']
                            e_types = set(e['types'])
                            if len(e_types.intersection(set(allowed_types))):
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


# TODO: this method is slow, improve efficiency
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


def category_to_json(category):
    return json.dumps(
        {
            'id': category['id'],
            'taxonomy': category['taxonomy']
        }
    )


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

    # Categories
    # TODO: add taxonomy relations(?) (parent / child)
    category_counts = Counter()
    for s in stories:
        for c in s['categories']:
            # currently only Aylien Taxonomy allowed thru
            if 'ay.' in c['id']:
                category_counts.update([category_to_json(c)])
    categories = []
    for category_json, count in category_counts.most_common():
        category = json.loads(category_json)
        category['count_in_cluster']: count
        categories.append(category)

    people = extract_frequent_entities(
        stories,
        allowed_types=['Human'], max_num=5, min_ratio=0.4)

    locations = extract_frequent_entities(
        stories,
        allowed_types=['Government', 'Local_government', 'Sovereign_state', 'Community', 'State_(polity)', 'U.S._state', 'Location', 'Country', 'Island_country', 'City'],
        max_num=5, min_ratio=0.4
    )

    organisations = extract_frequent_entities(
        stories,
        allowed_types=[
            'Public_company', 'Company', 'Business', 'Brick_and_mortar'
        ],
        max_num=5, min_ratio=0.4)

    event = {
        'id': str(uuid.uuid4()),
        'title': title,
        'description': desc,
        'start_date': start_date,
        'categories': categories,
        'people': people,
        'locations': locations,
        'organisations': organisations,
        'categories': categories,
        'stories': stories,
    }
    return event
