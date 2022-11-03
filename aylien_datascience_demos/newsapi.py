import json
import requests
import time
from copy import deepcopy
import os

from .aql_builder import params_to_aql


HEADERS = {
    'X-AYLIEN-NewsAPI-Application-ID': os.getenv('NEWSAPI_APP_ID'),
    'X-AYLIEN-NewsAPI-Application-Key': os.getenv('NEWSAPI_APP_KEY')
}
STORIES_ENDPOINT = 'https://api.aylien.com/news/stories'
CLUSTERS_ENDPOINT = 'https://api.aylien.com/news/clusters'
TRENDS_ENDPOINT = 'https://api.aylien.com/news/trends'
TIMESERIES_ENDPOINT = 'https://api.aylien.com/news/time_series'


class TimeseriesEndpointError(Exception):
    pass


def create_newsapi_query(params):
    template = {
        "language": "en",
        "period": "+1DAY"
    }
    aql = params_to_aql(params)
    return dict(template, **{'aql': aql})


def retrieve_stories(params,
                     n_pages=1,
                     headers=HEADERS,
                     endpoint=STORIES_ENDPOINT,
                     sleep=None):
    params = deepcopy(params)
    stories = []
    cursor = '*'
    for i in range(n_pages):
        params['cursor'] = cursor
        response = requests.get(
            endpoint,
            params,
            headers=headers
        )

        data = json.loads(response.text)
        stories += data['stories']
        if data.get('next_page_cursor', '*') != cursor:
            cursor = data['next_page_cursor']
            if sleep is not None:
                time.sleep(sleep)
        else:
            break
    return stories


def retrieve_clusters(cluster_params,
                      story_params=None,
                      get_stories=False,
                      n_cluster_pages=1,
                      n_story_pages=1,
                      headers=HEADERS,
                      clusters_endpoint=CLUSTERS_ENDPOINT,
                      stories_endpoint=STORIES_ENDPOINT,
                      sleep=None):

    cluster_params = deepcopy(cluster_params)
    clusters = []
    cursor = '*'
    for i in range(n_cluster_pages):
        cluster_params['cursor'] = cursor
        response = requests.get(
            clusters_endpoint,
            params=cluster_params,
            headers=headers
        )
        data = json.loads(response.text)
        clusters += data['clusters']
        if data['next_page_cursor'] != cursor:
            cursor = data['next_page_cursor']
            if sleep is not None:
                time.sleep(sleep)
        else:
            break

    if get_stories:
        story_params = deepcopy(story_params)
        for i, c in enumerate(clusters):
            # print(f'getting stories for cluster {i+1}/{n_clusters}')
            story_params['clusters'] = [c['id']]
            stories = retrieve_stories(
                params=story_params,
                headers=headers,
                endpoint=stories_endpoint,
                n_pages=n_story_pages,
                sleep=sleep,
            )
            c['stories'] = stories
    return clusters


def retrieve_timeseries(
        params,
        headers=HEADERS,
        endpoint=TIMESERIES_ENDPOINT):
    response = requests.get(
        endpoint,
        params,
        headers=headers
    )
    r = json.loads(response.text)
    if 'time_series' not in r:
        raise TimeseriesEndpointError(r)
    return r
