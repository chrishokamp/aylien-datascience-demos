import copy


def flatten_categories_to_aql(params):
    aqls = []
    for category in params['categories']:
        c_params = copy.deepcopy(params)
        c_params['categories'] = [category]
        aql = params_to_aql(c_params)
        aqls.append(aql)
    return aqls


def params_to_aql(params):
    """
    given include & exclude categories and industries, and entities,
    return the corresponding AQL
    """
    # TODO: let user compose AQL with other params?
    if 'aql' in params:
        return params['aql']

    # this is the full set of supported params
    params_schema = {
        'categories': None,
        'not_categories': None,
        'industries': None,
        'not_industries': None,
        'entity_surface_forms': None,
        'entity_ids': None,
        'entities_sentiment': None
    }
    params = dict(params_schema, **params)
    list_types = [
        'categories', 'not_categories',
        'industries', 'not_industries',
        'entity_surface_forms', 'entity_ids'
    ]
    for t in list_types:
        if params[t] is not None and type(params[t]) is not list:
            raise TypeError(f'Field {t} must be a list')
    aylien_categories_aql = make_aylien_categories_aql(
        params['categories'], params['not_categories'])
    industries_aql = make_industries_aql(
        params['industries'], params['not_industries'])
    # note difference between surface forms and entity ids
    entities_sentiment = params.get('entities_sentiment', None)

    entities_aql = make_entities_aql(
        params['entity_surface_forms'],
        params['entity_ids'],
        sentiment=entities_sentiment
    )
    aql_components = [
        c for c in [aylien_categories_aql, industries_aql, entities_aql]
        if len(c) > 0
    ]
    return ' AND '.join(aql_components)


def make_entities_aql(surface_forms, entity_ids, sentiment=None, min_prominence=0.7):
    sfs_aql = ''
    if surface_forms is not None and len(surface_forms) > 0:
        sfs_aql = f'surface_forms.text: (' + ' '.join([f'"{sf}"' for sf in surface_forms]) + ')'

    ids_aql = ''
    if entity_ids is not None and len(entity_ids) > 0:
        ids_aql = f' (' + ' '.join([f'id:{entity_id}' for entity_id in entity_ids]) + ')'

    entities_aql = ''
    if len(sfs_aql) or len(ids_aql):
        prominence_aql = ''
        if min_prominence is not None:
            prominence_aql = f'prominence_score:[{min_prominence} TO *]'
        sentiment_aql = ''
        if sentiment is not None and sentiment in ['positive', 'negative']:
            sentiment_aql = f'sentiment:{sentiment}'

        entities_aql_components = [c for c in [prominence_aql, sentiment_aql, sfs_aql, ids_aql]
                                   if len(c) > 0]
        entities_aql = 'entities: {{' + ' AND '.join(entities_aql_components)
        entities_aql += ' sort_by(overall_prominence)}}'
    return entities_aql


def make_text_query(include_texts=None, exclude_texts=None):
    include_string = ''
    if include_texts is not None and len(include_texts) > 0:
        for string in include_texts:
            include_string += '"' + string + '" '
        include_string = '(' + include_string + ')'

    exclude_string = ''
    if exclude_texts is not None and len(exclude_texts) > 0:
        for string in exclude_texts:
            exclude_string += '"' + string + '" '
        exclude_string = ' NOT (' + exclude_string + ')'

    return include_string + exclude_string


def make_industries_aql(include_industries=None, exclude_industries=None):
    if include_industries is not None and len(include_industries) > 0:
        include_industries_aql = 'industries:{{score:[0.7 TO *] AND ('
        include_industries_aql += include_industries[0]
        for tag in include_industries[1:]:
            include_industries_aql += ' ' + tag
        include_industries_aql += ')}}'
    else:
        include_industries_aql = ''

    if exclude_industries is not None and len(exclude_industries) > 0:
        exclude_industries_aql = ' NOT industries:{{('
        exclude_industries_aql += exclude_industries[0]
        for tag in exclude_industries[1:]:
            exclude_industries_aql += f' {tag}'
        exclude_industries_aql += ')}}'
    else:
        exclude_industries_aql = ''

    industries_aql = ''
    if len(include_industries_aql) or len(exclude_industries_aql):
        industries_aql = '(' + include_industries_aql + exclude_industries_aql + ')'
    return industries_aql


def make_aylien_categories_aql(include_categories=None, exclude_categories=None):
    if include_categories is not None and len(include_categories) > 0:
        include_categories_aql = 'categories:{{taxonomy:aylien AND score:[0.7 TO *] AND id:('
        include_categories_aql += include_categories[0]
        for tag in include_categories[1:]:
            include_categories_aql += f' {tag}'
        include_categories_aql += ')}}'
    else:
        include_categories_aql = ''

    if exclude_categories is not None and len(exclude_categories) > 0:
        exclude_categories_aql = ' NOT categories:{{taxonomy:aylien AND id:('
        exclude_categories_aql += exclude_categories[0]
        for tag in exclude_categories[1:]:
            exclude_categories_aql += f' {tag}'
        exclude_categories_aql += ')}}'
    else:
        exclude_categories_aql = ''

    categories_aql = ''
    if len(include_categories_aql) > 0 or len(exclude_categories_aql):
        categories_aql = '(' + include_categories_aql + exclude_categories_aql + ')'
    return categories_aql
