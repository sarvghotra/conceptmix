import json
import os
# from openai import OpenAI
try:
    from openai import AzureOpenAI
except ImportError:
    AzureOpenAI = None
import os
import json
import random
import time
import base64
import re
import requests
import inflect
import functools
import threading
import collections
inflect_engine = inflect.engine()

def load_all_configs(config_path, specified_skill=None):
    config_files = [f for f in os.listdir(config_path) if f.endswith('.json')]

    skills, descriptions, categories = [], [], []

    for config_file in config_files:
        file_path = os.path.join(config_path, config_file)
        with open(file_path, 'r') as file:
            data = json.load(file)
            for skill in data['skill'].keys():
                description = data['skill'][skill]
                if not description:
                    description = data['description'].format(skill=skill)
                skills.append(skill)
                descriptions.append(description)
                categories.append(data['category'])

    return skills, descriptions, categories

def load_configs(k, config_path, specified_skill=None):
    # config_files = ['object.json'] + [f for f in os.listdir(config_path) if f.endswith('.json')]
    config_files = [f for f in os.listdir(config_path) if f.endswith('.json')]
    if k > len(config_files):
        raise ValueError("k is larger than or equal to the available config files, leaving no files for additional words")

    descriptions = []
    skills = []
    categories = []

    if not specified_skill:
        random_config_files = [specified_skill + '.json']
    else:
        raise NotImplementedError
        # note that random.sample can only select from two different configs,
        # if we allow same configs, we need to use random.choices(config_files, k=k)
    # debug:


    with open(os.path.join(config_path, 'object.json'), 'r') as file:
        data = json.load(file)
        object = random.choice(list(data['skill'].keys()))
        print("object:", object)

    for i in range(k):
        file_path = os.path.join(config_path, random_config_files[i])
        with open(file_path, 'r') as file:
            data = json.load(file)
            skill = random.choice(list(data['skill'].keys()))

            description = data['skill'][skill]
            if not description:
                description = data['description'].format(skill=skill, object=object)
            skills.append(skill)
            descriptions.append(description)
            categories.append(data['category'])
    print("skills: [" + ", ".join(f"{category}: '{element}'" for category, element in zip(categories, skills)) + "]")
    print("descriptions: [" + ", ".join(f"'{element}'" for element in descriptions) + "]")
    print("=====")
    return skills, descriptions, categories

@functools.lru_cache(maxsize=None)
def _load_config_file(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data, tuple(data['skill'].keys())

def load_random_value(config_path, config_file):
    data, skills = _load_config_file(os.path.join(config_path, config_file))
    skill = random.choice(skills)
    description = data['skill'][skill]
    if not description:
        description = data['description']
    category = data['category']
    return skill, description, category

def load_sequential_value(config_path, config_file):
    file_path = os.path.join(config_path, config_file)
    skill_list = []

    with open(file_path, 'r') as file:
        data = json.load(file)
        total_skills = len(data['skill'])
        for i in range(len(data['skill'])):
            skill = list(data['skill'].keys())[i]
            skill_list.append(skill)
        description = data['skill'][skill]
        if not description:
            description = data['description']
        category = data['category']
    return total_skills, skill_list, description, category

# categories that relate two objects, and the key of their entries in the config
RELATION_KEYS = {'spatial': 'relation', 'verb': 'action', 'comparison': 'comparison'}
SMALL_SIZES = {'tiny', 'small', 'miniature'}

# attribute values that are implausible for people and animals ("a mossy mechanic", "a star-shaped horse").
# Other attributes are moved to a fitting object by the LLM (see refine_config_notes), but the two
# instances of a differentiation must differ in the attribute itself, so it is only sampled where it fits.
OBJECT_KINDS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'object_kinds.json')

@functools.lru_cache(maxsize=None)
def _excluded_attributes():
    """item -> {category: "all" or [values]} that differentiation does not give it."""
    with open(OBJECT_KINDS_PATH) as file:
        kinds = json.load(file)
    return {item: kind['exclude'] for kind in kinds.values() if isinstance(kind, dict) for item in kind['items']}

def attribute_fits(item, category, value):
    """Whether an object of this item can plausibly have this attribute value."""
    excluded = _excluded_attributes().get(item, {}).get(category)
    return excluded is None or (excluded != 'all' and value not in excluded)

class NoFittingObject(Exception):
    """A sampled differentiation fits none of the sampled objects; the config is sampled again."""

def attribute_phrase(category, value):
    """How an attribute value reads before its noun: 'red', 'star-shaped'."""
    return f"{value}-shaped" if category == 'shape' else str(value)

def refine_config_notes(config):
    """Instructions for the entries of the newer categories that a config contains."""
    notes = []
    if config.get('action') or config.get('comparison'):
        notes.append('"action" and "comparison" entries relate ObjectA to ObjectB by "id", as their "description" says, the same way as "relation" entries. ObjectA and ObjectB must be different objects.')
    for negation in config.get('negation', []):
        if 'item' in negation:
            notes.append('A "negation" entry with "item" names one kind of object that must be absent from the image: fill it with a singular noun (e.g. "chair") for something one would normally expect in this scene, which is not one of the "objects".')
            break
    if any('part' in n for n in config.get('negation', [])):
        notes.append('A "negation" entry with "part" names a part that the object with "ObjectA_id" normally has but is missing in the image (e.g. "a handle" for a mug).')
    if any('text' in o for o in config['objects']):
        notes.append('"text" of an object is the exact text written on it, of the kind given by its "text_kind"; make it short and fitting for the object and the scene, like something actually printed, painted or engraved on it, and not a word for the object itself or its attributes (not "ENORMOUS" on a huge toothpick). Keep "text_kind".')
    if any(c in o for o in config['objects'] for c in ('color', 'texture', 'shape', 'text_kind')):
        notes.append('This is a third case in which you may add an object and change the JSON beyond "?": an attribute ("color", "texture", "shape", or "text_kind" with "text") that does not make sense for its object, even loosely, such as a person who is "mossy", "plastic", "teal" or "star-shaped" or has text written on them, or an animal that is "knitted" or "cube-shaped". '
                     'Move such an attribute, with its value unchanged, to an object it fits: an existing object that does not have that key yet, or else a new object you add that goes with the original one, such as something it wears, holds or uses (a mossy jacket for a mechanic, a teal robe for a judge, a t-shirt with the text for a golfer). An added object carries the moved attribute besides "id" and "item", and "belongs_to" with the "id" of the object it goes with. '
                     'Explain each move. Keep attributes that make sense in a loose or imaginative sense (a crystal guitar, a square cloud, a purple dog), and never move "number" or "size".')
    items = [o['item'] for o in config['objects']]
    if len(set(items)) < len(items):
        notes.append('Objects with the same "item" are separate instances that differ in an attribute; keep them as they are.')
    if not notes:
        return ''
    return '\nNotes on the JSON:\n' + ''.join(f' * {n}\n' for n in notes)

def sample_initial_config(k, config_path, config_files, weights, specified_skill=None, object_prob=0.25):
    """Objects and skills of a config, before the LLM fills its "?" (see load_configs_new)."""
    # decide object number
    num_objects = 1
    if not specified_skill:
        for i in range(k):
            if random.random() < object_prob:
                num_objects += 1

    # each category can be used at most num_objects times (style at most once), so for
    # large k with few objects the rejection loop below never terminates; add objects
    # until the remaining skills can fit
    capacity = lambda n: sum(1 if f == 'style.json' else n for f in config_files)
    while not specified_skill and k + 1 - num_objects > capacity(num_objects):
        num_objects += 1

    k_other = k + 1 - num_objects
    if specified_skill is not None:
        random_config_files = [specified_skill + '.json']
    else:
        random_config_files = random.choices(config_files, weights=weights, k=k_other)

    max_count = lambda x: max([x.count(i) for i in set(x)]) if len(x) > 0 else 0

    while max_count(random_config_files) > num_objects or random_config_files.count('style.json') > 1:
        random_config_files = random.choices(config_files, weights=weights, k=k_other)


    skills = []
    descriptions = []
    categories = []
    initail_config = {'objects': []}
    # generate object
    for i in range(num_objects):
        skill, description, category = load_random_value(config_path, 'object.json')
        # make sure skill is not repeated
        while skill in skills:
            skill, description, category = load_random_value(config_path, 'object.json')
        skill_pl = inflect_engine.plural(skill)
        description = description.format(skill=skill, skills=skill_pl)
        initail_config['objects'].append({'id': i+1, 'item': skill})
        skills.append(skill)
        descriptions.append(description)
        categories.append(category)

    sequential = True if specified_skill else False
    differentiations = []  # (attribute, description), applied once the other attributes are placed
    placements = {}  # index in descriptions of an attribute -> index of the object it was given to
    for i in range(k_other):
        skill, description, category = load_random_value(config_path, random_config_files[i])
        if category == 'style':
            initail_config['style'] = skill
            description = description.format(skill=skill)
            skills.append(skill)
            descriptions.append(description)
            categories.append(category)
        elif category in RELATION_KEYS:
            initail_config.setdefault(RELATION_KEYS[category], []).append({'name': skill, 'description': description, 'ObjectA_id': '?', 'ObjectB_id': '?'})
        elif category == 'negation':
            # the LLM picks the absent object or missing part; described after refinement
            negation = {'name': skill, 'description': description}
            if skill == 'missing part':
                negation.update(ObjectA_id=random.randrange(num_objects) + 1, part='?')
            else:
                negation['item'] = '?'
            initail_config.setdefault('negation', []).append(negation)
        elif category == 'differentiation':
            differentiations.append((skill, description))
        else:
            idx = random.choice(range(num_objects))
            while category in initail_config['objects'][idx]:
                idx = random.choice(range(num_objects))
            if category == 'text':
                # the LLM writes the text itself, of the sampled kind
                initail_config['objects'][idx].update(text_kind=skill, text='?')
            else:
                initail_config['objects'][idx][category] = skill
            object_single = initail_config['objects'][idx]['item']
            object_plural = inflect_engine.plural(object_single)
            description = description.format(skill=skill, object=object_single, objects=object_plural)
            placements[len(descriptions)] = idx
            skills.append(skill)
            descriptions.append(description)
            categories.append(category)

    # differentiation: a second instance of a sampled object that differs from it in one
    # attribute ("one red cup and one blue cup"); geneval2_vqa tells them apart by it
    twinned = set()
    for attribute, description in differentiations:
        _, all_values = _load_config_file(os.path.join(config_path, attribute + '.json'))
        fitting_values = lambda o: [v for v in all_values if attribute_fits(o['item'], attribute, v)]
        candidates = [i for i in range(num_objects) if i not in twinned and len(fitting_values(initail_config['objects'][i])) > 1]
        if not candidates:
            raise NoFittingObject
        # a counted object ("three cups") plus a differing twin is easily written ambiguously
        uncounted = [i for i in candidates if 'number' not in initail_config['objects'][i]]
        idx = random.choice(uncounted or candidates)
        twinned.add(idx)
        obj = initail_config['objects'][idx]
        values = fitting_values(obj)
        value_a = obj.get(attribute) or random.choice(values)
        # two sizes from the same end of the scale ("small", "tiny") are hard to tell apart
        value_b = random.choice([v for v in values if v != value_a and (attribute != 'size' or (v in SMALL_SIZES) != (value_a in SMALL_SIZES))])
        obj[attribute] = value_a
        initail_config['objects'].append({'id': len(initail_config['objects']) + 1, 'item': obj['item'], attribute: value_b})
        description = description.format(skill=attribute, object=obj['item'], objects=inflect_engine.plural(obj['item']),
                                         value_a=inflect_engine.a(attribute_phrase(attribute, value_a)),
                                         value_b=inflect_engine.a(attribute_phrase(attribute, value_b)))
        skills.append(attribute)
        descriptions.append(description)
        categories.append('differentiation')

    return skills, descriptions, categories, initail_config, num_objects, placements

def load_configs_new(chat_api, k, config_path, specified_skill=None, object_prob=0.25, category_weights=None):
    if specified_skill:
        assert(k==1)
    config_files = [f for f in os.listdir(config_path) if f.endswith('.json') and f != 'object.json']
    # relative sampling weight per category, e.g. {'number': 3}; unlisted categories get 1
    weights = [(category_weights or {}).get(f[:-len('.json')], 1.0) for f in config_files]

    # sampled again until every differentiation fits an object (see attribute_fits)
    for _ in range(1000):
        try:
            skills, descriptions, categories, initail_config, num_objects, placements = sample_initial_config(k, config_path, config_files, weights, specified_skill, object_prob)
            break
        except NoFittingObject:
            continue
    else:
        raise ValueError("no sampled config where every differentiation fits an object")

    model = "gpt-4o-2024-05-13"
    refine_config_prompt = 'I am trying to create an image containing exactly the following things in a JSON format:\n' + json.dumps(initail_config) + '\n' + 'Could you check if there is "?" left in the JSON? If so, could you fill in the missing part? Make sure it makes sense when you fill the missing part. Do not fill in anything else unless it is indicated by "?". You may add additional objects, but only in the following two cases:\n * It is needed to fill in any "?" (Note when you fill "?", you should use existing objects first. If you still choose to add an object, explain why the existing objects cannot fulfill the need.); or \n * If there is an attribute specified in the JSON that contains relative information (e.g. "size") and there is no other object for reference. (The reason for adding an object for this case is because one cannot tell whether an object is huge without any other object in the image, but we are fine if there is no such attribute mentioned in the JSON. Note other existing object in JSON can be used for reference, and the reference object does not need to be the same object. If you still choose to add an object, explain why the existing objects cannot fulfill the need.)\nDO NOT add any object if none of the above situation is strictly satisifed, and DO NOT try to improve the image in other ways. If you choose to add an object, make sure it fits in the image naturally. Please only add the necessary objects, and the added objects should only have "id" and "item" specified, and should be appended to "objects".'
    refine_config_prompt += refine_config_notes(initail_config)
    completion = chat_api(
        model = model,
        messages = [
            {"role": "user", "content": refine_config_prompt}
        ]
    )
    full_response = completion.choices[0].message.content
    # print(refine_config_prompt)
    print(json.dumps(initail_config))
    print(full_response)
    json_only_prompt = 'Could you respond only the JSON?'
    completion = chat_api(
        model = model,
        messages = [
            {"role": "user", "content": refine_config_prompt},
            {"role": "assistant", "content": full_response},
            {"role": "user", "content": json_only_prompt}
        ]
    )
    json_response = completion.choices[0].message.content
    # print(json_response)
    # check if it starts with ```json and ends with ```, remove if so
    if json_response.startswith('```json'):
        json_response = json_response[7:]
    if json_response.endswith('```'):
        json_response = json_response[:-3]
    try:
        refined_config = json.loads(json_response)
    except:
        print("json_response:", json_response)
        raise ValueError("The refined config is not a valid JSON format")

    # check if id is 1, 2, 3, ...
    for i, object in enumerate(refined_config['objects']):
        assert(int(object['id']) == i + 1)
    extra_skills = []
    extra_descriptions = []
    extra_categories = []
    for object in refined_config['objects'][num_objects:]:
        extra_skills.append(object['item'])
        extra_descriptions.append(f"the image contain one {object['item']}.")
        extra_categories.append('object')
    # attributes the LLM moved from an object they do not fit ("a mossy mechanic" -> his mossy jacket)
    for pos, idx in placements.items():
        category, skill = categories[pos], skills[pos]
        key = 'text_kind' if category == 'text' else category
        holders = [o for o in refined_config['objects'] if str(o.get(key)) == str(skill)]
        if not holders or any(o['item'] == initail_config['objects'][idx]['item'] for o in holders):
            continue  # kept, or dropped (config_has_skills then rejects the config)
        template = _load_config_file(os.path.join(config_path, category + '.json'))[0]
        template = template['skill'][skill] or template['description']
        item = holders[0]['item']
        descriptions[pos] = template.format(skill=skill, object=item, objects=inflect_engine.plural(item))
    for category, key in RELATION_KEYS.items():
        for relation in refined_config.get(key, []):
            objectA = refined_config['objects'][int(relation['ObjectA_id'])-1]['item']
            objectB = refined_config['objects'][int(relation['ObjectB_id'])-1]['item']
            objectAs, objectBs = inflect_engine.plural(objectA), inflect_engine.plural(objectB)
            if objectA == objectB:
                objectB = "another " + objectB
            description = relation['description'].format(ObjectA=objectA, ObjectB=objectB, ObjectAs=objectAs, ObjectBs=objectBs)
            skills.append(relation['name'])
            descriptions.append(description)
            categories.append(category)
    for negation in refined_config.get('negation', []):
        if 'part' in negation:
            object_single = refined_config['objects'][int(negation['ObjectA_id'])-1]['item']
            description = negation['description'].format(object=object_single, part=negation['part'])
        else:
            description = negation['description'].format(item=negation['item'])
        skills.append(negation['name'])
        descriptions.append(description)
        categories.append('negation')

    return skills, descriptions, categories, refined_config


def load_configs_sequentially(curr_skill, curr_description, curr_category, client, k, config_path, specified_skill=None, object_prob=0.25):
    # this function can be integrated into load_configs_new later
    if specified_skill:
        assert(k==1)

    k_other = 1
    num_objects = 1
    random_config_files = [specified_skill + '.json']

    skills = []
    descriptions = []
    categories = []
    initail_config = {'objects': []}
    # generate object
    for i in range(num_objects):
        skill, description, category = load_random_value(config_path, 'object.json')
        # make sure skill is not repeated
        while skill in skills:
            skill, description, category = load_random_value(config_path, 'object.json')
        skill_pl = inflect_engine.plural(skill)
        description = description.format(skill=skill, skills=skill_pl)
        initail_config['objects'].append({'id': i+1, 'item': skill})
        skills.append(skill)
        descriptions.append(description)
        categories.append(category)

    for i in range(k_other):
        skill, description, category = curr_skill, curr_description, curr_category,
        if category == 'style':
            initail_config['style'] = skill
            description = description.format(skill=skill)
            skills.append(skill)
            descriptions.append(description)
            categories.append(category)
        elif category == 'spatial':
            if 'relation' not in initail_config:
                initail_config['relation'] = []
            initail_config['relation'].append({'name': skill, 'description': description, 'ObjectA_id': '?', 'ObjectB_id': '?'})
        else:
            idx = random.choice(range(num_objects))
            while category in initail_config['objects'][idx]:
                idx = random.choice(range(num_objects))
            initail_config['objects'][idx][category] = skill
            object_single = initail_config['objects'][idx]['item']
            object_plural = inflect_engine.plural(object_single)
            description = description.format(skill=skill, object=object_single, objects=object_plural)
            skills.append(skill)
            descriptions.append(description)
            categories.append(category)


    model = "gpt-4o-2024-05-13"
    refine_config_prompt = 'I am trying to create an image containing exactly the following things in a JSON format:\n' + json.dumps(initail_config) + '\n' + 'Could you check if there is "?" left in the JSON? If so, could you fill in the missing part? Make sure it makes sense when you fill the missing part. Do not fill in anything else unless it is indicated by "?". You may add additional objects, but only in the following two cases:\n * It is needed to fill in any "?" (Note when you fill "?", you should use existing objects first. If you still choose to add an object, explain why the existing objects cannot fulfill the need.); or \n * If there is an attribute specified in the JSON that contains relative information (e.g. "size") and there is no other object for reference. (The reason for adding an object for this case is because one cannot tell whether an object is huge without any other object in the image, but we are fine if there is no such attribute mentioned in the JSON. Note other existing object in JSON can be used for reference, and the reference object does not need to be the same object. If you still choose to add an object, explain why the existing objects cannot fulfill the need.)\nDO NOT add any object if none of the above situation is strictly satisifed, and DO NOT try to improve the image in other ways. If you choose to add an object, make sure it fits in the image naturally. Please only add the necessary objects, and the added objects should only have "id" and "item" specified, and should be appended to "objects".'
    completion = client.chat.completions.create(
        model = model,
        messages = [
            {"role": "user", "content": refine_config_prompt}
        ]
    )
    full_response = completion.choices[0].message.content
    # print(refine_config_prompt)
    print(json.dumps(initail_config))
    print(full_response)
    json_only_prompt = 'Could you respond only the JSON?'
    completion = client.chat.completions.create(
        model = model,
        messages = [
            {"role": "user", "content": refine_config_prompt},
            {"role": "assistant", "content": full_response},
            {"role": "user", "content": json_only_prompt}
        ]
    )
    json_response = completion.choices[0].message.content
    # print(json_response)
    # check if it starts with ```json and ends with ```, remove if so
    if json_response.startswith('```json'):
        json_response = json_response[7:]
    if json_response.endswith('```'):
        json_response = json_response[:-3]
    try:
        refined_config = json.loads(json_response)
    except:
        print("json_response:", json_response)
        raise ValueError("The refined config is not a valid JSON format")

    # check if id is 1, 2, 3, ...
    for i, object in enumerate(refined_config['objects']):
        assert(int(object['id']) == i + 1)
    extra_skills = []
    extra_descriptions = []
    extra_categories = []
    for object in refined_config['objects'][num_objects:]:
        extra_skills.append(object['item'])
        extra_descriptions.append(f"the image contain one {object['item']}.")
        extra_categories.append('object')
    if 'relation' in refined_config:
        for relation in refined_config['relation']:
            objectA = refined_config['objects'][int(relation['ObjectA_id'])-1]['item']
            objectB = refined_config['objects'][int(relation['ObjectB_id'])-1]['item']
            if objectA == objectB:
                objectB = "another " + objectB
            description = relation['description'].format(ObjectA=objectA, ObjectB=objectB)
            skills.append(relation['name'])
            descriptions.append(description)
            categories.append('spatial')

    return skills, descriptions, categories, refined_config

# GenEval2-style VQA (https://github.com/facebookresearch/GenEval2), built from the refined
# config rather than asked of the LLM. Each question has a fixed answer ("Yes", "No" or a number
# word); skill tags follow GenEval2 (object / count / attribute / position), plus style, verb,
# comparison, negation, differentiation and text.
SPATIAL_PHRASES = {
    'top': 'on top of',
    'bottom': 'at the bottom of',
    'above': 'above',
    'below': 'below',
    'left': 'to the left of',
    'right': 'to the right of',
    'behind': 'behind',
    'in front of': 'in front of',
    'inside': 'inside',
    'outside': 'outside of',
}
ATTRIBUTE_CATEGORIES = ['size', 'color', 'shape', 'texture']  # order of the questions per object
# fields: is (Is/Are for the subject), a/b (references), as/bs (plurals) of subject and target
COMPARISON_QUESTIONS = {
    'more': "Are there more {as_} than {bs} in the image?",
    'fewer': "Are there fewer {as_} than {bs} in the image?",
    'as many': "Are there as many {as_} as {bs} in the image?",
    'bigger': "{is_} the {a} bigger than the {b}?",
    'smaller': "{is_} the {a} smaller than the {b}?",
    'taller': "{is_} the {a} taller than the {b}?",
    'shorter': "{is_} the {a} shorter than the {b}?",
}
COUNT_COMPARISONS = {'more': lambda a, b: a > b, 'fewer': lambda a, b: a < b, 'as many': lambda a, b: a == b}

def object_counts(refined_config):
    """How many of each object the prompt shows: its "number", or one as in GenEval2 ("a pig").

    None for an object without a "number" in a count comparison ("more cats than dogs"),
    whose count the prompt leaves open."""
    compared = set()
    for comparison in refined_config.get('comparison', []):
        if comparison.get('name') in COUNT_COMPARISONS:
            for key in ('ObjectA_id', 'ObjectB_id'):
                try:
                    compared.add(int(comparison[key]) - 1)
                except (ValueError, KeyError, TypeError):
                    pass
    return [int(o['number']) if o.get('number') not in (None, '') else None if i in compared else 1
            for i, o in enumerate(refined_config['objects'])]

def atom_count(vqa_list, skills):
    """GenEval2's atom count: every question except counts answered "one"."""
    return sum(not (s == 'count' and a == 'one') for (_, a), s in zip(vqa_list, skills))

def geneval2_vqa(refined_config):
    """GenEval2-style questions for a refined config: (vqa_list, skills, atom_count).

    Per object, as in GenEval2: [count,] attributes, object; questions whose subject is the
    object (relations, actions, comparisons, missing parts) follow its block; absent objects
    and style come last. As in GenEval2, an object without a "number" is shown once and
    gets the count question "one" (see object_counts; generate_sentence writes it as "a").
    Several objects of one kind (differentiation) are named by the attribute they differ in
    ("the red cup"). Raises ValueError on configs that cannot be asked unambiguously.
    """
    objects = refined_config['objects']

    names, plurals, distinct_by = [], [], []
    for o in objects:
        twins = [p for p in objects if p['item'] == o['item']]
        category = None
        if len(twins) > 1:
            for c in ATTRIBUTE_CATEGORIES:
                values = [str(p.get(c) or '') for p in twins]
                if all(values) and len(set(values)) == len(values):
                    category = c
                    break
            else:
                raise ValueError(f"repeated object without a distinguishing attribute: {o['item']}")
        prefix = attribute_phrase(category, o[category]) + ' ' if category else ''
        names.append(prefix + o['item'])
        plurals.append(prefix + inflect_engine.plural(o['item']))
        distinct_by.append(category)

    counts = object_counts(refined_config)
    # plural only when there are several, as GenEval2 does ('the cat' / 'the cats')
    references = [plurals[i] if c and c > 1 else names[i] for i, c in enumerate(counts)]
    is_ = lambda i: 'Are' if counts[i] and counts[i] > 1 else 'Is'
    does = lambda i: 'Do' if counts[i] and counts[i] > 1 else 'Does'

    def object_index(entry, key):
        try:
            i = int(entry[key]) - 1
        except (ValueError, KeyError, TypeError):
            raise ValueError(f"unresolved {key}: {entry}")
        if not 0 <= i < len(objects):
            raise ValueError(f"unresolved {key}: {entry}")
        return i

    subject_questions = collections.defaultdict(list)  # object index -> [[question, answer, skill]]
    for category, key in RELATION_KEYS.items():
        for relation in refined_config.get(key, []):
            a, b = object_index(relation, 'ObjectA_id'), object_index(relation, 'ObjectB_id')
            if a == b:
                raise ValueError(f"relation of an object to itself: {relation}")
            name = relation['name']
            if category == 'comparison':
                if name not in COMPARISON_QUESTIONS:
                    raise ValueError(f"unknown comparison: {relation}")
                as_, bs = plurals[a], plurals[b]
                if name in COUNT_COMPARISONS:
                    if (distinct_by[a] or distinct_by[b]) and objects[a]['item'] != objects[b]['item']:
                        # prompts compare kinds ("more pancakes than trucks"), not one of several trucks;
                        # within one kind they compare its variants ("more red cups than blue cups")
                        as_, bs = inflect_engine.plural(objects[a]['item']), inflect_engine.plural(objects[b]['item'])
                    elif None not in (counts[a], counts[b]) and not COUNT_COMPARISONS[name](counts[a], counts[b]):
                        raise ValueError(f"comparison contradicts the numbers: {relation}")
                question = COMPARISON_QUESTIONS[name].format(is_=is_(a), a=references[a], b=references[b], as_=as_, bs=bs)
            else:
                phrase = SPATIAL_PHRASES.get(name, name) if category == 'spatial' else name
                question = f"{is_(a)} the {references[a]} {phrase} the {references[b]}?"
            subject_questions[a].append([question, "Yes", 'position' if category == 'spatial' else category])

    absent = []
    present_items = {o['item'] for o in objects}
    for negation in refined_config.get('negation', []):
        if 'part' in negation:
            i = object_index(negation, 'ObjectA_id')
            part = str(negation['part']).strip()
            if part in ('', '?'):
                raise ValueError(f"unresolved missing part: {negation}")
            # the LLM often writes a bare singular ("lid"): "Does the thermos have a lid?"
            if part.split()[0].lower() not in ('a', 'an', 'any', 'the') and not inflect_engine.singular_noun(part):
                part = inflect_engine.a(part)
            subject_questions[i].append([f"{does(i)} the {references[i]} have {part}?", "No", 'negation'])
        else:
            item = str(negation.get('item', '')).strip()
            if item in ('', '?') or item in present_items:
                raise ValueError(f"unresolved or present absent object: {negation}")
            absent.append([f"Are there any {inflect_engine.plural(item)} in the image?", "No", 'negation'])

    vqa = []
    for i, o in enumerate(objects):
        ref = references[i]
        if counts[i] is not None:
            vqa.append([f"How many {plurals[i]} are in the image?", inflect_engine.number_to_words(counts[i]), 'count'])
        for category in ATTRIBUTE_CATEGORIES:
            if o.get(category) and category != distinct_by[i]:
                vqa.append([f"{is_(i)} the {ref} {attribute_phrase(category, o[category])}?", "Yes", 'attribute'])
        if o.get('text') not in (None, ''):
            if str(o['text']).strip() == '?':
                raise ValueError(f"unresolved text: {o}")
            on = 'them' if counts[i] and counts[i] > 1 else 'it'
            vqa.append([f'{does(i)} the {ref} have the text "{o["text"]}" written on {on}?', "Yes", 'text'])
        # for one of several objects of a kind, its existence question also checks the attribute
        vqa.append([f"Are there any {plurals[i]} in the image?", "Yes", 'differentiation' if distinct_by[i] else 'object'])
        vqa += subject_questions[i]
    vqa += absent
    if refined_config.get('style'):
        vqa.append([f"Is the image in {refined_config['style']} style?", "Yes", 'style'])

    vqa_list = [[q, a] for q, a, _ in vqa]
    skills = [s for _, _, s in vqa]
    return vqa_list, skills, atom_count(vqa_list, skills)

def clean_sentence(text):
    """Strip quotes and labels the model sometimes wraps its prompt in."""
    text = text.strip()
    for label in ('Prompt:', 'Natural:', 'Caption:'):
        if text.startswith(label):
            text = text[len(label):].strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in '"\'':
        text = text[1:-1].strip()
    return text

def vqa_mismatches(chat_api, sentence, vqa_list):
    """Questions of vqa_list whose expected answer does not follow from the prompt alone.

    A fresh-context judge (it sees only the prompt, not the config) answers every question
    for an image that follows the prompt exactly; returns [(question, expected, got)] for
    each answer that differs, so an empty list means the VQA labels fit the prompt."""
    judge_prompt = (
        f"Here is a prompt for an image generator:\n\"{sentence}\"\n\n"
        "Imagine an image that follows this prompt exactly. Answer each question about that image "
        "based only on what the prompt says. Answer \"Yes\" or \"No\", or a number word for \"How many\" questions; "
        "answer \"Unclear\" if the prompt does not determine the answer (e.g. an attribute is never stated, "
        "or it is ambiguous which object it belongs to). Synonyms and close wording count as stated "
        "(e.g. a \"giant\" object is enormous, a \"flying saucer\" is a UFO, a \"surreal painting\" is in surrealism style).\n"
        "Respond only with one line per question, formatted as \"<number>. <answer>\".\n\n"
    )
    judge_prompt += "\n".join(f"{i}. {q}" for i, (q, _) in enumerate(vqa_list, 1))
    completion = chat_api(
        model="gpt-4o-2024-05-13",
        messages=[{"role": "user", "content": judge_prompt}],
        temperature=0.0,
        max_tokens=1024,
    )
    answers = {}
    for line in completion.choices[0].message.content.splitlines():
        m = re.match(r'\s*(\d+)[.):]\s*(.+)', line)
        if m:
            answers[int(m.group(1))] = m.group(2)
    def norm(a):
        a = re.sub(r'[^a-z0-9 ]', '', a.lower()).strip()
        return inflect_engine.number_to_words(int(a)) if a.isdigit() else a
    mismatches = []
    for i, (q, expected) in enumerate(vqa_list, 1):
        got = answers.get(i, '')
        if norm(got) != norm(expected):
            mismatches.append((q, expected, got))
    return mismatches

# Question forms prompt_vqa asks for, with their GenEval2 skill tags.
PROMPT_VQA_FORMS = [
    ('"Are there any <plural noun> in the image?"', 'Yes', 'object'),
    ('"How many <plural noun> are in the image?"', 'a number word ("one" for "a"/"an")', 'count'),
    ('"Is the <noun> <attribute>?" or "Are the <plural noun> <attribute>?"', 'Yes', 'attribute'),
    ('"Is the <noun> <spatial relation> the <other noun>?" (on top of, under, to the left of, inside, ...)', 'Yes', 'position'),
    ('"Is the <noun> <verb>ing the <other noun>?"', 'Yes', 'verb'),
    ('"Does the <noun> have the text \"<text>\" written on it?"', 'Yes', 'text'),
]
PROMPT_VQA_MAX = 8  # cap on added questions, against a model that asks about every detail

def prompt_vqa(chat_api, sentence, vqa_list, skills):
    """Adds GenEval2-style questions for important content of the written prompt that the
    config's questions (geneval2_vqa) do not cover, e.g. an object or count the writer added.

    Only salient content is asked about (objects, their counts and attributes, text, relations
    and actions between objects), not the setting, lighting or minor details.
    Returns (vqa_list, skills, atom_count) with the added questions appended."""
    forms = "\n".join(f"- {q} -> {a} (skill \"{s}\")" for q, a, s in PROMPT_VQA_FORMS)
    existing = "\n".join(f"- {q} -> {a}" for q, a in vqa_list)
    request = (
        f"Here is a prompt for an image generator:\n\"{sentence}\"\n\n"
        "Images generated from it are checked with yes/no and counting questions. These questions are already asked:\n"
        f"{existing}\n\n"
        "Add questions for the important content of the prompt that these do not cover yet, so that together they check "
        "everything the prompt clearly asks for. Important content is:\n"
        "- every object the prompt asks to be shown as a distinct thing (people, animals, items, vehicles, buildings, plants): "
        "ask both whether it is there and how many there are, if the prompt gives an exact number or says \"a\"/\"an\"/\"one\";\n"
        "- the color, material or texture, shape and size the prompt states for an object, and text written on it;\n"
        "- an action or spatial relation the prompt states between two objects.\n"
        "Do not ask about unimportant details: the setting or location (a kitchen, a beach, a forest floor), background "
        "scenery, ground, sky, weather, time of day, lighting, shadows, mood, camera or art style, generic or subjective "
        "adjectives (ripe, cozy, vintage, beautiful), parts of an object (its rim, its handle), poses, or anything only implied. "
        "When in doubt, leave it out; most prompts need no or only a few added questions.\n\n"
        f"Write each question in one of these forms, with the answer an image following the prompt exactly would give:\n{forms}\n"
        "Name objects with the words of the prompt. Do not repeat or rephrase a question that is already asked.\n"
        "Respond only with a JSON list of [question, answer, skill] triples, or [] if nothing important is missing."
    )
    completion = chat_api(
        model="gpt-4o-2024-05-13",
        messages=[{"role": "user", "content": request}],
        temperature=0.0,
        max_tokens=1024,
    )
    text = completion.choices[0].message.content
    start, end = text.find('['), text.rfind(']')
    try:
        added = json.loads(text[start:end + 1]) if start >= 0 else None
    except json.JSONDecodeError:
        added = None
    if not isinstance(added, list):
        raise ValueError(f"prompt_vqa: no JSON list in {text!r}")

    allowed = {s for _, _, s in PROMPT_VQA_FORMS}
    asked = {q.strip().lower() for q, _ in vqa_list}
    vqa_list, skills = list(vqa_list), list(skills)
    for entry in added[:PROMPT_VQA_MAX]:
        if not (isinstance(entry, list) and len(entry) == 3):
            continue
        question, answer, skill = (str(x).strip() for x in entry)
        answer = answer.lower()
        if answer.isdigit():
            answer = inflect_engine.number_to_words(int(answer))
        if skill not in allowed or not question.endswith('?') or question.lower() in asked:
            continue
        if skill == 'count':
            if not re.fullmatch(r'[a-z]+(-[a-z]+)?', answer) or answer in ('yes', 'no', 'zero', 'none'):
                continue
        elif answer != 'yes':
            continue
        else:
            answer = 'Yes'
        asked.add(question.lower())
        vqa_list.append([question, answer])
        skills.append(skill)
    return vqa_list, skills, atom_count(vqa_list, skills)

def config_has_skills(refined_config, skills, categories):
    """Whether the refined config still holds every sampled skill (the LLM may drop or alter one)."""
    objects = refined_config['objects']
    for skill, category in zip(skills, categories):
        if category == 'object':
            found = any(o['item'] == skill for o in objects)
        elif category == 'style':
            found = refined_config.get('style') == skill
        elif category in RELATION_KEYS:
            found = any(r.get('name') == skill for r in refined_config.get(RELATION_KEYS[category], []))
        elif category == 'negation':
            found = any(n.get('name') == skill for n in refined_config.get('negation', []))
        elif category == 'differentiation':
            found = any(p is not o and p['item'] == o['item'] and o.get(skill) and p.get(skill) and str(o[skill]) != str(p[skill])
                        for o in objects for p in objects)
        elif category == 'text':
            found = any(o.get('text_kind') == skill and str(o.get('text', '')).strip() not in ('', '?') for o in objects)
        else:
            found = any(str(o.get(category)) == str(skill) for o in objects)
        if not found:
            return False
    return True

# Phrasings a prompt is written in, one sampled per prompt, so the set does not share one voice.
PROMPT_REGISTERS = [
    "a short text-to-image prompt: a noun phrase or a few comma-separated fragments, not a full sentence",
    "a single natural sentence, like a photo caption",
    "one or two flowing sentences of descriptive alt text",
    "two or three flowing sentences of descriptive alt text",
]

# Bad (stiff, list-like) -> good rewrites of the same content, shown to the model as examples.
PROMPT_EXAMPLES = [
    ("A giant strawberry sits on a flat surface. Next to it is a large compass. A wrench lies beside the compass.",
     "A giant strawberry next to a large compass in a garage, with a wrench lying beside it"),
    ("A giant spaceship with a porcelain texture is in the image.",
     "A giant porcelain spaceship drifting through a starry sky."),
    ("This oil painting depicts a canyon in the background. In the foreground, positioned outside the boundaries of the canyon, there are exactly five tulips and exactly five horses. Each tulip is shaped like a pentagon. Each horse is shaped like a star and has a matte texture.",
     "An oil painting of five matte, star-shaped horses grazing among five pentagon-shaped tulips on a ridge outside a canyon at dusk."),
    ("A teacher stands in a room next to a turquoise bathtub with a glass texture. A lavender vacuum cleaner with a bumpy texture is on the floor nearby.",
     "A teacher standing beside a turquoise glass bathtub in a sunny bathroom, with a bumpy lavender vacuum cleaner nearby."),
    ("There are three dogs. The dogs are positioned on the left side of the cart. The cart has the word \"FRESH\" on it. A fox is bigger than the cart. The image contains no farmer. A clock is missing its hands. The clock is on top of the cart.",
     "Three dogs to the left of a cart with \"FRESH\" painted on it, a fox bigger than the cart, and a clock without hands on top of the cart, with no farmer around."),
]

def describe_object(o):
    """An object of a config with its attributes, as the writer should name it ("a red cup")."""
    words = [attribute_phrase(c, o[c]) for c in ATTRIBUTE_CATEGORIES if o.get(c)]
    return " ".join(words + [o['item']])

def generate_sentence(chat_api, skills, descriptions, json_config=None, generate_questions=True):
    k = len(descriptions)
    model = "gpt-4o-2024-05-13"
    register = random.choice(PROMPT_REGISTERS)
    config = json_config or {}
    objects = config.get('objects', [])
    # objects without a "number" are shown once (see object_counts), though object.json says "one or more"
    singles = [o for o, c in zip(objects, object_counts(config)) if c == 1 and not o.get('number')] if objects else []
    for o in singles:
        descriptions = [f"the image contains one {o['item']}." if d == f"the image contains one or more {inflect_engine.plural(o['item'])}." else d
                        for d in descriptions]
    prompt = "Write a prompt for a text-to-image model, the way a person would actually type it, describing a scene with the following properties:\n"
    for description in descriptions:
        prompt += f"* {description}\n"
    if json_config:
        # the entries' "description" templates ("{ObjectA} is ...") repeat the properties above in a confusing form
        view = {**config, 'objects': [{key: v for key, v in o.items() if key != 'text_kind'} for o in objects]}
        for key in list(RELATION_KEYS.values()) + ['negation']:
            if key in view:
                view[key] = [{f: v for f, v in e.items() if f != 'description'} for e in view[key]]
        prompt += "The same scene as JSON, with every object you must include and its attributes:\n```json\n" + json.dumps(view) + "\n```\n"
    prompt += (
        "Rules:\n"
        "- State every property clearly; none may be left out or only implied. Use the given words for attributes (colors, sizes, shapes, textures, styles); natural forms like \"surreal\" for surrealism or \"star-shaped\" are fine.\n"
        "- Fold attributes into the noun phrase (\"a porcelain spaceship\", \"two velvet shovels\") instead of \"with a ... texture\" or \"has a ... size\".\n"
        "- You may add a setting, background or other things if they fit the scene, but do not give the listed objects extra colors, sizes, shapes, materials or finishes (no added \"wooden\", \"glossy\", \"polished\" or \"rusty\"), and do not add more of a listed object.\n"
        "- Give counts as exact numbers (\"three horses\"), never vague amounts like \"some\", \"several\" or \"one or more\". Never write \"exactly\".\n"
        "- Write fluent, everyday English with varied structure. Do not refer to \"the image\" or \"the picture\" (\"An oil painting of ...\" is fine to state a style).\n"
    )
    relations = [config.get(key) for key in RELATION_KEYS.values()]
    if any(relations):
        prompt += ("- State every relation and action between two objects with both objects named and in the given direction (\"a cat on top of the piano\", \"a girl holding a kite\"). "
                   "Place objects relative to each other, never only relative to the picture (not \"on the left\" or \"in the background\" on its own). "
                   "A relation to several objects of a kind is to all of them (\"a napkin to the right of the three chairs\", not \"of one chair\"). "
                   "Explanations after \"meaning\" only define the relation: do not copy them (write \"a coin at the base of the lamp\", not \"touching its lowest point\").\n")
    if config.get('comparison'):
        prompt += "- Write each comparison with its own comparative word (\"more cats than dogs\", \"a horse bigger than the giraffe\"): \"taller\" is not \"bigger\", and \"towering over\" or \"a tall horse beside the giraffe\" do not state either. Do not replace a comparison with counts that are not listed.\n"
    if any(o.get('text') for o in objects):
        prompt += "- Quote text written on an object exactly, in double quotes, and say it is written on it (a mug with \"HELLO\" printed on it, a sign reading \"OPEN\"); never \"named\", \"branded\" or \"a logo\" without the quoted text.\n"
    if config.get('negation'):
        prompt += "- Say plainly what is absent or missing, with \"no\" or \"without\" (\"a desk with no chair\", \"a mug without a handle\"), not with words like \"handle-less\", \"crownless\" or \"pitted\"; do not show it anywhere else in the scene.\n"
    if singles:
        prompt += ("- Show exactly one of each of these, written with \"a\"/\"an\" or \"one\", never in the plural or with another number: "
                   + "; ".join(inflect_engine.a(describe_object(o)) for o in singles) + ".\n")
    if any(o.get('belongs_to') for o in objects):
        prompt += "- An object with \"belongs_to\" is worn, held or used by the object with that \"id\": write it that way (\"a mechanic in a mossy jacket\").\n"
    items = [o.get('item') for o in objects]
    if len(set(items)) < len(items):
        prompt += "- For several objects of the same kind, make clear which attributes belong to which (\"one red cup and one blue cup\").\n"
    prompt += (
        f"- Write it as {register}, as long as it needs to be to state every property.\n"
        "Examples of stiff prompts and natural rewrites of them (match the naturalness, not the length or structure):\n"
    )
    for bad, good in PROMPT_EXAMPLES:
        prompt += f"Stiff: {bad}\nNatural: {good}\n"
    prompt += "If the properties have obvious issues or conflicts, or are hard to realize in one image, respond \"WRONG\" and explain. Otherwise, respond only with the prompt itself, without quotes or labels."
    completion = chat_api(
        # model="gpt-4",
        model=model,
        messages=[
            {"role": "user", "content": prompt},
        ]
    )
    sentence = clean_sentence(completion.choices[0].message.content)
    if sentence.startswith("WRONG"):
        print("sentence:", sentence)
        return sentence, None
    verification = "Read your prompt again: does it make sense in a very loose sense (e.g., a person cannot be triangle shaped, but a cloud can be square-shaped and a tree can be rectangle-shaped)? Respond only \"OK\" if it does, or \"WRONG\" followed by the reason if it does not."
    completion = chat_api(
        model=model,
        messages=[
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": sentence},
            {"role": "user", "content": verification}
        ]
    )
    verdict = completion.choices[0].message.content.strip()
    if not verdict.upper().startswith("OK"):
        # anything but a clear OK (including a chatty explanation) rejects the prompt
        final_sentence = verdict if verdict.startswith("WRONG") else "WRONG (unclear verification): " + verdict
        print("final_sentence:", final_sentence)
        return final_sentence, None
    final_sentence = sentence

    if not generate_questions:
        # questions are built from the config instead (see geneval2_vqa)
        return final_sentence, []

    if json_config:
        question_generation = f"A student just draw a picture based on your description. Can you help me verify whether the student did a good job? Specifically, I want to know if the image follows your description and also follows the properties I mentioned earlier. You should ask me one yes or no question for each property, and I will tell you if they are satisfied. For example, for properties like \"the image contains one or more {skills[0]}\", the corresponding question should be \"Does the image contain {skills[0]}?\". Respond only the {k} questions, one for each property, in the same order of the properties, and each on a new line."
    else:
        question_generation = f"A student just draw a picture based on your description. Can you help me verify whether the student did a good job? Specifically, I want to know if the image follows your description and also follows the propertied I mentioned earlier. You should ask me one yes or no question for each property, and I will tell you if they are satisfied. Because the wording of your desciption might be slightly different different from the properties, please stick to the wording of your desciption when you ask the questions. For example, you should avoid using the word \"object\", and the first question should be \"Does the image contain {skills[0]}?\". Note the properties do not have information duplication, please make sure the questions do not have information duplication as well. Respond only the {k} questions, one for each property, in the same order of the properties, and each on a new line."
    completion = chat_api(
        model=model,
        messages=[
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": sentence},
            # {"role": "user", "content": verification},
            # {"role": "assistant", "content": final_sentence},
            {"role": "user", "content": question_generation}
        ]
    )
    questions = completion.choices[0].message.content
    print("=====")
    print("final_sentence:", final_sentence)
    print("=====")
    print("questions: \n" + questions)
    print("=====")

    return final_sentence, questions.split('\n')


def robustify_api_call(func, attempt=10):
    def wrapper(*args, **kwargs):
        last_error = None
        for i in range(attempt):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                print(f"Attempt {i+1} failed: {e}", flush=True)
                last_error = e
                time.sleep(2**i)
        # chained, so callers can tell an unreachable server from a bad response
        raise Exception(f"Failed after {attempt} attempts") from last_error
    return wrapper

def robusify_api_call_multiple_func(funcs, attempt=10):
    def wrapper(*args, **kwargs):
        for i in range(attempt):
            for idx, func in enumerate(funcs):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    print(f"Attempt {i+1} failed with {idx}-th client: {e}", flush=True)
            time.sleep(10 * (2**i))
        raise Exception(f"Failed after {attempt} attempts")
    return wrapper

def load_Azure_clients(model='gpt-4o-2024-05-13'):
    if model == 'gpt-4o-2024-05-13':
        clients = [
            AzureOpenAI(
                azure_endpoint=endpoint,
                api_key=api_key,
                api_version='2024-04-01-preview'
            ) for endpoint, api_key in zip([
                # add your api information here
            ],
            [
                # add your api information here
            ])
        ]
        return clients
    else:
        raise ValueError("Model not supported")

def robusify_Azure_api_call(model='gpt-4o-2024-05-13'):
    clients = load_Azure_clients(model)
    return robusify_api_call_multiple_func([client.chat.completions.create for client in clients])

# Qwen3.6 model card (https://huggingface.co/Qwen/Qwen3.6-27B): recommended sampling parameters
# and output length for Instruct (non-thinking) mode and thinking mode (general tasks); thinking
# is switched via the chat template. vLLM's reasoning parser (serve_llm.sh) returns the thinking
# separately from the content, so callers only see the answer.
QWEN36_NON_THINKING_PARAMS = {
    'temperature': 0.7,
    'top_p': 0.8,
    'top_k': 20,
    'min_p': 0.0,
    'presence_penalty': 1.5,
    'repetition_penalty': 1.0,
    'max_tokens': 32768,
    'chat_template_kwargs': {'enable_thinking': False},
}
QWEN36_THINKING_PARAMS = {
    'temperature': 1.0,
    'top_p': 0.95,
    'top_k': 20,
    'min_p': 0.0,
    'presence_penalty': 0.0,
    'repetition_penalty': 1.0,
    'max_tokens': 32768,
    'chat_template_kwargs': {'enable_thinking': True},
}
# token usage of all local LLM calls, for reporting cost (e.g. of thinking)
LLM_USAGE = collections.Counter()
LLM_USAGE_LOCK = threading.Lock()

def load_local_chat_api(base_url='http://localhost:8000/v1', model='Qwen3.6-27B', timeout=600, thinking=False):
    """Chat API backed by a local OpenAI-compatible server (e.g. `vllm serve`).

    Drop-in replacement for robusify_Azure_api_call: the returned function takes
    `model=..., messages=...` and returns an object with `.choices[0].message.content`.
    The `model` argument passed by callers (hardcoded GPT names) is ignored.
    With thinking=True, per-call temperature and max_tokens are ignored: greedy decoding
    and short budgets make the thinking loop or get cut off before the answer.
    """
    from types import SimpleNamespace
    from requests.adapters import HTTPAdapter
    url = base_url.rstrip('/') + '/chat/completions'
    # one pooled session shared across threads, so concurrent callers reuse connections
    session = requests.Session()
    session.mount('http://', HTTPAdapter(pool_connections=1, pool_maxsize=1024))
    session.mount('https://', HTTPAdapter(pool_connections=1, pool_maxsize=1024))

    served_model = model

    def chat_api(model=None, messages=None, **kwargs):
        if thinking:
            kwargs.pop('temperature', None)
            kwargs.pop('max_tokens', None)
        payload = {
            'model': served_model,
            'messages': messages,
            **(QWEN36_THINKING_PARAMS if thinking else QWEN36_NON_THINKING_PARAMS),
            **kwargs,  # per-call overrides, e.g. temperature
        }
        response = session.post(url, json=payload, timeout=timeout)
        response.raise_for_status()
        result = response.json()
        with LLM_USAGE_LOCK:
            LLM_USAGE['calls'] += 1
            LLM_USAGE.update({k: v for k, v in (result.get('usage') or {}).items() if k in ('prompt_tokens', 'completion_tokens')})
        content = result['choices'][0]['message']['content'] or ''
        # safety net in case thinking is not disabled server-side
        if '</think>' in content:
            content = content.split('</think>', 1)[1]
        content = content.strip()
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])

    return robustify_api_call(chat_api, attempt=6)

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def save_image(image, file_path):
    if image.startswith("http"):
        image_response = requests.get(image)
        with open(file_path, "wb") as file:
            file.write(image_response.content)
    else:
        image.save(file_path)




import sys
if __name__ == "__main__":
    k = int(sys.argv[1])
    client = AzureOpenAI(
        # add your api information here
    )


    while True:
        specified_skill = "color"
        skills, descriptions, categories, refined_config = load_configs_new(client, k, "./config", specified_skill)
        if specified_skill not in categories:
            import pdb; pdb.set_trace()
        sentence, questions = generate_sentence(client, skills, descriptions, json_config=refined_config)

