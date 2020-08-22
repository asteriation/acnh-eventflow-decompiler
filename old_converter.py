from collections import namedtuple
import evfl
import json
import sys

FILE = sys.argv[1]
INDENT = 4

id_ctr = 0
def new_id():
    global id_ctr
    id_ctr += 1
    return f'v{id_ctr - 1}'

class Actor:
    Param = namedtuple('Param', ['name', 'type'], defaults=['any'])

    class Action:
        def __init__(self, name, params, conversion=None):
            self.name = name
            self.params = params
            self.conversion = conversion or f'<.name>(' + ', '.join(f'<{p.name}>' for p in params) + ')'
            self.used = False
            self.default = conversion is None

        def format(self, params):
            conversion = self.conversion.replace('<.name>', self.name)
            for p in self.params:
                if p.name not in params:
                    value = repr(None)
                else:
                    value = repr(params[p.name])
                conversion = conversion.replace(f'<{p.name}>', value)
            self.used = True
            return conversion

        def __str__(self):
            return f'{self.name}: {self.conversion.replace("<.name>", self.name)}'

        def prototype(self):
            return f'action {self.name}(' + ', '.join(
                f'{p.name}: {p.type}' for p in self.params
            ) + ')'

    class Query:
        def __init__(self, name, params, rv='any', conversion=None):
            self.name = name
            self.params = params
            self.rv = rv
            self.conversion = conversion or f'<.name>(' + ', '.join(f'<{p.name}>' for p in params) + ')'
            self.used = False
            self.default = conversion is None

        def format(self, params):
            conversion = self.conversion.replace('<.name>', self.name)
            for p in self.params:
                if p.name not in params:
                    value = repr(None)
                else:
                    value = repr(params[p.name])
                conversion = conversion.replace(f'<{p.name}>', value)
            self.used = True
            return conversion

        def __str__(self):
            return f'{self.name}: {self.conversion.replace("<.name>", self.name)}'

        def prototype(self):
            return f'query {self.name}(' + ', '.join(
                f'{p.name}: {p.type}' for p in self.params
            ) + f') -> {self.rv}'

    def __init__(self, name):
        self.name = name
        self.actions = {}
        self.queries = {}

    def register_action(self, action):
        if action.name not in self.actions:
            self.actions[action.name] = action
            action.name = f'{self.name}.{action.name}'

    def register_query(self, query):
        if query.name not in self.queries:
            self.queries[query.name] = query
            query.name = f'{self.name}.{query.name}'

    def __str__(self):
        return f'Actor {self.name}\n' + '\n'.join([
            'actions:',
            *[f'- {a}' for a in self.actions.values()],
            'queries:',
            *[f'- {q}' for q in self.queries.values()],
        ])

    def format(self):
        return '\n'.join([
            f'actor {self.name}:',
            *indent_strings([
                a.prototype()
                for a in self.actions.values()
                if a.default and a.used
            ], 1),
            *indent_strings([
                q.prototype()
                for q in self.queries.values()
                if q.default and q.used
            ], 1)
        ])


def indent_strings(strs, ind_lvl):
    return [' ' * (INDENT * ind_lvl) + x for x in strs]

def process_entry(entry, actors):
    output = [f'flow {entry.name}:']
    body = process_event(entry.main_event.v, actors, 1)
    return '\n'.join(output + body)

def process_event(event, actors, ind_lvl):
    if event == None:
        return []
    if type(event.data) is evfl.event.ActionEvent:
        return process_action_event(event.data, actors, ind_lvl)
    elif type(event.data) is evfl.event.SwitchEvent:
        return process_switch_event(event.data, actors, ind_lvl)
    elif type(event.data) is evfl.event.SubFlowEvent:
        return process_subflow_event(event.data, actors, ind_lvl)
    else:
        print('unknown event: ', event.data)
        return indent_strings(['[unknown]'], ind_lvl)

def process_action_event(event, actors, ind_lvl):
    actor = event.actor.v.identifier.name
    action = event.actor_action.v.v[15:]
    params = event.params.data if event.params else {}
    nxt = event.nxt.v

    actors[actor].register_action(Actor.Action(action, [Actor.Param(p) for p in params.keys()]))

    return indent_strings([
        actors[actor].actions[action].format(params),
        *process_event(nxt, actors, 0),
    ], ind_lvl)

def process_switch_event(event, actors, ind_lvl):
    actor = event.actor.v.identifier.name
    query = event.actor_query.v.v[14:]
    params = event.params.data if event.params else {}
    cases = event.cases

    actors[actor].register_query(Actor.Query(query, [Actor.Param(p) for p in params.keys()]))

    # group by following event
    groups = {}
    events = {}
    for k, v in cases.items():
        e = v.v.name
        if e not in groups:
            groups[e] = []
        groups[e].append(k)
        events[e] = v.v

    q = actors[actor].queries[query]
    
    if q.rv == 'bool':
        true_event, false_event = None, None
        for k, v in groups.items():
            if v[0] == 0:
                false_event = k
            else:
                true_event = k
        if true_event and not false_event:
            out = [
                f'if not {q.format(params)}:',
                *indent_strings(['return'], 1),
                *process_event(events[true_event], actors, 0),
            ]
        elif false_event and not true_event:
            out = [
                f'if {q.format(params)}:',
                *indent_strings(['return'], 1),
                *process_event(events[true_event], actors, 0),
            ]
        else:
            out = [
                f'if {q.format(params)}:',
                *process_event(events[true_event], actors, 1),
                'else:',
                *process_event(events[false_event], actors, 1),
            ]
    elif len(groups) == 1:
        ev = events[[*groups.keys()][0]]
        values = [repr(x) for x in [*groups.values()][0]]
        if len(values) == 1:
            op = f'!= {values[0]}'
        else:
            op = f'not in ({", ".join(values)})'
        out = [
            f'if {q.format(params)} {op}:',
            *indent_strings(['return'], 1),
            *process_event(ev, actors, 0)
        ]
    else:
        var = new_id()
        out = [
            f'{var} = {q.format(params)}',
        ]
        for k, v in groups.items():
            ev = events[k]
            values = [repr(x) for x in v]
            if len(values) == 1:
                op = f'== {values[0]}'
            else:
                op = f'in ({", ".join(values)})'
            out.extend([
                ('else ' if len(out) > 1 else '') + f'if {var} {op}:',
                *process_event(ev, actors, 1),
            ])
    return indent_strings(out, ind_lvl)

def process_subflow_event(event, actors, ind_lvl):
    called = event.entry_point_name
    nxt = event.nxt.v

    return indent_strings([
        f'run {called}',
        *process_event(nxt, actors, 0),
    ], ind_lvl)

def process(filename):
    flow = evfl.EventFlow()
    with open(filename, 'rb') as f:
        flow.read(f.read())

    flowchart = flow.flowchart

    name = flowchart.name
    entry_points = flowchart.entry_points

    actors = {r.identifier.name: Actor(r.identifier.name) for r in flowchart.actors}

    with open('actors.json', 'rt') as f:
        actor_data = json.load(f)

    for actor_name, data in actor_data.items():
        if actor_name not in actors:
            continue
        for action, info in data['actions'].items():
            actors[actor_name].register_action(Actor.Action(
                action,
                [Actor.Param(p['name'], p['type']) for p in info['params']],
                info['conversion'] if 'conversion' in info else None
            ))
        for query, info in data['queries'].items():
            actors[actor_name].register_query(Actor.Query(
                query,
                [Actor.Param(p['name'], p['type']) for p in info['params']],
                info['return'] if 'return' in info else 'any',
                info['conversion'] if 'conversion' in info else None,
            ))

    print(f'{filename} - {name}')

    flows = ''
    for ep in entry_points:
        flows += process_entry(ep, actors) + '\n\n'

    for v in actors.values():
        print(v.format() + '\n')

    print(flows)

process(FILE)


