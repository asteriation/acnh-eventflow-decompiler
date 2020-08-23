from __future__ import annotations

from typing import Any, Dict, List, NamedTuple, Optional, Union

from datatype import AnyType, Type

HINTS: Dict[str, str] = {}

class Param(NamedTuple):
    name: str
    type: Type = AnyType

class Action:
    def __init__(self, actor_name: str, name: str, params: List[Param], conversion: Optional[str] = None) -> None:
        self.actor_name = actor_name
        self.name = name
        self.params = params
        self.conversion = conversion or f'<.name>(' + ', '.join(f'<{p.name}>' for p in params) + ')'
        self.default = conversion is None
        self.auto = False

    def format(self, params: Dict[str, Any]) -> str:
        conversion = self.conversion.replace('<.name>', f'{self.actor_name}.{self.name}')
        for p in self.params:
            try:
                assert p.name in params
                # some places do this with a string value instead of an Argument value
                if p.name == f'EntryVariableKeyInt_{params[p.name]}' or \
                    p.name == f'EntryVariableKeyBool_{params[p.name]}' or \
                    p.name == f'EntryVariableKeyFloat_{params[p.name]}':
                    value = params[p.name]
                else:
                    value = p.type.format(params[p.name])
            except:
                print(self, p, params)
                raise
            conversion = conversion.replace(f'<<{p.name}>>', str(params[p.name]))
            conversion = conversion.replace(f'<{p.name}>', value)
        return conversion

    def hint(self, params: Dict[str, Any]) -> List[str]:
        return [HINTS[p] for p in params.values() if isinstance(p, str) and p in HINTS]

    def __str__(self) -> str:
        name = self.name
        if self.default:
            auto_s = ' (auto)' if self.auto else ''
            conv = self.conversion.replace('<.name>', name)
            for p in self.params:
                conv = conv.replace(f'<<{p.name}>>', f'{p.name}: {p.type}')
                conv = conv.replace(f'<{p.name}>', f'{p.name}: {p.type}')
            return conv + auto_s
        else:
            return f'{name}: {self.conversion}'

    def export(self) -> Dict[str, Any]:
        e: Dict[str, Any] = {
            'params': {p.name: p.type for p in self.params},
        }
        if not self.default:
            e['conversion'] = self.conversion
        name = self.name
        return {
            name: e
        }

class Query:
    def __init__(self, actor_name: str, name: str, params: List[Param], rv: Type = AnyType, inverted: bool = False, conversion: Optional[Union[str, Dict[str, Any]]] = None, neg_conversion: Optional[Union[str, Dict[str, Any]]] = None) -> None:
        self.actor_name = actor_name
        self.name = name
        self.params = params
        self.rv = rv
        self.inverted = inverted
        self.conversion = conversion or f'<.name>(' + ', '.join(f'<{p.name}>' for p in params) + ')'
        self.neg_conversion = neg_conversion or ''
        assert isinstance(self.conversion, str) == isinstance(self.neg_conversion, str)
        if not self.neg_conversion:
            self.neg_conversion = f'not {self.conversion}'
        elif isinstance(self.neg_conversion, str) and isinstance(self.conversion, str):
            self.neg_conversion = self.neg_conversion.replace(f'<.conversion>', self.conversion)
        elif not isinstance(self.neg_conversion, str) and not isinstance(self.conversion, str):
            self.neg_conversion['values'] = [
                    x.replace(f'<.conversion>', y)
                    for x, y in zip(self.neg_conversion['values'], self.conversion['values'])
            ]
        self.default = conversion is None
        self.auto = False
        self.num_values = rv.num_values()

    def format(self, params: Dict[str, Any], negated: bool) -> str:
        if negated:
            conversion_used = self.neg_conversion
        else:
            conversion_used = self.conversion
        if not isinstance(conversion_used, str):
            pivot = params[conversion_used['key']]
            conversion = conversion_used['values'][pivot]
        else:
            conversion = conversion_used
        conversion = conversion.replace('<.name>', f'{self.actor_name}.{self.name}')
        for p in self.params:
            try:
                assert p.name in params
                # some places do this with a string value instead of an Argument value
                if p.name == f'EntryVariableKeyInt_{params[p.name]}' or \
                    p.name == f'EntryVariableKeyBool_{params[p.name]}' or \
                    p.name == f'EntryVariableKeyFloat_{params[p.name]}':
                    value = params[p.name]
                else:
                    value = p.type.format(params[p.name])
            except:
                print(self, p, params)
                raise
            conversion = conversion.replace(f'<<{p.name}>>', str(params[p.name]))
            conversion = conversion.replace(f'<{p.name}>', value)
        return conversion

    def hint(self, params: Dict[str, Any]) -> List[str]:
        return [HINTS[p] for p in params.values() if isinstance(p, str) and p in HINTS]

    def __str__(self) -> str:
        name = self.name
        if self.default:
            assert isinstance(self.conversion, str)
            inv_s = ' (inverted)' if self.inverted else ''
            auto_s = ' (auto)' if self.auto else ''
            conv = self.conversion.replace('<.name>', name)
            for p in self.params:
                conv = conv.replace(f'<<{p.name}>>', f'{p.name}: {p.type}')
                conv = conv.replace(f'<{p.name}>', f'{p.name}: {p.type}')
            rv_s = f' -> {self.rv}'
            return conv + rv_s + inv_s + auto_s
        else:
            inv_s = ' (inverted)' if self.inverted else ''
            return f'{name}: {self.conversion}{inv_s}'

    def export(self) -> Dict[str, Any]:
        e: Dict[str, Any] = {
            'params': {p.name: p.type for p in self.params},
        }
        if self.inverted:
            e['inverted'] = True
        if not self.default:
            e['conversion'] = self.conversion
            e['neg_conversion'] = self.neg_conversion
        e['return'] = self.rv
        name = self.name
        return {
            name: e
        }

class Actor:
    def __init__(self, name: str) -> None:
        self.name = name
        self.actions: Dict[str, Action] = {}
        self.queries: Dict[str, Query] = {}
        self.locked = False

    def register_action(self, action: Action) -> None:
        if action.name not in self.actions or True:
            self.actions[action.name] = action
            if self.locked:
                action.auto = True

    def register_query(self, query: Query) -> None:
        if query.name not in self.queries or True: # TODO
            self.queries[query.name] = query
            if self.locked:
                query.auto = True

    def lock_registration(self) -> None:
        self.locked = True

    def __str__(self):
        return f'Actor {self.name}\n' + '\n'.join([
            'actions:',
            *[f'- {a}' for a in self.actions.values()],
            'queries:',
            *[f'- {q}' for q in self.queries.values()],
        ])

    def export(self) -> Dict[str, Any]:
        e: Dict[str, Any] = {
            'actions': {},
            'queries': {},
        }
        for action in self.actions.values():
            e['actions'].update(action.export())
        for query in self.queries.values():
            e['queries'].update(query.export())
        return {
            self.name: e
        }

