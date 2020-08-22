from __future__ import annotations

from abc import ABC, abstractmethod
from collections import namedtuple
import re
from typing import Any, List, Dict, Optional, Union, Set, Tuple, Callable

import evfl

Param = namedtuple('Param', ['name', 'type'], defaults=['any'])

def _indent(level: int) -> str:
    return ' ' * (4 * level) # 4 spaces per level

def _format_type(type_: str, value: Any) -> str:
    if type_.startswith('int'):
        assert isinstance(value, int)
        if type_ != 'int':
            n = int(type_[3:])
            assert 0 <= value < n
        return repr(int(value))
    elif type_.startswith('enum'):
        assert not isinstance(value, bool) and isinstance(value, int)
        vals = type_[5:-1].split(',')
        assert 0 <= value < len(vals)
        return vals[value]
    elif type_ == 'float':
        assert not isinstance(value, bool) and isinstance(value, (int, float))
        return repr(float(value))
    elif type_ == 'str':
        assert isinstance(value, str)
        return repr(value)
    elif type_ == 'bool':
        assert isinstance(value, bool) or (isinstance(value, int) and 0 <= value <= 1)
        return 'true' if value else 'false'
    elif type_ == 'any':
        return repr(value)
    else:
        raise ValueError(f'bad type: {type_}')

def _count_type_values(type_: str) -> int:
    if type_.startswith('int') and type_ != 'int':
        return int(type_[3:])
    elif type_ == 'bool':
        return 2
    elif type_.startswith('enum'):
        return len(type_.split(','))
    else:
        return 999999999

class Action:
    def __init__(self, name: str, params: List[Param], conversion: Optional[str] = None) -> None:
        self.name = name
        self.params = params
        self.conversion = conversion or f'<.name>(' + ', '.join(f'<{p.name}>' for p in params) + ')'
        self.default = conversion is None
        self.auto = False

    def format(self, params: Dict[str, Any]) -> str:
        conversion = self.conversion.replace('<.name>', self.name)
        for p in self.params:
            assert p.name in params
            try:
                value = _format_type(p.type, params[p.name])
            except:
                print(self, p, params)
                raise
            conversion = conversion.replace(f'<{p.name}>', value)
        return conversion

    def __str__(self) -> str:
        name = self.name
        if '.' in name:
            name = name[name.index('.') + 1:]
        if self.default:
            auto_s = ' (auto)' if self.auto else ''
            conv = self.conversion.replace('<.name>', name)
            for p in self.params:
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
        if '.' in name:
            name = name[name.index('.') + 1:]
        return {
            name: e
        }

class Query:
    def __init__(self, name: str, params: List[Param], rv: str = 'any', inverted: bool = False, conversion: Optional[Union[str, Dict[str, Any]]] = None, neg_conversion: Optional[Union[str, Dict[str, Any]]] = None) -> None:
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
        self.num_values = _count_type_values(rv)

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
        conversion = conversion.replace('<.name>', self.name)
        for p in self.params:
            assert p.name in params
            try:
                value = _format_type(p.type, params[p.name])
            except:
                print(self, p, params)
                raise
            conversion = conversion.replace(f'<{p.name}>', value)
        return conversion

    def __str__(self) -> str:
        name = self.name
        if '.' in name:
            name = name[name.index('.') + 1:]
        if self.default:
            assert isinstance(self.conversion, str)
            inv_s = ' (inverted)' if self.inverted else ''
            auto_s = ' (auto)' if self.auto else ''
            conv = self.conversion.replace('<.name>', name)
            for p in self.params:
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
        if '.' in name:
            name = name[name.index('.') + 1:]
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
        if action.name not in self.actions:
            self.actions[action.name] = action
            action.name = f'{self.name}.{action.name}'
            if self.locked:
                action.auto = True

    def register_query(self, query: Query) -> None:
        if query.name not in self.queries:
            self.queries[query.name] = query
            query.name = f'{self.name}.{query.name}'
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

class Predicate(ABC):
    @abstractmethod
    def generate_code(self) -> str:
        pass

    def __invert__(self) -> Predicate:
        return NotPredicate(self)

    def __and__(self, other: Predicate) -> Predicate:
        assert isinstance(other, Predicate)
        if isinstance(other, AndPredicate):
            return AndPredicate([self]) & other
        return AndPredicate([self, other])

    def __or__(self, other: Predicate) -> Predicate:
        assert isinstance(other, Predicate)
        if isinstance(other, OrPredicate):
            return OrPredicate([self]) | other
        return OrPredicate([self, other])

class ConstPredicate(Predicate):
    def __init__(self, value: bool) -> None:
        self.value = value

    def generate_code(self) -> str:
        return _format_type('bool', self.value)

class QueryPredicate(Predicate):
    def __init__(self, query: Query, params: Dict[str, Any], values: List[Any]) -> None:
        assert len(values) > 0

        self.query = query
        self.params = params
        self.values = values
        self.negated = False

        if self.query.rv == 'bool' and self.values == [False]:
            self.negated = True

        if query.inverted:
            self.negated = not self.negated

    def generate_code(self) -> str:
        if self.query.rv == 'bool':
            if len(self.values) == 1:
                return self.query.format(self.params, self.negated)
            else:
                return 'False' if self.negated else 'True'
        else:
            if len(self.values) == 1:
                op = '!=' if self.negated else '=='
                try:
                    return f'{self.query.format(self.params, False)} {op} {_format_type(self.query.rv, self.values[0])}'
                except:
                    print(self.query, self.values)
                    raise
            else:
                op = 'not in' if self.negated else 'in'
                try:
                    vals_s = [_format_type(self.query.rv, v) for v in self.values]
                except:
                    print(self.query, self.values)
                    raise
                return f'{self.query.format(self.params, False)} {op} ({", ".join(vals_s)})'

    def __invert__(self) -> Predicate:
        qp = QueryPredicate(self.query, self.params, self.values)
        qp.negated = not self.negated
        return qp

class NotPredicate(Predicate):
    def __init__(self, inner: Predicate) -> None:
        self.inner = inner

    def generate_code(self) -> str:
        return f'not ({self.inner.generate_code()})'

    def __invert__(self) -> Predicate:
        return self.inner

class AndPredicate(Predicate):
    def __init__(self, inners: List[Predicate]) -> None:
        self.inners = inners

    def generate_code(self) -> str:
        return ' and '.join([f'({inner.generate_code()})' for inner in self.inners])

    def __invert__(self) -> Predicate:
        # if the majority of inner predicates are negated, convert to or
        num_not_predicates = sum(
            isinstance(p, NotPredicate) or (isinstance(p, QueryPredicate) and p.negated)
            for p in self.inners
        )
        if num_not_predicates * 2 >= len(self.inners):
            return OrPredicate([~p for p in self.inners])
        return Predicate.__invert__(self)

    def __and__(self, other: Predicate) -> Predicate:
        assert isinstance(other, Predicate)

        if isinstance(other, AndPredicate):
            return AndPredicate(self.inners + other.inners)
        else:
            return AndPredicate(self.inners + [other])

class OrPredicate(Predicate):
    def __init__(self, inners: List[Predicate]) -> None:
        self.inners = inners

    def generate_code(self) -> str:
        return ' or '.join([f'({inner.generate_code()})' for inner in self.inners])

    def __invert__(self) -> Predicate:
        # if the majority of inner predicates are negated, convert to or
        num_not_predicates = sum(1 for p in self.inners if isinstance(p, NotPredicate) or (isinstance(p, QueryPredicate) and p.negated))
        if num_not_predicates * 2 >= len(self.inners):
            return AndPredicate([~p for p in self.inners])
        return Predicate.__invert__(self)

    def __or__(self, other: Predicate) -> Predicate:
        assert isinstance(other, Predicate)

        if isinstance(other, OrPredicate):
            return OrPredicate(self.inners + other.inners)
        else:
            return OrPredicate(self.inners + [other])

class Node(ABC):
    def __init__(self, name: str) -> None:
        self.name = name
        self.in_edges: List[Node] = []
        self.out_edges: List[Node] = []

    def add_in_edge(self, src: Node) -> None:
        if src not in self.in_edges:
            self.in_edges.append(src)

    def add_out_edge(self, dest: Node) -> None:
        if dest not in self.out_edges:
            self.out_edges.append(dest)

    def del_in_edge(self, src: Node) -> None:
        self.in_edges.remove(src)

    def del_out_edge(self, dest: Node) -> None:
        self.out_edges.remove(dest)

    def reroute_in_edge(self, old_src: Node, new_src: Node) -> None:
        if old_src in self.in_edges:
            self.in_edges[self.in_edges.index(old_src)] = new_src

    def reroute_out_edge(self, old_dest: Node, new_dest: Node) -> None:
        if old_dest in self.out_edges:
            self.out_edges[self.out_edges.index(old_dest)] = new_dest

    def simplify(self) -> None:
        pass

    @abstractmethod
    def generate_code(self, indent_level: int = 0) -> str:
        pass

    def __str__(self) -> str:
        return f'Node[name={self.name}' + \
            f', in_edges=[{", ".join(n.name for n in self.in_edges)}]' + \
            f', out_edges=[{", ".join(n.name for n in self.out_edges)}]' + \
            ']'

class RootNode(Node):
    def __init__(self, name: str) -> None:
        Node.__init__(self, name)

    def add_in_edge(self, src: Node) -> None:
        pass

    def generate_code(self, indent_level: int = 0) -> str:
        return f'{_indent(indent_level)}flow {self.name}:\n' + '\n'.join(n.generate_code(indent_level + 1) for n in self.out_edges)

    def __str__(self) -> str:
        return f'RootNode[name={self.name}' + \
            f', out_edges=[{", ".join(n.name for n in self.out_edges)}]' + \
            ']'

class ActionNode(Node):
    def __init__(self, name: str, action: Action, params: Dict[str, Any], nxt: Optional[str]) -> None:
        Node.__init__(self, name)
        self.action = action
        self.params = params
        self.nxt = nxt

    def generate_code(self, indent_level: int = 0) -> str:
        return f'{_indent(indent_level)}{self.action.format(self.params)}\n' + '\n'.join(n.generate_code(indent_level) for n in self.out_edges)

    def __str__(self) -> str:
        return f'ActionNode[name={self.name}' + \
            f', action={self.action}' + \
            f', params={self.params}' + \
            f', in_edges=[{", ".join(n.name for n in self.in_edges)}]' + \
            f', out_edges=[{", ".join(n.name for n in self.out_edges)}]' + \
            ']'

class SwitchNode(Node):
    def __init__(self, name: str, query: Query, params: Dict[str, Any]) -> None:
        Node.__init__(self, name)
        self.query = query
        self.params = params
        self.cases: Dict[str, List[Any]] = {}
        self.terminal_node: Optional[Node] = None

        assert sum(len(x) for x in self.cases.values()) <= _count_type_values(self.query.rv)

    def del_out_edge(self, dest: Node) -> None:
        Node.del_out_edge(self, dest)
        if dest.name in self.cases:
            del self.cases[dest.name]
        if self.terminal_node is dest:
            self.terminal_node = None

    def reroute_out_edge(self, old_dest: Node, new_dest: Node) -> None:
        Node.reroute_out_edge(self, old_dest, new_dest)
        if old_dest.name in self.cases:
            c = self.cases[old_dest.name]
            del self.cases[old_dest.name]
            self.cases[new_dest.name] = c
        if self.terminal_node is old_dest:
            self.terminal_node = new_dest

    def add_case(self, node_name: str, value: Any) -> None:
        if node_name not in self.cases:
            self.cases[node_name] = []
        self.cases[node_name].append(value)

    def register_terminal_node(self, terminal_node: Node) -> None:
        # todo: improve when switch node doesn't need a terminal node contact
        if sum(len(x) for x in self.cases.values()) == _count_type_values(self.query.rv):
            self.terminal_node = None
            return

        self.add_out_edge(terminal_node)
        terminal_node.add_in_edge(self)

        self.terminal_node = terminal_node

    def generate_code(self, indent_level: int = 0) -> str:
        if len(self.cases) == 0:
            if self.terminal_node:
                return self.terminal_node.generate_code(indent_level)
            else:
                return ''

        if self.query.rv == 'bool':
            if len(self.cases) == 1:
                values = [*self.cases.values()][0]
                if len(values) == 2:
                    # true/false branch are identical, no branch needed
                    return self.out_edges[0].generate_code(indent_level)
                else:
                    # if (not) X, else return -> invert unless goto
                    assert self.terminal_node is not None

                    if isinstance(self.terminal_node, GotoNode):
                        return f'{_indent(indent_level)}if {self.query.format(self.params, not values[0])}:\n' + \
                                self.out_edges[0].generate_code(indent_level + 1)
                    else:
                        __invert__s = '' if not values[0] else 'not '
                        return f'{_indent(indent_level)}if k{self.query.format(self.params, values[0])}):\n' + \
                                self.terminal_node.generate_code(indent_level + 1) + \
                                self.out_edges[0].generate_code(indent_level)
            else:
                # T/F explicitly spelled out
                true_node: Node
                false_node: Node

                if self.cases[self.out_edges[0].name][0]:
                    true_node, false_node = self.out_edges
                else:
                    false_node, true_node = self.out_edges
                return f'{_indent(indent_level)}if {self.query.format(self.params, False)}:\n' + \
                        true_node.generate_code(indent_level + 1) + \
                        f'{_indent(indent_level)}else:\n' + \
                        false_node.generate_code(indent_level + 1)
        elif len(self.cases) == 1:
            # if [query] in (...), if [query] = X ... else return -> negate and return unless goto
            values = [*self.cases.values()][0]
            if len(values) == self.query.num_values:
                # all branches identical, no branch needd
                return self.out_edges[0].generate_code(indent_level)

            assert self.terminal_node is not None

            try:
                if isinstance(self.terminal_node, GotoNode):
                    op = f'== {_format_type(self.query.rv, values[0])}' if len(values) == 1 else 'in (' + ', '.join(_format_type(self.query.rv, v) for v in values) + ')'

                    return f'{_indent(indent_level)}if {self.query.format(self.params, False)} {op}:\n' + \
                            self.out_edges[0].generate_code(indent_level + 1)
                else:
                    __invert__op = f'!= {_format_type(self.query.rv, values[0])}' if len(values) == 1 else 'not in (' + ', '.join(_format_type(self.query.rv, v) for v in values) + ')'

                    return f'{_indent(indent_level)}if {self.query.format(self.params, False)} {__invert__op}:\n' + \
                            self.terminal_node.generate_code(indent_level + 1) + \
                            self.out_edges[0].generate_code(indent_level)
            except:
                print(self.query, values)
                raise
        else:
            # generic case:
            # f{name} = [query]
            # if/elif checks on f{name}, else implicit
            vname = f'f{self.name}'

            cases: List[str] = []
            for event, values in sorted(self.cases.items(), key=lambda x: min(x[1])):
                try:
                    else_s = 'el' if cases else ''
                    op_s = f'== {_format_type(self.query.rv, values[0])}' if len(values) == 1 else 'in (' + ', '.join(_format_type(self.query.rv, v)  for v in sorted(values)) + ')'
                    cases.append(
                            f'{_indent(indent_level)}{else_s}if {vname} {op_s}:\n' +
                            [e for e in self.out_edges if e.name == event][0].generate_code(indent_level + 1)
                    )
                except:
                    print(self.query, self.cases.values())
                    raise

            return f'{_indent(indent_level)}{vname} = {self.query.format(self.params, False)}\n' + ''.join(cases)

    def __str__(self) -> str:
        return f'SwitchNode[name={self.name}' + \
            f', query={self.query}' + \
            f', params={self.params}' + \
            f', cases={self.cases}' + \
            f', in_edges=[{", ".join(n.name for n in self.in_edges)}]' + \
            f', out_edges=[{", ".join(n.name for n in self.out_edges)}]' + \
            ']'

class ForkNode(Node):
    def __init__(self, name: str, join_node: JoinNode, forks: List[str]) -> None:
        Node.__init__(self, name)
        self.join_node = join_node
        self.forks = forks

    def generate_code(self, indent_level: int = 0) -> str:
        s = f'{_indent(indent_level)}fork:\n'
        self.join_node.disable()
        for i, e in enumerate(self.out_edges):
            s += f'{_indent(indent_level + 1)}branch{i}:\n{e.generate_code(indent_level + 2)}'
        self.join_node.enable()
        return s + self.join_node.generate_code(indent_level)

    def __str__(self) -> str:
        return f'ForkNode[name={self.name}' + \
            f', join_node={self.join_node.name}' + \
            f', in_edges=[{", ".join(n.name for n in self.in_edges)}]' + \
            f', out_edges=[{", ".join(n.name for n in self.out_edges)}]' + \
            ']'

class JoinNode(Node):
    def __init__(self, name: str, nxt: Optional[str]) -> None:
        Node.__init__(self, name)
        self.nxt = nxt
        self.disabled = False

    def disable(self) -> None:
        self.disabled = True

    def enable(self) -> None:
        self.disabled = False

    def generate_code(self, indent_level: int = 0) -> str:
        if not self.disabled:
            return "\n".join(e.generate_code(indent_level) for e in self.out_edges)
        else:
            return ''

    def __str__(self) -> str:
        return f'JoinNode[name={self.name}' + \
            f', in_edges=[{", ".join(n.name for n in self.in_edges)}]' + \
            f', out_edges=[{", ".join(n.name for n in self.out_edges)}]' + \
            ']'

class SubflowNode(Node):
    def __init__(self, name: str, ns: str, called_root_name: str, nxt: Optional[str] = None) -> None:
        Node.__init__(self, name)
        self.ns = ns
        self.called_root_name = called_root_name
        self.nxt = nxt

    def generate_code(self, indent_level: int = 0) -> str:
        ns = f'{self.ns}::' if self.ns else ''
        return f'{_indent(indent_level)}run {ns}{self.called_root_name}\n' + '\n'.join(e.generate_code(indent_level) for e in self.out_edges)

    def __str__(self) -> str:
        return f'SubflowNode[name={self.name}' + \
            f', ns={self.ns}' + \
            f', called_root_name={self.called_root_name}' + \
            f', in_edges=[{", ".join(n.name for n in self.in_edges)}]' + \
            f', out_edges=[{", ".join(n.name for n in self.out_edges)}]' + \
            ']'

class TerminalNode(Node):
    def __init__(self, name: str) -> None:
        Node.__init__(self, name)

    def add_out_edge(self, src: Node) -> None:
        pass

    def generate_code(self, indent_level: int = 0) -> str:
        return f'{_indent(indent_level)}return\n'

    def __str__(self) -> str:
        return f'TerminalNode[name={self.name}' + \
            f', in_edges=[{", ".join(n.name for n in self.in_edges)}]' + \
            ']'

class GotoNode(TerminalNode):
    def __init__(self, name: str) -> None:
        TerminalNode.__init__(self, name)

    def generate_code(self, indent_level: int = 0) -> str:
        return f'{_indent(indent_level)}goto {self.name}\n'

    def __str__(self) -> str:
        return f'GotoNode[name={self.name}' + \
            f', in_edges=[{", ".join(n.name for n in self.in_edges)}]' + \
            f', out_edges=[{", ".join(n.name for n in self.out_edges)}]' + \
            ']'

class NoopNode(Node):
    def __init__(self, name: str) -> None:
        Node.__init__(self, name)

    def generate_code(self, indent_level: int = 0) -> str:
        return f'{_indent(indent_level)}pass\n'

    def __str__(self) -> str:
        return 'Noop' + Node.__str__(self)

class EntryPointNode(Node):
    def __init__(self, name: str, entry_label: Optional[str] = None) -> None:
        Node.__init__(self, name)
        self.entry_label = self.name if entry_label is None else entry_label

    def generate_code(self, indent_level: int = 0) -> str:
        return f'{_indent(0)}entrypoint {self.entry_label}:\n' + '\n'.join(e.generate_code(indent_level) for e in self.out_edges)

    def __str__(self) -> str:
        return f'EntryPointNode[name={self.name}' + \
            f', entry_label={self.entry_label}' + \
            f', in_edges=[{", ".join(n.name for n in self.in_edges)}]' + \
            f', out_edges=[{", ".join(n.name for n in self.out_edges)}]' + \
            ']'

class GroupNode(Node):
    def __init__(self, root: Node) -> None:
        Node.__init__(self, f'grp!{root.name}')
        self.root = root
        self.goto_node = GotoNode(f'grp-end!{root.name}')
        self.nodes = self.__enumerate_group()

    def __enumerate_group(self) -> List[Node]:
        visited = {self.root}
        s = [self.root]
        while s:
            n = s.pop()
            for c in n.out_edges:
                if c not in visited:
                    visited.add(c)
                    s.append(c)
        return list(visited)

    def recalculate_group(self) -> None:
        self.nodes = self.__enumerate_group()

    def simplify(self) -> None:
        for node in self.nodes:
            node.simplify()

    def generate_code(self, indent_level: int = 0) -> str:
        return self.root.generate_code(indent_level) + \
                (f'{_indent(indent_level - 1)}{self.goto_node.name}::\n' if self.goto_node.in_edges else '') + \
                '\n'.join(e.generate_code(indent_level) for e in self.out_edges)

    def __str__(self) -> str:
        return f'GroupNode[name={self.name}' + \
            f', root={self.root}' + \
            f', in_edges=[{", ".join(n.name for n in self.in_edges)}]' + \
            f', out_edges=[{", ".join(n.name for n in self.out_edges)}]' + \
            ']'

class IfElseNode(Node):
    Rule = namedtuple('Rule', ['predicate', 'node'])

    def __init__(self, name: str, rules: List[IfElseNode.Rule], default: Node) -> None:
        Node.__init__(self, name)
        self.rules = rules
        self.default = default

    def simplify(self) -> None:
        # prefer non-negated penultimate branch if possible
        penul_p = self.rules[-1].predicate
        if isinstance(penul_p, NotPredicate) or (isinstance(penul_p, QueryPredicate) and penul_p.negated):
            self.rules[-1], self.default = IfElseNode.Rule(~self.rules[-1].predicate, self.default), self.rules[-1].node

        # prefer else branch for noops if easily swappable (last if branch)
        if isinstance(self.rules[-1].node, NoopNode) and not isinstance(self.default, NoopNode):
            self.rules[-1], self.default = IfElseNode.Rule(~self.rules[-1].predicate, self.default), self.rules[-1].node

    def generate_code(self, indent_level: int = 0) -> str:
        code = ''
        for predicate, node in self.rules:
            el_s = 'el' if code else ''
            code += _indent(indent_level) + f'{el_s}if {predicate.generate_code()}:\n' + \
                    node.generate_code(indent_level + 1)
        if not isinstance(self.default, NoopNode):
            code += f'{_indent(indent_level)}else:\n' + self.default.generate_code(indent_level + 1)
        return code

    def del_out_edge(self, dest: Node) -> None:
        self.reroute_out_edge(dest, NoopNode(f'pass!{self.name}')) # todo: this noop node is not in CFG.nodes

    def reroute_out_edge(self, old_dest: Node, new_dest: Node) -> None:
        Node.reroute_out_edge(self, old_dest, new_dest)

        for i in range(len(self.rules)):
            if self.rules[i].node is old_dest:
                self.rules[i] = IfElseNode.Rule(self.rules[i].predicate, new_dest)
        if self.default is old_dest:
            self.default = new_dest

    def __str__(self) -> str:
        return f'IfElseNode[name={self.name}' + \
            f', rules={self.rules}' + \
            f', default={self.default}' + \
            f', in_edges=[{", ".join(n.name for n in self.in_edges)}]' + \
            f', out_edges=[{", ".join(n.name for n in self.out_edges)}]' + \
            ']'

class CFG:
    def __init__(self, name: str) -> None:
        self.name = name
        self.roots: List[RootNode] = []
        self.actors: Dict[str, Actor] = {}
        self.nodes: Dict[str, Node] = {}

    def __assign_components(self) -> List[List[RootNode]]:
        visited: Set[RootNode] = set()
        components: List[List[RootNode]] = []
        for root in self.roots:
            if root in visited:
                continue
            component = self.__get_connected_roots(root)
            components.append(component)
            visited.update(set(component))
        return components

    def __get_connected_roots(self, root) -> List[RootNode]:
        visited: Set[Node] = set()
        component: List[RootNode] = []
        visited.add(root)
        s: List[Node] = [root]
        while s:
            node = s.pop()
            if isinstance(node, RootNode):
                component.append(node)
            for n in node.in_edges + node.out_edges:
                if n not in visited:
                    visited.add(n)
                    s.append(n)
        return component

    def __is_cut(self, nodes: List[Node]) -> bool:
        ns = set(nodes)
        reachable: Set[Node] = set()
        for n in ns:
            reachable.update(self.__find_postorder(n))
        for n in reachable:
            if reachable.intersection(n.in_edges) and n not in ns:
                return False
        return True

    def __separate_overlapping_flows(self) -> None:
        components = self.__assign_components()
        for component in components:
            nreachable = [len(self.__find_postorder(r)) for r in component]
            main_entry, main_entry_i = component[0], nreachable[0]
            for r, i in zip(component, nreachable):
                if i > main_entry_i:
                    main_entry, main_entry_i = r, i

            change = True
            while change:
                change = False
                for r in component:
                    if r is main_entry:
                        continue
                    if self.__is_cut(r.out_edges):
                        change = True
                        new_root = self.__detach_root(r)
                        new_component = self.__get_connected_roots(new_root)
                        component = self.__get_connected_roots(main_entry)
                        components.append(new_component)
                        break
                    else:
                        self.__convert_root_to_entrypoints(r)

    def __detach_nodes_with_noop(self, src: Node, dest: Node) -> None:
        new_noop_node = NoopNode(f'noop!{src.name}-{dest.name}')
        self.nodes[new_noop_node.name] = new_noop_node

        src.reroute_out_edge(dest, new_noop_node)
        dest.del_in_edge(src)
        new_noop_node.add_in_edge(src)

    def __detach_nodes_with_call(self, src: Node, dest: Node, entry_point: str) -> None:
        new_call_node = SubflowNode(f'ext!{src.name}-{dest.name}', '', entry_point)
        self.nodes[new_call_node.name] = new_call_node

        src.reroute_out_edge(dest, new_call_node)
        dest.del_in_edge(src)
        new_call_node.add_in_edge(src)

    def __detach_root(self, root: RootNode) -> RootNode:
        entry_point = root.out_edges[0]
        return self.__detach_node_as_sub(entry_point)

    def __detach_node_as_sub(self, entry_point: Node) -> RootNode:
        new_root = RootNode(f'sub_{entry_point.name}')
        new_root.add_out_edge(entry_point)

        for caller in entry_point.in_edges[:]:
            self.__detach_nodes_with_call(caller, entry_point, new_root.name)

        entry_point.in_edges = [new_root]
        self.nodes[new_root.name] = new_root
        self.roots.append(new_root)

        return new_root

    def __convert_root_to_entrypoints(self, root: RootNode) -> None:
        # determine the "exclusive" part of the graph owned by root (i.e. no in edges)
        # for all destinations of out edges of exclusive subgraph:
        #   inject a label, funnel connections through label
        #   disconnect subgraph from label via call
        excl, connections = self.__get_exclusive_subgraph(root)
        labels = set([self.__convert_node_to_entrypoint(n, n.name) for n in connections])
        for node in excl:
            leaving_nodes: Set[EntryPointNode] = set(node.out_edges).intersection(labels) # type: ignore
            for label in leaving_nodes:
                self.__detach_nodes_with_call(node, label, label.entry_label)

    def __get_exclusive_subgraph(self, root: RootNode) -> Tuple[Set[Node], Set[Node]]:
        reachable = set(self.__find_postorder(root))
        other_roots = [r for r in self.__get_connected_roots(root) if r is not root]
        other_reachable: Set[Node] = set()
        for other in other_roots:
            other_reachable.update(set(self.__find_postorder(other)))

        exclusive = reachable - other_reachable
        connections: Set[Node] = set()
        for node in exclusive:
            connections.update(set(node.out_edges) - exclusive)

        return exclusive, connections

    def __convert_node_to_entrypoint(self, node: Node, name: str) -> EntryPointNode:
        if isinstance(node, EntryPointNode):
            return node

        entry_point_node = EntryPointNode(f'{node.name}-entrypoint', name)
        entry_point_node.add_out_edge(node)
        entry_point_node.in_edges = node.in_edges
        node.in_edges = [entry_point_node]

        for caller in entry_point_node.in_edges:
            caller.reroute_out_edge(node, entry_point_node)

        self.nodes[entry_point_node.name] = entry_point_node
        return entry_point_node

    def __add_terminal_nodes(self) -> None:
        for root in self.roots:
            tn = TerminalNode(f'tm!{root.name}')
            for node in self.__find_postorder(root):
                if len(node.out_edges) == 0 and not isinstance(node, TerminalNode):
                    node.add_out_edge(tn)
                    tn.add_in_edge(node)
                elif isinstance(node, SwitchNode):
                    node.register_terminal_node(tn)

            if len(tn.in_edges) > 0:
                self.nodes[tn.name] = tn

    def __find_dominator_tree(self, root: Node) -> Dict[Node, Node]:
        # https://www.cs.rice.edu/~keith/Embed/dom.pdf
        dom = {root: root}

        rpo = self.__find_reverse_postorder(root)
        rpo_index = {k: i for i, k in enumerate(rpo)}
        changed = True
        while changed:
            changed = False
            for b in rpo[1:]: # exclude root
                new_idom: Optional[Node] = None
                for p in b.in_edges:
                    if p in dom:
                        if new_idom is None:
                            new_idom = p
                        # lca of p, new_idom
                        while p is not new_idom:
                            while rpo_index[p] > rpo_index[new_idom]:
                                p = dom[p]
                            while rpo_index[p] < rpo_index[new_idom]:
                                new_idom = dom[new_idom]

                assert new_idom is not None

                if b not in dom or dom[b] is not new_idom:
                    dom[b] = new_idom
                    changed = True
        return dom

    def __find_postorder_helper(self, root: Node, pred: Callable[[Node], bool], visited: Set[str]) -> List[Node]:
        po: List[Node] = []
        if not pred(root):
            return po
        for node in root.out_edges:
            if node.name not in visited:
                visited.add(node.name)
                po.extend(self.__find_postorder_helper(node, pred, visited))
        po.append(root)
        return po

    def __find_postorder(self, root: Node, pred: Callable[[Node], bool] = lambda n: True) -> List[Node]:
        return self.__find_postorder_helper(root, pred, set())

    def __find_reverse_postorder(self, root: Node, pred: Callable[[Node], bool] = lambda n: True) -> List[Node]:
        return self.__find_postorder(root, pred)[::-1]

    def __path_exists(self, src: Node, dest: Node) -> bool:
        # todo: don't be this dumb
        return dest in self.__find_postorder(src)

    def __split_loops_helper(self, node: Node, active: Set[Node], visited: Set[Node], added_entrypoints: Dict[Node, Node]) -> None:
        active.add(node)

        for nxt in node.out_edges[:]:
            if nxt not in visited and nxt not in active:
                self.__split_loops_helper(nxt, active, visited, added_entrypoints)
            elif nxt in active and nxt not in visited:
                entrypoint = self.__convert_node_to_entrypoint(nxt, nxt.name)
                self.__detach_nodes_with_call(node, entrypoint, entrypoint.entry_label)

                active.add(entrypoint)
                added_entrypoints[nxt] = entrypoint

        active.remove(node)
        visited.add(node)
        if node in added_entrypoints:
            active.remove(added_entrypoints[node])
            visited.add(added_entrypoints[node])

    def __split_loops(self) -> None:
        for root in self.roots:
            self.__split_loops_helper(root, set(), set(), {})

    def __convert_switch_to_if(self) -> None:
        for root in self.roots:
            for node in self.__find_postorder(root):
                if not isinstance(node, SwitchNode):
                    continue
                if len(node.out_edges) != 2:
                    continue

                inner: Node
                default: Node

                if len(node.cases) == 1:
                    assert node.terminal_node is not None

                    inner = node.out_edges[0]
                    default = node.terminal_node
                else:
                    inner = node.out_edges[0]
                    default = node.out_edges[1]

                ifelse_node = IfElseNode(node.name, [
                    IfElseNode.Rule(QueryPredicate(node.query, node.params, node.cases[inner.name]), inner)
                ], default)

                ifelse_node.in_edges = node.in_edges
                for caller in node.in_edges:
                    caller.reroute_out_edge(node, ifelse_node)

                for exit_node in node.out_edges:
                    ifelse_node.add_out_edge(exit_node)
                    exit_node.reroute_in_edge(node, ifelse_node)

                del self.nodes[node.name]
                self.nodes[ifelse_node.name] = ifelse_node

    def __collapse_andor(self) -> None:
        for root in self.roots:
            change = True
            while change:
                change = False
                for node in self.__find_postorder(root):
                    if not isinstance(node, IfElseNode):
                        continue
                    if len(node.rules) != 1:
                        continue
                    ifelse_children = [c for c in node.out_edges if isinstance(c, IfElseNode)][::-1]
                    for child in ifelse_children:
                        if len(child.rules) != 1:
                            continue
                        if len(child.in_edges) != 1:
                            continue
                        other_child, = [c for c in node.out_edges if c is not child]
                        if other_child not in child.out_edges:
                            continue

                        true_branch, = [c for c in child.out_edges if c is not other_child]
                        false_branch = other_child

                        # get predicate for node -> child branch
                        node_predicate = node.rules[0].predicate
                        if node.default is child:
                            node_predicate = ~node_predicate
                        # get predicate for child -> true_branch
                        child_predicate = child.rules[0].predicate
                        if child.default is true_branch:
                            child_predicate = ~child_predicate

                        # replace node + child with [predicate, true_branch], default=false_branch
                        predicate = node_predicate & child_predicate
                        new_node = IfElseNode(node.name, [
                            IfElseNode.Rule(predicate, true_branch)
                        ], false_branch)

                        self.__merge_coupled_nodes(node, child, new_node)

                        change = True
                        break

    def __collapse_if(self) -> None:
        for root in self.roots:
            for node in self.__find_postorder(root):
                if not isinstance(node, IfElseNode) or len(node.rules) != 1:
                    continue

                child_ok = isinstance(node.rules[0].node, IfElseNode) and len(node.rules[0].node.in_edges) == 1
                default_ok = isinstance(node.default, IfElseNode) and len(node.default.in_edges) == 1
                if default_ok and not child_ok:
                    predicate = node.rules[0].predicate
                    value_branch = node.rules[0].node
                    else_branch = node.default
                elif child_ok and not default_ok:
                    predicate = ~node.rules[0].predicate
                    value_branch = node.default
                    else_branch = node.rules[0].node
                elif child_ok and default_ok:
                    child_score = len(self.__find_postorder(node.rules[0].node))
                    default_score = len(self.__find_postorder(node.default))
                    if child_score <= default_score:
                        predicate = node.rules[0].predicate
                        value_branch = node.rules[0].node
                        else_branch = node.default
                    else:
                        predicate = ~node.rules[0].predicate
                        value_branch = node.default
                        else_branch = node.rules[0].node
                else:
                    continue

                assert isinstance(else_branch, IfElseNode)

                # try not to join with ifelse block if it would add a noop branch
                if self.__path_exists(else_branch, value_branch):
                    node.rules[0] = IfElseNode.Rule(~predicate, else_branch)
                    node.default = value_branch
                    continue

                # don't join with ifelse block if endpoint would change
                dom = self.__find_dominator_tree(root)
                cur_end = self.__find_block_end(node, dom)
                else_end = self.__find_block_end(else_branch, dom)
                if cur_end != else_end:
                    continue

                ifelse_node = IfElseNode(node.name, [
                    IfElseNode.Rule(predicate, value_branch),
                    *else_branch.rules
                ], else_branch.default)

                self.__merge_coupled_nodes(node, else_branch, ifelse_node)

    def __merge_coupled_nodes(self, parent_node: Node, child_node: Node, new_node: Node) -> None:
        assert child_node.in_edges == [parent_node]

        new_node.in_edges = parent_node.in_edges
        for caller in parent_node.in_edges:
            caller.reroute_out_edge(parent_node, new_node)

        for parent_out in parent_node.out_edges:
            if parent_out is child_node:
                continue
            new_node.add_out_edge(parent_out)
            parent_out.reroute_in_edge(parent_node, new_node)

        for exit_node in child_node.out_edges:
            new_node.add_out_edge(exit_node)
            exit_node.reroute_in_edge(child_node, new_node)

        del self.nodes[parent_node.name]
        del self.nodes[child_node.name]
        self.nodes[new_node.name] = new_node

    def __collapse_cases(self) -> None:
        for root in self.roots:
            for node in self.__find_postorder(root):
                # if not isinstance(node, SwitchNode):
                    # continue
                if len(set(node.out_edges)) < 2:
                    continue

                node = self.__try_collapse_block(root, node)

    def __find_block_end(self, node: Node, dom: Dict[Node, Node]) -> Optional[Node]:
        if not all(dom[child] is node or isinstance(child, TerminalNode) for child in node.out_edges):
            return None

        end: Optional[Node] = None
        for k in self.__find_postorder(node)[:-1]:
            if k not in dom:
                continue
            if isinstance(k, TerminalNode):
                inner_dom = self.__find_dominator_tree(node)
                if inner_dom[k] is not node:
                    continue
            elif dom[k] is not node:
                continue

            if not all(self.__path_exists(child, k) for child in node.out_edges):
                continue

            end = k
            break

        return end

    def __try_collapse_block(self, entry: Node, root: Node) -> Node:
        dom = self.__find_dominator_tree(entry)
        end = self.__find_block_end(root, dom)

        # print(entry.name, root.name, end)

        if end is None: # no end found, do not collapse
            return root

        if root.out_edges == [end]: # nothing to collapse
            return root

        # detach end from root if necessary
        if end in root.out_edges:
            self.__detach_nodes_with_noop(root, end)

        # print('Collapsing', root.name, 'to', end.name)

        sw = GroupNode(root)
        for in_node in root.in_edges:
            sw.add_in_edge(in_node)
            in_node.reroute_out_edge(root, sw)
        root.in_edges = []

        inner_terminal: Node = sw.goto_node

        # avoid goto -> return
        if isinstance(end, TerminalNode):
            inner_terminal = NoopNode(f'grp-end!{root.name}')

        s: List[Node] = [root]
        deleted = []
        while s:
            n = s.pop()
            if n is end or isinstance(n, TerminalNode):
                continue

            if end in n.out_edges:
                n.del_out_edge(end)

            if n in end.in_edges:
                end.del_in_edge(n)

            if n.name in self.nodes:
                del self.nodes[n.name]
                deleted.append(n)

            if isinstance(n, SwitchNode):
                n.register_terminal_node(inner_terminal)
                if n.terminal_node is not None:
                    self.nodes[inner_terminal.name] = inner_terminal

            s.extend(nd for nd in n.out_edges if nd in dom and dom[nd] in deleted)

        sw.add_out_edge(end)
        end.add_in_edge(sw)

        self.nodes[sw.name] = sw

        return sw

    def __extract_reused_blocks(self, nodes: List[Node] = None) -> None:
        if nodes is None:
            nodes = list(self.nodes.values())

        # print('call', nodes)
        # if a node has two different node parents and is the root of a DAG, and is not a single line,
        #   extract to subflow
        for node in nodes:
            if len(set(node.in_edges)) >= 2 and self.__is_cut([node]) and \
                    (node.out_edges or not isinstance(node, (ActionNode, TerminalNode, GotoNode, NoopNode, EntryPointNode, SubflowNode))):
                # print('detaching', node.name, [n.name for n in node.in_edges])
                self.__detach_node_as_sub(node)
            elif isinstance(node, GroupNode):
                l = node.nodes[:]
                l.remove(node.root)
                self.__extract_reused_blocks(l)
                node.recalculate_group()

    def __simplify_all(self) -> None:
        for node in self.nodes.values():
            node.simplify()

    def generate_code(self) -> str:
        self.roots.sort(key=lambda x: x.name)
        code = '\n'.join(root.generate_code() for root in self.roots).split('\n')

        # strip ununsed labels
        goto_regex = re.compile('^\s*goto (\S+)')
        label_regex = re.compile('^\s*(\S+)::')

        used: Set[str] = set()
        for line in code:
            m = goto_regex.match(line)
            if m:
                used.add(m.group(1))

        stripped_code = []
        for line in code:
            m = label_regex.match(line)
            if (not m or m.group(1) in used):
                stripped_code.append(line)

        return '\n'.join(stripped_code)

    def export_actors(self) -> Dict[str, Any]:
        e: Dict[str, Any] = {}
        for actor in self.actors.values():
            e.update(actor.export())
        return e

    @staticmethod
    def read(data: bytes, actor_data: Dict[str, Any]) -> CFG:
        flow = evfl.EventFlow()
        flow.read(data)

        flowchart = flow.flowchart

        cfg = CFG(flowchart.name)
        cfg.actors = {r.identifier.name: Actor(r.identifier.name) for r in flowchart.actors}
        for actor_name, d in actor_data.items():
            if actor_name not in cfg.actors:
                cfg.actors[actor_name] = Actor(actor_name)
                # continue
            for action, info in d['actions'].items():
                cfg.actors[actor_name].register_action(Action(
                    action,
                    [Param(name, type_) for name, type_ in info['params'].items()],
                    info.get('conversion', None),
                ))
            for query, info in d['queries'].items():
                cfg.actors[actor_name].register_query(Query(
                    query,
                    [Param(name, type_) for name, type_ in info['params'].items()],
                    info.get('return', 'any'),
                    info.get('inverted', False),
                    info.get('conversion', None),
                    info.get('neg_conversion', None),
                ))

        for a in cfg.actors.values():
            a.lock_registration()

        cfg.nodes = {}
        node: Node

        for event in flowchart.events:
            name = event.name
            event = event.data

            if isinstance(event, evfl.event.ActionEvent):
                actor = event.actor.v.identifier.name
                action = event.actor_action.v.v[15:]
                params = event.params.data if event.params else {}
                nxt = event.nxt.v.name if event.nxt.v else None

                cfg.actors[actor].register_action(Action(action, [Param(p) for p in params.keys()]))
                act = cfg.actors[actor].actions[action]

                node = ActionNode(name, act, params, nxt)
            elif isinstance(event, evfl.event.SwitchEvent):
                actor = event.actor.v.identifier.name
                query = event.actor_query.v.v[14:]
                params = event.params.data if event.params else {}
                cases = event.cases

                cfg.actors[actor].register_query(Query(query, [Param(p) for p in params.keys()]))
                q = cfg.actors[actor].queries[query]

                node = SwitchNode(name, q, params)
                for value, ev in cases.items():
                    node.add_case(ev.v.name, value)
            elif isinstance(event, evfl.event.ForkEvent):
                join = event.join.v.name
                forks = [f.v.name for f in event.forks] if event.forks else []

                join_node = JoinNode(join, None) if join not in cfg.nodes else cfg.nodes[join]
                assert isinstance(join_node, JoinNode)
                cfg.nodes[join] = join_node
                node = ForkNode(name, join_node, forks)
                node.add_out_edge(join_node)
                join_node.add_in_edge(node)
            elif isinstance(event, evfl.JoinEvent):
                if name not in cfg.nodes:
                    node = JoinNode(name, event.nxt.v.name if event.nxt.v else None)
                else:
                    node = cfg.nodes[name]
                    assert isinstance(node, JoinNode)
                    node.nxt = event.nxt.v.name if event.nxt.v else None
            elif isinstance(event, evfl.event.SubFlowEvent):
                ns = event.res_flowchart_name
                called = event.entry_point_name
                nxt = event.nxt.v.name if event.nxt.v else None

                node = SubflowNode(name, ns, called, nxt)
            else:
                raise ValueError(f'unknown event: {event}')

            cfg.nodes[name] = node

        for name, node in cfg.nodes.items():
            if isinstance(node, ActionNode) or isinstance(node, SubflowNode) or isinstance(node, JoinNode):
                if node.nxt is not None:
                    node.add_out_edge(cfg.nodes[node.nxt])
                    cfg.nodes[node.nxt].add_in_edge(node)
            elif isinstance(node, SwitchNode):
                for out_name in node.cases.keys():
                    node.add_out_edge(cfg.nodes[out_name])
                    cfg.nodes[out_name].add_in_edge(node)
            elif isinstance(node, ForkNode):
                for out_name in node.forks:
                    node.add_out_edge(cfg.nodes[out_name])
                    cfg.nodes[out_name].add_in_edge(node)
            else:
                raise RuntimeError('bad node type')

        for entry_point in flowchart.entry_points:
            node = RootNode(entry_point.name)
            if entry_point.main_event.v is not None:
                main_event = entry_point.main_event.v.name

                node.add_out_edge(cfg.nodes[main_event])
                cfg.nodes[main_event].add_in_edge(node)

            cfg.nodes[entry_point.name] = node
            cfg.roots.append(node)

        cfg.__separate_overlapping_flows()
        cfg.__add_terminal_nodes()

        old_nodes = None
        while old_nodes != set(cfg.nodes.values()):
            old_nodes = set(cfg.nodes.values())

            cfg.__split_loops()
            cfg.__convert_switch_to_if()
            cfg.__collapse_andor()
            cfg.__collapse_if()
            cfg.__collapse_cases()
            cfg.__extract_reused_blocks()

            cfg.__simplify_all()

        return cfg

