from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, NamedTuple, Optional, Union

from datatype import AnyType, BoolType, Type
from indent import indent
from predicates import Predicate, NotPredicate, QueryPredicate
from actors import Param, Action, Query, Actor

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
        while old_src in self.in_edges:
            self.in_edges[self.in_edges.index(old_src)] = new_src

    def reroute_out_edge(self, old_dest: Node, new_dest: Node) -> None:
        while old_dest in self.out_edges:
            self.out_edges[self.out_edges.index(old_dest)] = new_dest

    def simplify(self) -> None:
        pass

    @abstractmethod
    def generate_code(self, indent_level: int = 0, generate_pass: bool = False) -> str:
        pass

    def __str__(self) -> str:
        return f'Node[name={self.name}' + \
            f', in_edges=[{", ".join(n.name for n in self.in_edges)}]' + \
            f', out_edges=[{", ".join(n.name for n in self.out_edges)}]' + \
            ']'

class RootNode(Node):
    @dataclass
    class VarDef:
        name: str
        type_: Type
        initial_value: Union[int, bool, float]

        def __str__(self) -> str:
            return f'{self.name}: {self.type_} = {self.initial_value}'

    def __init__(self, name: str, vardefs: List[VarDef] = []) -> None:
        Node.__init__(self, name)
        self.vardefs = vardefs[:]

    def add_in_edge(self, src: Node) -> None:
        pass

    def generate_code(self, indent_level: int = 0, generate_pass: bool = False) -> str:
        return f'{indent(indent_level)}flow {self.name}({", ".join(str(v) for v in self.vardefs)}):\n' + \
                '\n'.join(n.generate_code(indent_level + 1) for n in self.out_edges)

    def __str__(self) -> str:
        return f'RootNode[name={self.name}' + \
            f', vardefs=[{", ".join(str(v) for v in self.vardefs)}]' + \
            f', out_edges=[{", ".join(n.name for n in self.out_edges)}]' + \
            ']'

class ActionNode(Node):
    def __init__(self, name: str, action: Action, params: Dict[str, Any], nxt: Optional[str]) -> None:
        Node.__init__(self, name)
        self.action = action
        self.params = params
        self.nxt = nxt

    def generate_code(self, indent_level: int = 0, generate_pass: bool = False) -> str:
        hint_s = ''.join(f'{indent(indent_level)}# {hint}\n' for hint in self.action.hint(self.params))
        return hint_s + f'{indent(indent_level)}{self.action.format(self.params)}\n' + '\n'.join(n.generate_code(indent_level) for n in self.out_edges)

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

        assert sum(len(x) for x in self.cases.values()) <= self.query.rv.num_values()

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
        if sum(len(x) for x in self.cases.values()) == self.query.rv.num_values():
            self.terminal_node = None
            return

        self.add_out_edge(terminal_node)
        terminal_node.add_in_edge(self)

        self.terminal_node = terminal_node

    def generate_code(self, indent_level: int = 0, generate_pass: bool = False) -> str:
        if len(self.cases) == 0:
            return f'{indent(indent_level)}if {self.query.format(self.params, False)}:\n' + \
                    NoopNode('').generate_code(indent_level + 1, True)

        hint_s = ''.join(f'{indent(indent_level)}# {hint}\n' for hint in self.query.hint(self.params))
        if self.query.rv == BoolType:
            if len(self.cases) == 1:
                values = [*self.cases.values()][0]
                if len(values) == 2:
                    # true/false branch are identical, no branch needed
                    return self.out_edges[0].generate_code(indent_level)
                else:
                    # if (not) X, else return -> invert unless goto
                    assert self.terminal_node is not None

                    if isinstance(self.terminal_node, NoopNode):
                        return hint_s + f'{indent(indent_level)}if {self.query.format(self.params, not values[0])}:\n' + \
                                self.out_edges[0].generate_code(indent_level + 1, True)
                    else:
                        __invert__s = '' if not values[0] else 'not '
                        return hint_s + f'{indent(indent_level)}if k{self.query.format(self.params, values[0])}):\n' + \
                                self.terminal_node.generate_code(indent_level + 1, True) + \
                                self.out_edges[0].generate_code(indent_level)
            else:
                # T/F explicitly spelled out
                true_node: Node
                false_node: Node

                if self.cases[self.out_edges[0].name][0]:
                    true_node, false_node = self.out_edges
                else:
                    false_node, true_node = self.out_edges
                return hint_s + f'{indent(indent_level)}if {self.query.format(self.params, False)}:\n' + \
                        true_node.generate_code(indent_level + 1, True) + \
                        f'{indent(indent_level)}else:\n' + \
                        false_node.generate_code(indent_level + 1, True)
        elif len(self.cases) == 1:
            # if [query] in (...), if [query] = X ... else return -> negate and return unless goto
            values = [*self.cases.values()][0]
            if len(values) == self.query.num_values:
                # all branches identical, no branch needd
                return self.out_edges[0].generate_code(indent_level)

            assert self.terminal_node is not None

            try:
                if isinstance(self.terminal_node, NoopNode):
                    op = f'== {self.query.rv.format(values[0])}' if len(values) == 1 else 'in (' + ', '.join(self.query.rv.format(v) for v in values) + ')'

                    return hint_s + f'{indent(indent_level)}if {self.query.format(self.params, False)} {op}:\n' + \
                            self.out_edges[0].generate_code(indent_level + 1, True)
                else:
                    __invert__op = f'!= {self.query.rv.format(values[0])}' if len(values) == 1 else 'not in (' + ', '.join(self.query.rv.format(v) for v in values) + ')'

                    return hint_s + f'{indent(indent_level)}if {self.query.format(self.params, False)} {__invert__op}:\n' + \
                            self.terminal_node.generate_code(indent_level + 1, True) + \
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
                    op_s = f'== {self.query.rv.format(values[0])}' if len(values) == 1 else 'in (' + ', '.join(self.query.rv.format(v)  for v in sorted(values)) + ')'
                    cases.append(
                            f'{indent(indent_level)}{else_s}if {vname} {op_s}:\n' +
                            [e for e in self.out_edges if e.name == event][0].generate_code(indent_level + 1, True)
                    )
                except:
                    print(self.query, self.cases.values())
                    raise

            return hint_s + f'{indent(indent_level)}{vname} = {self.query.format(self.params, False)}\n' + ''.join(cases)

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

    def generate_code(self, indent_level: int = 0, generate_pass: bool = False) -> str:
        s = f'{indent(indent_level)}fork:\n'
        self.join_node.disable()
        for i, e in enumerate(self.out_edges):
            s += f'{indent(indent_level + 1)}branch{i}:\n{e.generate_code(indent_level + 2, True)}'
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

    def generate_code(self, indent_level: int = 0, generate_pass: bool = False) -> str:
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
    def __init__(self, name: str, ns: str, called_root_name: str, nxt: Optional[str] = None, params: Optional[Dict[str, Any]] = None) -> None:
        Node.__init__(self, name)
        self.ns = ns
        self.called_root_name = called_root_name
        self.nxt = nxt
        self.params = params.copy() if params else {}

    def generate_code(self, indent_level: int = 0, generate_pass: bool = False) -> str:
        ns = f'{self.ns}::' if self.ns else ''
        # todo: type-checked subflow params
        # todo: hints
        param_s = ', '.join(f'{name}={AnyType.format(value)}' for name, value in self.params.items())
        return f'{indent(indent_level)}run {ns}{self.called_root_name}({param_s})\n' + '\n'.join(e.generate_code(indent_level) for e in self.out_edges)

    def __str__(self) -> str:
        return f'SubflowNode[name={self.name}' + \
            f', ns={self.ns}' + \
            f', called_root_name={self.called_root_name}' + \
            f', params={self.params}' + \
            f', in_edges=[{", ".join(n.name for n in self.in_edges)}]' + \
            f', out_edges=[{", ".join(n.name for n in self.out_edges)}]' + \
            ']'

class TerminalNode(Node):
    def __init__(self, name: str) -> None:
        Node.__init__(self, name)

    def add_out_edge(self, src: Node) -> None:
        pass

    def generate_code(self, indent_level: int = 0, generate_pass: bool = False) -> str:
        return f'{indent(indent_level)}return\n'

    def __str__(self) -> str:
        return f'TerminalNode[name={self.name}' + \
            f', in_edges=[{", ".join(n.name for n in self.in_edges)}]' + \
            ']'

class DeadendTerminalNode(TerminalNode):
    def __init__(self, name: str) -> None:
        TerminalNode.__init__(self, name)

    def generate_code(self, indent_level: int = 0, generate_pass: bool = False) -> str:
        return ''

    def __str__(self) -> str:
        return f'DeadendTerminalNode[name={self.name}' + \
            f', in_edges=[{", ".join(n.name for n in self.in_edges)}]' + \
            ']'

class NoopNode(Node):
    def __init__(self, name: str) -> None:
        Node.__init__(self, name)

    def generate_code(self, indent_level: int = 0, generate_pass: bool = False) -> str:
        if generate_pass:
            return f'{indent(indent_level)}pass\n'
        else:
            return ''

    def __str__(self) -> str:
        return 'Noop' + Node.__str__(self)

class EntryPointNode(Node):
    def __init__(self, name: str, entry_label: Optional[str] = None) -> None:
        Node.__init__(self, name)
        self.entry_label = self.name if entry_label is None else entry_label

    def generate_code(self, indent_level: int = 0, generate_pass: bool = False) -> str:
        return f'{indent(0)}entrypoint {self.entry_label}:\n' + '\n'.join(e.generate_code(indent_level) for e in self.out_edges)

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
        self.pass_node = NoopNode(f'grp-end!{root.name}')
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

    def generate_code(self, indent_level: int = 0, generate_pass: bool = False) -> str:
        return self.root.generate_code(indent_level) + \
                '\n'.join(e.generate_code(indent_level) for e in self.out_edges)

    def __str__(self) -> str:
        return f'GroupNode[name={self.name}' + \
            f', root={self.root}' + \
            f', in_edges=[{", ".join(n.name for n in self.in_edges)}]' + \
            f', out_edges=[{", ".join(n.name for n in self.out_edges)}]' + \
            ']'

class IfElseNode(Node):
    class Rule(NamedTuple):
        predicate: Predicate
        node: Node

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

    def generate_code(self, indent_level: int = 0, generate_pass: bool = False) -> str:
        code = ''
        for predicate, node in self.rules:
            hint_s = ''.join(f'{indent(indent_level)}# {hint}\n' for hint in predicate.hint())
            el_s = 'el' if code else ''
            code += hint_s
            code += indent(indent_level) + f'{el_s}if {predicate.generate_code()}:\n' + \
                    node.generate_code(indent_level + 1, True)
        if not isinstance(self.default, NoopNode):
            code += f'{indent(indent_level)}else:\n' + self.default.generate_code(indent_level + 1, True)
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

class WhileNode(Node):
    def __init__(self, name: str, loop_cond: Predicate, loop_body: Node, loop_exit: Node) -> None:
        Node.__init__(self, name)
        self.loop_cond = loop_cond
        self.loop_body = loop_body # should be detached from root (self)
        self.loop_exit = loop_exit

    def generate_code(self, indent_level: int = 0, generate_pass: bool = False) -> str:
        hint_s = ''.join(f'{indent(indent_level)}# {hint}\n' for hint in self.loop_cond.hint())
        code = hint_s
        code += f'{indent(indent_level)}while {self.loop_cond.generate_code()}:\n'
        code += self.loop_body.generate_code(indent_level + 1, True)
        code += self.loop_exit.generate_code(indent_level)
        return code

    def del_out_edge(self, dest: Node) -> None:
        self.reroute_out_edge(dest, NoopNode(f'pass!{self.name}')) # todo: this noop node is not in CFG.nodes

    def reroute_out_edge(self, old_dest: Node, new_dest: Node) -> None:
        if old_dest == self.loop_body:
            self.loop_body = new_dest
        else:
            self.loop_exit = new_dest
        Node.reroute_out_edge(self, old_dest, new_dest)

    def __str__(self) -> str:
        return f'WhileNode[name={self.name}' + \
            f', loop_cond={self.loop_cond}' + \
            f', loop_body={self.loop_body}' + \
            f', loop_exit={self.loop_exit}' + \
            f', in_edges=[{", ".join(n.name for n in self.in_edges)}]' + \
            f', out_edges=[{", ".join(n.name for n in self.out_edges)}]' + \
            ']'

class DoWhileNode(Node):
    def __init__(self, name: str, loop_cond: Predicate, loop_body: Node, loop_exit: Node) -> None:
        Node.__init__(self, name)
        self.loop_cond = loop_cond
        self.loop_body = loop_body
        self.loop_exit = loop_exit

    def generate_code(self, indent_level: int = 0, generate_pass: bool = False) -> str:
        hint_s = ''.join(f'{indent(indent_level)}# {hint}\n' for hint in self.loop_cond.hint())
        code = ''
        code += f'{indent(indent_level)}do:\n'
        code += self.loop_body.generate_code(indent_level + 1, True)
        code += hint_s
        code += f'{indent(indent_level)}while {self.loop_cond.generate_code()}\n'
        code += self.loop_exit.generate_code(indent_level)
        return code

    def del_out_edge(self, dest: Node) -> None:
        self.reroute_out_edge(dest, NoopNode(f'pass!{self.name}')) # todo: this noop node is not in CFG.nodes

    def reroute_out_edge(self, old_dest: Node, new_dest: Node) -> None:
        if old_dest == self.loop_body:
            self.loop_body = new_dest
        else:
            self.loop_exit = new_dest
        Node.reroute_out_edge(self, old_dest, new_dest)

    def __str__(self) -> str:
        return f'DoWhileNode[name={self.name}' + \
            f', loop_cond={self.loop_cond}' + \
            f', loop_body={self.loop_body}' + \
            f', loop_exit={self.loop_exit}' + \
            f', in_edges=[{", ".join(n.name for n in self.in_edges)}]' + \
            f', out_edges=[{", ".join(n.name for n in self.out_edges)}]' + \
            ']'

