from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, NamedTuple, Optional, Union, Tuple

from datatype import Type
from predicates import Predicate, NotPredicate, QueryPredicate
from actors import Action, Query

class Node(ABC):
    def __init__(self, name: str) -> None:
        self.name = name
        self.in_edges: List[Node] = []
        self.out_edges: List[Node] = []
        self.group_node: Optional[GroupNode] = None

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

    def remap_subflow(self, old: Tuple[str, str], new: Tuple[str, str]) -> None:
        for node in self.out_edges:
            node.remap_subflow(old, new)

    def simplify(self) -> None:
        pass

    def __str__(self) -> str:
        return f'Node[name={self.name}' + \
            ', in_edges=[{", ".join(n.name for n in self.in_edges)}]' + \
            f', out_edges=[{", ".join(n.name for n in self.out_edges)}]' + \
            ']'

    def get_dot(self) -> str:
        return f'n{id(self)} [label=<<b>{self.name}</b><br/>{type(self).__name__}>];' + \
                ''.join(f'n{id(self)} -> n{id(nx)}' + (f'[lhead=cluster{id(nx)}]' if isinstance(nx, GroupNode) else '') + ';' for nx in self.out_edges)

class RootNode(Node):
    @dataclass
    class VarDef:
        name: str
        type_: Type
        initial_value: Union[int, bool, float]

        def __str__(self) -> str:
            return f'{self.name}: {self.type_} = {self.initial_value}'

        def __hash__(self) -> int:
            return hash(self.name) ^ hash(self.type_.type) ^ hash(self.initial_value)

    def __init__(self, name: str, vardefs: List[VarDef] = []) -> None:
        Node.__init__(self, name)
        self.vardefs = vardefs[:]

    def add_in_edge(self, src: Node) -> None:
        pass

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

    def remap_subflow(self, old: Tuple[str, str], new: Tuple[str, str]) -> None:
        if (self.ns, self.called_root_name) == old:
            self.ns, self.called_root_name = new
        super().remap_subflow(old, new)

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

    def __str__(self) -> str:
        return f'TerminalNode[name={self.name}' + \
            f', in_edges=[{", ".join(n.name for n in self.in_edges)}]' + \
            ']'

class DeadendTerminalNode(TerminalNode):
    def __init__(self, name: str) -> None:
        TerminalNode.__init__(self, name)

    def __str__(self) -> str:
        return f'DeadendTerminalNode[name={self.name}' + \
            f', in_edges=[{", ".join(n.name for n in self.in_edges)}]' + \
            ']'

class NoopNode(Node):
    def __init__(self, name: str) -> None:
        Node.__init__(self, name)

    def __str__(self) -> str:
        return 'Noop' + Node.__str__(self)

class EntryPointNode(Node):
    def __init__(self, name: str, entry_label: Optional[str] = None) -> None:
        Node.__init__(self, name)
        self.entry_label = self.name if entry_label is None else entry_label

    def __str__(self) -> str:
        return f'EntryPointNode[name={self.name}' + \
            f', entry_label={self.entry_label}' + \
            f', in_edges=[{", ".join(n.name for n in self.in_edges)}]' + \
            f', out_edges=[{", ".join(n.name for n in self.out_edges)}]' + \
            ']'

class GroupNode(Node):
    def __init__(self, root: Node) -> None:
        Node.__init__(self, f'grp_{root.name}')
        self.root = root
        self.pass_node = NoopNode(f'grp-end_{root.name}')
        self.nodes = self.__enumerate_group()

        self.root.group_node = self

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

    def remap_subflow(self, old: Tuple[str, str], new: Tuple[str, str]) -> None:
        for node in self.nodes:
            node.remap_subflow(old, new)
        super().remap_subflow(old, new)

    def __str__(self) -> str:
        return f'GroupNode[name={self.name}' + \
            f', root={self.root}' + \
            f', in_edges=[{", ".join(n.name for n in self.in_edges)}]' + \
            f', out_edges=[{", ".join(n.name for n in self.out_edges)}]' + \
            ']'

    def get_dot(self) -> str:
        return f'subgraph cluster{id(self)} {"{"} label=<<b>{self.name}</b><br/>{type(self).__name__}>;' + \
                f'n{id(self)}[shape="none"][style="invis"][label=""];' + \
                ''.join(n.get_dot() for n in self.nodes) + \
                '}' + ''.join(f'n{id(self)} -> n{id(nx)}[ltail=cluster{id(self)}' + (f',lhead=cluster{id(nx)}' if isinstance(nx, GroupNode) else '') + '];' for nx in self.out_edges)

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

    def del_out_edge(self, dest: Node) -> None:
        self.reroute_out_edge(dest, NoopNode(f'pass_{self.name}')) # todo: this noop node is not in CFG.nodes

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

    def del_out_edge(self, dest: Node) -> None:
        self.reroute_out_edge(dest, NoopNode(f'pass_{self.name}')) # todo: this noop node is not in CFG.nodes

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

    def del_out_edge(self, dest: Node) -> None:
        self.reroute_out_edge(dest, NoopNode(f'pass_{self.name}')) # todo: this noop node is not in CFG.nodes

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

