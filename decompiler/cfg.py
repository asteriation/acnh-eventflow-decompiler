from __future__ import annotations

import re
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union, Sequence

from .datatype import AnyType, ActorType, Type, Argument, infer_type
from .predicates import Predicate, ConstPredicate, QueryPredicate
from .actors import Param, Action, Query, Actor
from .nodes import Node, RootNode, ActionNode, SwitchNode, SubflowNode, TerminalNode, DeadendTerminalNode, NoopNode, EntryPointNode, GroupNode, IfElseNode, WhileNode, DoWhileNode
from .codegen import CodeGenerator

class CFG:
    def __init__(self, name: str) -> None:
        self.name = name
        self.roots: List[RootNode] = []
        self.actors: Dict[Tuple[str, str, str], Actor] = {}
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
            if set(n.in_edges) - reachable and n not in ns and not isinstance(n, TerminalNode):
                return False
        return True

    def __update_vardefs(self, root: RootNode, name: str, old_type: Type, new_type: Type) -> Optional[Type]:
        if old_type == new_type or old_type == AnyType:
            return None
        if new_type.type == '_placeholder' and old_type.type != '_placeholder':
            return None
        if old_type.type != '_placeholder':
            new_type = AnyType
        for vardef in root.vardefs:
            if vardef.name == name:
                vardef.type_ = new_type
                return new_type
        root.vardefs.append(RootNode.VarDef(name, new_type, initial_value=None))
        return new_type

    def __add_implicit_vardefs(self) -> None:
        known_roots_types = {r.name: {v.name: v.type_ for v in r.vardefs} for r in self.roots}
        initial_values = {r.name: {v.name: v.initial_value for v in r.vardefs} for r in self.roots}
        placeholder_type = Type('_placeholder')
        changed = True
        while changed:
            changed = False
            for root in self.roots:
                vardefs = known_roots_types[root.name]
                i = 0
                q: List[Node] = [root]
                while i < len(q):
                    for node in self.__find_reverse_postorder(q[i]):
                        if isinstance(node, SubflowNode):
                            params = node.params
                            for name, value in params.items():
                                if isinstance(value, Argument):
                                    signature = {} if node.ns else known_roots_types.get(node.called_root_name, {})
                                    type_ = signature.get(name, placeholder_type)
                                    result = self.__update_vardefs(root, value, vardefs.get(value, placeholder_type), type_)
                                    if result is not None or value not in vardefs:
                                        vardefs[value] = result if result is not None else placeholder_type
                                        changed = True
                            if node.ns == '':
                                expected = known_roots_types[node.called_root_name] = known_roots_types.get(node.called_root_name, {})
                                for name, value in params.items():
                                    type_ = placeholder_type
                                    if isinstance(value, Argument):
                                        if value in vardefs:
                                            type_ = vardefs[value]
                                    else:
                                        type_ = infer_type(value)

                                    other_root = self.nodes[node.called_root_name]
                                    assert isinstance(other_root, RootNode)
                                    result = self.__update_vardefs(other_root, name, expected.get(name, placeholder_type), type_)
                                    if result is not None or name not in expected:
                                        expected[name] = result if result is not None else placeholder_type
                                        changed = True
                                expected = {**expected}
                                for name in list(expected.keys()):
                                    if name in initial_values[node.called_root_name]:
                                        del expected[name]
                                for name, type_ in expected.items():
                                    if name not in params:
                                        params[name] = Argument(name)
                                        changed = True
                        elif isinstance(node, GroupNode):
                            q.append(node.root)
                        else:
                            calls: Sequence[Tuple[Union[Action, Query], Dict[str, Any]]] = []
                            if isinstance(node, ActionNode):
                                calls = [(node.action, node.params)]
                            elif isinstance(node, SwitchNode):
                                calls = [(node.query, node.params)]
                            elif isinstance(node, (WhileNode, DoWhileNode)):
                                calls = node.loop_cond.get_queries()
                            elif isinstance(node, IfElseNode):
                                calls = sum((r.predicate.get_queries() for r in node.rules), [])
                            for function, params in calls:
                                if function.actor_name[2]:
                                    value = function.actor_name[2]
                                    type_ = ActorType
                                    result = self.__update_vardefs(root, value, vardefs.get(value, placeholder_type), type_)
                                    if result is not None or value not in vardefs:
                                        vardefs[value] = result if result is not None else placeholder_type
                                        changed = True
                                for name, value in params.items():
                                    if isinstance(value, Argument):
                                        candidates = [p.type for p in function.params if p.name == name]
                                        if not candidates:
                                            type_ = placeholder_type
                                        else:
                                            type_ = candidates[0]
                                        result = self.__update_vardefs(root, value, vardefs.get(value, placeholder_type), type_)
                                        if result is not None or value not in vardefs:
                                            vardefs[value] = result if result is not None else placeholder_type
                                            changed = True
                    i += 1

        for root in self.roots:
            vardefs = known_roots_types[root.name]
            for name, type_ in vardefs.items():
                if type_ == placeholder_type:
                    self.__update_vardefs(root, name, placeholder_type, AnyType)

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
        new_noop_node = NoopNode(f'noop_{src.name}-{dest.name}')
        self.nodes[new_noop_node.name] = new_noop_node

        src.reroute_out_edge(dest, new_noop_node)
        dest.del_in_edge(src)
        new_noop_node.add_in_edge(src)

    def __detach_nodes_with_call(self, src: Node, dest: Node, entry_point: str, vardefs: List[RootNode.VarDef]) -> None:
        new_call_node = SubflowNode(f'ext_{src.name}-{dest.name}', '', entry_point,
                params={v.name: Argument(v.name) for v in vardefs})
        self.nodes[new_call_node.name] = new_call_node

        src.reroute_out_edge(dest, new_call_node)
        dest.del_in_edge(src)
        new_call_node.add_in_edge(src)

    def __find_root(self, node: Node) -> RootNode:
        # no good if graph isn't separated yet but..
        s: List[Node] = [node]
        visited: Set[Node] = set()
        while s:
            node = s.pop()
            if isinstance(node, RootNode):
                return node
            if node.group_node is not None:
                if node.group_node not in visited:
                    s.append(node.group_node)
                    visited.add(node.group_node)
            else:
                in_nodes = set(n for n in node.in_edges if n != node)
                new_in_nodes = in_nodes - visited
                for n in new_in_nodes:
                    visited.add(n)
                    s.append(n)

        raise RuntimeError('root not found')

    def __detach_root(self, root: RootNode) -> RootNode:
        entry_point = root.out_edges[0]
        return self.__detach_node_as_sub(root, entry_point)

    def __detach_node_as_sub(self, root: RootNode, entry_point: Node) -> RootNode:
        name = entry_point.name if not isinstance(entry_point, EntryPointNode) else entry_point.entry_label
        new_root = RootNode(f'Sub_{name}', [RootNode.VarDef(v.name, v.type_, None) for v in root.vardefs], local=True)
        new_root.add_out_edge(entry_point)

        self.nodes[new_root.name] = new_root
        self.roots.append(new_root)

        for caller in entry_point.in_edges[:]:
            self.__detach_nodes_with_call(caller, entry_point, new_root.name, new_root.vardefs)
        entry_point.in_edges = [new_root]

        # may have been detached from a group, which is not in self.nodes
        for n in self.__find_postorder(new_root):
            if n not in self.nodes:
                self.nodes[n.name] = n

        return new_root

    def __convert_root_to_entrypoints(self, root: RootNode) -> None:
        # determine the "exclusive" part of the graph owned by root (i.e. no in edges)
        # for all destinations of out edges of exclusive subgraph:
        #   inject a label, funnel connections through label
        #   disconnect subgraph from label via call
        excl, connections = self.__get_exclusive_subgraph(root)
        simple_connections = [n for n in connections if not isinstance(n, GroupNode) and len(n.out_edges) <= 1 and all(type(o) == TerminalNode for o in n.out_edges)]
        complex_connections = [n for n in connections if n not in simple_connections]

        labels = set([self.__convert_node_to_entrypoint(n, n.name) for n in complex_connections])
        for node in excl:
            leaving_nodes: Set[EntryPointNode] = set(node.out_edges).intersection(labels) # type: ignore
            for label in leaving_nodes:
                self.__detach_nodes_with_call(node, label, label.entry_label, [])

        # for node in simple_connections:
        # probably should make a copy here

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
            tn = TerminalNode(f'tm_{root.name}')
            for node in self.__find_postorder(root):
                if isinstance(node, SwitchNode):
                    node.register_terminal_node(tn)
                elif len(node.out_edges) == 0 and not isinstance(node, TerminalNode):
                    node.add_out_edge(tn)
                    tn.add_in_edge(node)

            if len(tn.in_edges) > 0:
                self.nodes[tn.name] = tn

    def __find_dominator_tree(self, root: Node, reverse: bool = False) -> Dict[Node, Node]:
        back_edges = self.__add_while_back_edge()
        roots: List[Node]
        if reverse:
            roots = self.__find_terminals(root)
        else:
            roots = [root]
        dummy_root = NoopNode('')
        for root in roots:
            if not reverse:
                dummy_root.out_edges.append(root)
                root.in_edges.append(dummy_root)
            else:
                dummy_root.in_edges.append(root)
                root.out_edges.append(dummy_root)
        dom: Dict[Node, Node] = {dummy_root: dummy_root}

        # https://www.cs.rice.edu/~keith/Embed/dom.pdf
        rpo = self.__find_reverse_postorder(dummy_root, reverse=reverse)
        rpo_index = {k: i for i, k in enumerate(rpo)}
        changed = True
        while changed:
            changed = False
            for b in rpo[1:]: # skip dummy_root

                new_idom: Optional[Node] = None
                for p in (b.in_edges if not reverse else b.out_edges):
                    if p in dom:
                        if new_idom is None:
                            new_idom = p
                        # lca of p, new_idom
                        while p is not new_idom:
                            while rpo_index[p] > rpo_index[new_idom]:
                                p = dom[p]
                            while rpo_index[p] < rpo_index[new_idom]:
                                new_idom = dom[new_idom]

                assert new_idom is not None, str([r.name for r in rpo])

                if b not in dom or dom[b] is not new_idom:
                    dom[b] = new_idom
                    changed = True
        for root in roots:
            if not reverse:
                root.in_edges.remove(dummy_root)
            else:
                root.out_edges.remove(dummy_root)
            dom[root] = root
        for n in list(dom.keys()):
            if dom[n] is dummy_root:
                del dom[n]
        self.__del_while_back_edge(back_edges)
        return dom

    def __find_postorder_helper(self, root: Node, pred: Callable[[Node], bool], visited: Set[str], reverse: bool = False) -> List[Node]:
        po: List[Node] = []
        if not pred(root):
            return po
        for node in (root.out_edges if not reverse else root.in_edges):
            if node.name not in visited:
                visited.add(node.name)
                po.extend(self.__find_postorder_helper(node, pred, visited, reverse))
        po.append(root)
        return po

    def __find_postorder(self, roots: Union[Node, List[Node]], pred: Callable[[Node], bool] = lambda n: True, reverse: bool = False) -> List[Node]:
        roots = [roots] if isinstance(roots, Node) else roots
        out = []
        visited: Set[str] = set()
        for root in roots:
            out.extend(self.__find_postorder_helper(root, pred, visited, reverse))
        return out

    def __find_reverse_postorder(self, roots: Union[Node, List[Node]], pred: Callable[[Node], bool] = lambda n: True, reverse: bool = False) -> List[Node]:
        return self.__find_postorder(roots, pred, reverse)[::-1]

    def __path_exists(self, src: Node, dest: Node) -> bool:
        # todo: don't be this dumb
        return dest in self.__find_postorder(src)

    def __dominates(self, dominator: Node, dominatee: Node, tree: Dict[Node, Node]) -> bool:
        if dominator == dominatee:
            return True

        node = dominatee
        while node in tree and node != tree[node]:
            if tree[node] == dominator:
                return True
            node = tree[node]

        return False

    def __add_while_back_edge(self) -> List[Tuple[Node, Node]]:
        edges: List[Tuple[Node, Node]] = []
        for node in self.nodes.values():
            if isinstance(node, (WhileNode, DoWhileNode)):
                terminals = self.__find_terminals(node.loop_body)
                for terminal in terminals:
                    terminal.out_edges.append(node)
                    node.in_edges.append(terminal)
                    edges.append((terminal, node))
        return edges

    def __del_while_back_edge(self, edges: List[Tuple[Node, Node]]) -> None:
        for from_, to in edges:
            from_.out_edges.remove(to)
            to.in_edges.remove(from_)

    def __is_contained_subgraph(self, src: Node, dest: Node) -> bool:
        # check that dest dominates all reachable nodes via paths not through src,
        # in the dominator tree rooted at src
        # we create a pointer node to src, so that src <--> X loops do not result in
        # mutual dominance
        ptr_node = NoopNode('temp')
        ptr_node.add_out_edge(src)
        src.add_in_edge(ptr_node)
        # ptr_node = src

        dom = self.__find_dominator_tree(ptr_node)
        rv = True
        for node in self.__find_postorder(dest, lambda n: n != src):
            if not self.__dominates(dest, node, dom):
                rv = False
                break

        src.del_in_edge(ptr_node)
        return rv

    def __split_loops_helper(self, root: RootNode, node: Node, active: Set[Node], visited: Set[Node], replacement_nodes: Dict[Node, Node]) -> bool:
        active.add(node)

        for nxt in node.out_edges[:]:
            if nxt not in visited and nxt not in active:
                rv = self.__split_loops_helper(root, nxt, active, visited, replacement_nodes)
                if rv:
                    return True
            elif nxt in active and nxt not in visited:
                # loop starting at nxt, with a final node -> nxt edge to complete cycle
                # while (...) {...} loop conditions
                # - node has 1 child
                # - nxt 2 children A and B, with path from A -> B and not B -> A
                # - every node reachable from A without going through node has a path to node
                # -> A is body, B is exit
                # do { ...} while (...) loop conditions
                # - 1 child (body), exit branch node must be node
                # - every node reachable from nxt without going through node has a path to node
                # - node should have two children (nxt, out), out should not have a path to nxt
                loop_cond: Predicate
                if len(node.out_edges) == 1 and len(nxt.out_edges) == 2 and isinstance(nxt, SwitchNode):
                    # check for while pattern
                    A, B = nxt.out_edges
                    if self.__path_exists(B, A):
                        A, B = B, A
                    if self.__path_exists(A, B) and not self.__path_exists(B, A):
                        if self.__is_contained_subgraph(A, nxt):
                            if A.name in nxt.cases:
                                loop_cond = QueryPredicate(nxt.query, nxt.params, nxt.cases[A.name])
                            else:
                                assert B.name in nxt.cases and len(nxt.cases) == 1
                                loop_cond = ~QueryPredicate(nxt.query, nxt.params, nxt.cases[B.name])
                            node.del_out_edge(nxt)
                            nxt.del_in_edge(node)
                            while_node = self.__inject_while(nxt, loop_cond, A, B)
                            active.add(while_node)
                            replacement_nodes[nxt] = while_node
                            return True
                elif len(node.out_edges) == 1 and len(nxt.out_edges) == 1 and isinstance(nxt, SwitchNode):
                    # check for while true pattern
                    if self.__is_contained_subgraph(nxt.out_edges[0], nxt):
                        A = nxt.out_edges[0]
                        node.del_out_edge(nxt)
                        inf_terminal = DeadendTerminalNode(f'infloop_{nxt.name}')
                        nxt.add_out_edge(inf_terminal)
                        inf_terminal.add_in_edge(nxt)
                        self.nodes[inf_terminal.name] = inf_terminal

                        while_node = self.__inject_while(nxt, ConstPredicate(True), A, inf_terminal)
                        active.add(while_node)
                        replacement_nodes[nxt] = while_node
                        return True
                elif len(node.out_edges) == 2 and isinstance(node, SwitchNode):
                    # check for do-while pattern
                    A, B = node.out_edges
                    if B == nxt:
                        A, B = B, A
                    if not self.__path_exists(B, A):
                        if self.__is_contained_subgraph(A, node):
                            # delete node and funnel nxt inputs through a DoWhileNode
                            for in_edge in node.in_edges:
                                in_edge.del_out_edge(node)
                            for out_edge in node.out_edges:
                                out_edge.del_in_edge(node)
                            del self.nodes[node.name]

                            if A.name in node.cases:
                                loop_cond = QueryPredicate(node.query, node.params, node.cases[A.name])
                            else:
                                assert B.name in node.cases and len(node.cases) == 1
                                loop_cond = ~QueryPredicate(node.query, node.params, node.cases[B.name])
                            do_while_node = DoWhileNode(f'dw_{nxt.name}', loop_cond, nxt, B)

                            for in_edge in A.in_edges:
                                in_edge.reroute_out_edge(A, do_while_node)
                                do_while_node.add_in_edge(in_edge)
                                A.del_in_edge(in_edge)
                            do_while_node.add_out_edge(A)
                            A.add_in_edge(do_while_node)
                            do_while_node.add_out_edge(B)
                            B.add_in_edge(do_while_node)
                            self.nodes[do_while_node.name] = do_while_node
                            active.add(do_while_node)
                            replacement_nodes[nxt] = do_while_node
                            return True

                # doesn't match our loop patterns, so rewrite with goto
                entrypoint = self.__convert_node_to_entrypoint(nxt, nxt.name)
                self.__detach_nodes_with_call(node, entrypoint, entrypoint.entry_label, [])

                if nxt != entrypoint:
                    active.add(entrypoint)
                    replacement_nodes[nxt] = entrypoint
                return True

        active.remove(node)
        visited.add(node)
        if node in replacement_nodes:
            active.remove(replacement_nodes[node])
            visited.add(replacement_nodes[node])
        return False

    def __inject_while(self, loop_head, loop_cond, loop_body, loop_tail) -> WhileNode:
        while_node = WhileNode(loop_head.name, loop_cond, loop_body, loop_tail)

        for in_edge in loop_head.in_edges:
            in_edge.reroute_out_edge(loop_head, while_node)
            while_node.add_in_edge(in_edge)
        for out_edge in loop_head.out_edges:
            out_edge.reroute_in_edge(loop_head, while_node)
            while_node.add_out_edge(out_edge)

        self.nodes[loop_head.name] = while_node
        return while_node

    def __split_loops(self) -> None:
        for root in self.roots:
            while self.__split_loops_helper(root, root, set(), set(), {}):
                pass

    def __collapse_unconditional_switch(self) -> None:
        for root in self.roots:
            for node in self.__find_postorder(root):
                if not isinstance(node, SwitchNode):
                    continue
                if len(node.cases) > 1:
                    continue
                if len(node.cases) == 0:
                    # replace node with node.terminal_node
                    assert node.terminal_node is not None
                    replacement = node.terminal_node
                elif sum(len(v) for v in node.cases.values()) == node.query.num_values:
                    assert len(node.out_edges) == 1
                    replacement = node.out_edges[0]
                else:
                    continue
                for parent in node.in_edges:
                    parent.reroute_out_edge(node, replacement)
                    replacement.add_in_edge(parent)
                replacement.del_in_edge(node)
                del self.nodes[node.name]

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
                rdom = self.__find_dominator_tree(node, reverse=True)
                cur_end = self.__find_block_end(node, dom, rdom)
                else_end = self.__find_block_end(else_branch, dom, rdom)
                if cur_end != else_end:
                    continue

                ifelse_node = IfElseNode(node.name, [
                    IfElseNode.Rule(predicate, value_branch),
                    *else_branch.rules
                ], else_branch.default)

                self.__merge_coupled_nodes(node, else_branch, ifelse_node)

    def __merge_coupled_nodes(self, parent_node: Node, child_node: Node, new_node: Node) -> None:
        for caller in parent_node.in_edges:
            new_node.in_edges.append(caller)
            caller.reroute_out_edge(parent_node, new_node)

        for caller in child_node.in_edges:
            if caller is not parent_node:
                new_node.in_edges.append(caller)
                caller.reroute_out_edge(child_node, new_node)

        for parent_out in parent_node.out_edges:
            if parent_out is not child_node:
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

    def __find_terminals(self, node: Node) -> List[Node]:
        return [n for n in self.__find_postorder(node) if not n.out_edges]

    def __find_block_end(self, node: Node, dom: Dict[Node, Node], rdom: Dict[Node, Node]) -> Optional[Node]:
        if node not in rdom:
            return None

        if not all((child in dom and dom[child] is node) or isinstance(child, TerminalNode) for child in node.out_edges):
            return None

        end = rdom[node]
        if end not in dom:
            return None

        if isinstance(end, TerminalNode):
            inner_dom = self.__find_dominator_tree(node)
            if inner_dom[end] is not node:
                return None
        elif dom[end] is not node:
            return None

        if not all(self.__path_exists(child, end) for child in node.out_edges):
            return None

        return end

    def __try_collapse_block(self, entry: RootNode, root: Node) -> Node:
        dom = self.__find_dominator_tree(entry)
        rdom = self.__find_dominator_tree(entry, reverse=True)
        end = self.__find_block_end(root, dom, rdom)

        # print(entry.name, root.name, end)
        # print('Dominator Tree')
        # for name, node in dom.items():
            # print('\t', name.name, '->', node.name)
        # print('Reverse Dominator Tree')
        # for name, node in rdom.items():
            # print('\t', name.name, '->', node.name)

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

        inner_terminal: Node = sw.pass_node

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

            if isinstance(n, EntryPointNode):
                for node in n.in_edges:
                    if node not in dom:
                        self.__detach_nodes_with_call(node, n, n.entry_label, [])

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

    def __extract_reused_blocks(self, extract_single: bool = False, nodes: List[Node] = None) -> None:
        if nodes is None:
            nodes = list(self.nodes.values())

        # print('call', nodes)
        # if a node has two different node parents and is the root of a DAG, and is not a single line,
        #   extract to subflow
        for node in nodes:
            if len(set(node.in_edges)) >= 2 and self.__is_cut([node]) and \
                    (extract_single or node.out_edges or not isinstance(node, (ActionNode, TerminalNode, NoopNode, EntryPointNode, SubflowNode))):
                self.__detach_node_as_sub(self.__find_root(node), node)
            elif isinstance(node, GroupNode):
                l = node.nodes[:]
                l.remove(node.root)
                self.__extract_reused_blocks(extract_single, l)
                node.recalculate_group()

    def __remove_redundant_entrypoints(self) -> None:
        remapped_entrypoints: Dict[str, Tuple[str, List[RootNode.VarDef]]] = {}
        new_roots = []
        for root in self.roots:
            new_root = root
            for node in self.__find_postorder(root):
                if isinstance(node, (RootNode, EntryPointNode)) and len(node.out_edges) == 1 \
                        and isinstance(node.out_edges[0], EntryPointNode):
                    new_parent: Node
                    if isinstance(node, RootNode):
                        new_root = RootNode(node.name, node.vardefs, node.local)
                        new_parent = new_root
                        remapped_entrypoints[node.out_edges[0].entry_label] = (node.name, node.vardefs)
                    else:
                        new_parent = EntryPointNode(node.name, node.entry_label)
                        remapped_entrypoints[node.out_edges[0].entry_label] = (node.entry_label, [])
                    self.__merge_coupled_nodes(node, node.out_edges[0], new_parent)
            new_roots.append(new_root)
        self.roots = new_roots
        remap = {('', old): ('', new[0], new[1]) for old, new in remapped_entrypoints.items()}
        for root in self.roots:
            root.remap_subflow(remap)

    def __collapse_subflow_only_root(self) -> None:
        remapped_roots: Dict[str, Tuple[str, str]] = {}
        for root in self.roots:
            if len(root.out_edges) == 1 \
                    and isinstance(root.out_edges[0], SubflowNode) \
                    and (not root.out_edges[0].out_edges or \
                            isinstance(root.out_edges[0].out_edges[0], TerminalNode)):
                if root.local:
                    if root.name not in remapped_roots:
                        remapped_roots[root.name] = (root.out_edges[0].ns, root.out_edges[0].called_root_name)
                elif root.out_edges[0].ns == '' and \
                        root.out_edges[0].called_root_name not in remapped_roots:
                    called_root_ = [r for r in self.roots if r.name == root.out_edges[0].called_root_name]
                    if called_root_ and called_root_[0].local:
                        called_root = called_root_[0]
                        called_root.name, root.name = root.name, called_root.name
                        called_root.local, root.local = root.local, called_root.local
                        called_root.vardefs, root.vardefs = root.vardefs, called_root.vardefs
                        self.nodes[root.name], self.nodes[called_root.name] = root, called_root
                        remapped_roots[root.name] = ('', called_root.name)

        self.roots = [root for root in self.roots if root.name not in remapped_roots]
        remap: Dict[Tuple[str, str], Tuple[str, str, List[RootNode.VarDef]]] = {('', old): (new[0], new[1], []) for old, new in remapped_roots.items()}
        for root in self.roots:
            root.remap_subflow(remap)

    def __simplify_all(self) -> None:
        for node in self.nodes.values():
            node.simplify()

    def __remove_trailing_return(self) -> None:
        for root in self.roots:
            prev_node: Node = root
            cur_node: Node = root
            while cur_node.out_edges:
                if isinstance(cur_node, (WhileNode, DoWhileNode)):
                    prev_node, cur_node = cur_node, cur_node.loop_exit
                    continue
                if len(cur_node.out_edges) != 1:
                    break
                prev_node, cur_node = cur_node, cur_node.out_edges[0]
            if isinstance(cur_node, TerminalNode):
                if not isinstance(prev_node, RootNode):
                    prev_node.del_out_edge(cur_node)

    def generate_code(self, generator: CodeGenerator) -> str:
        code: List[str] = []
        if code:
            code.append('')

        self.roots.sort(key=lambda x: x.name)
        code.extend('\n'.join(generator.generate_code(root) for root in self.roots).split('\n'))

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

    def import_functions(self, actors: List[Tuple[str, str, str]], actions: Dict[str, Any],
                         queries: Dict[str, Any]) -> None:
        for actor_name, sec_name, arg_name in actors:
            full_actor_name = (actor_name, sec_name, arg_name)
            if actor_name not in self.actors:
                self.actors[full_actor_name] = Actor(full_actor_name)
            for action, info in actions.items():
                self.actors[full_actor_name].register_action(Action(
                    full_actor_name,
                    action,
                    [Param(name, Type(type_)) for name, type_ in info['params'].items()],
                    info.get('varargs'),
                    info.get('conversion', None),
                ))
            for query, info in queries.items():
                self.actors[full_actor_name].register_query(Query(
                    full_actor_name,
                    query,
                    [Param(name, Type(type_)) for name, type_ in info['params'].items()],
                    info.get('varargs'),
                    Type(info.get('return', 'any')),
                    info.get('inverted', False),
                    info.get('conversion', None),
                    info.get('neg_conversion', None),
                ))

        for a in self.actors.values():
            a.lock_registration()

    def export_actors(self) -> Dict[str, Any]:
        e: Dict[str, Any] = {}
        for actor in self.actors.values():
            e.update(actor.export())
        return e

    def add_actor(self, actor: Actor) -> None:
        if actor.name not in self.actors:
            self.actors[actor.name] = actor

    def add_node(self, node: Node) -> None:
        self.nodes[node.name] = node

        if isinstance(node, RootNode):
            self.roots.append(node)

    def add_edge(self, src: str, dest: str) -> None:
        assert src in self.nodes
        assert dest in self.nodes

        src_node = self.nodes[src]
        dest_node = self.nodes[dest]

        src_node.add_out_edge(dest_node)
        dest_node.add_in_edge(src_node)

    def prepare(self) -> None:
        self.__separate_overlapping_flows()
        self.__add_terminal_nodes()

        self.__split_loops()

    def restructure(
        self,
        remove_redundant_switch: bool = False,
        switch_to_if: bool = True,
        collapse_andor: bool = True,
        collapse_if: bool = True,
        collapse_case: bool = True,
        remove_trailing_return: bool = True,
        extract_reused_blocks: bool = True,
        extract_single_statement: bool = False,
        remove_redundant_entrypoints: bool = True,
        collapse_subflow_only: bool = True,
        simplify_ifelse_order: bool = True,
        secondary_max_iter: int = 10000,
    ) -> None:
        if remove_redundant_switch:
            self.__collapse_unconditional_switch()
        if switch_to_if:
            self.__convert_switch_to_if()
        if collapse_andor:
            self.__collapse_andor()
        if collapse_if:
            self.__collapse_if()
        if collapse_case:
            self.__collapse_cases()
        old_nodes = None
        i = 0
        while i < secondary_max_iter and old_nodes != set(self.nodes.values()):
            old_nodes = set(self.nodes.values())
            if extract_reused_blocks:
                self.__extract_reused_blocks(extract_single_statement)
            if remove_redundant_entrypoints:
                self.__remove_redundant_entrypoints()
            if collapse_subflow_only:
                self.__collapse_subflow_only_root()
            if simplify_ifelse_order:
                self.__simplify_all()
            i += 1

        if remove_trailing_return:
            self.__remove_trailing_return()

    def final_pass(self) -> None:
        self.__add_implicit_vardefs()

    def get_dot(self, search_from_roots: bool = False) -> str:
        # search_from_roots = True may be more useful for debugging in some cases
        if not search_from_roots:
            return f'digraph {self.name}' + '{ compound=true; ' + ''.join(n.get_dot() for n in self.nodes.values()) + '}'
        else:
            return f'digraph {self.name}' + '{ compound=true; ' + ''.join(n.get_dot() for r in self.roots for n in self.__find_postorder(r)) + '}'

    def check_graph(self) -> None:
        for node in self.nodes.values():
            for in_edge in node.in_edges:
                assert node in in_edge.out_edges, f'{node} - {in_edge}'
            for out_edge in node.out_edges:
                assert node in out_edge.in_edges, f'{node} - {out_edge}'
