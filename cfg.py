from __future__ import annotations

import re
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from datatype import Type
from predicates import Predicate, ConstPredicate, QueryPredicate
from actors import Param, Action, Query, Actor
from nodes import Node, RootNode, ActionNode, SwitchNode, SubflowNode, TerminalNode, DeadendTerminalNode, NoopNode, EntryPointNode, GroupNode, IfElseNode, WhileNode, DoWhileNode

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
            if set(n.in_edges) - reachable and n not in ns and not isinstance(n, TerminalNode):
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

    def __find_root(self, node: Node) -> List[RootNode]:
        # no good if graph isn't separated yet but..
        s: List[Node] = [node]
        visited: Set[Node] = set()
        roots: List[RootNode] = []
        while s:
            node = s.pop()
            if isinstance(node, RootNode):
                roots.append(node)
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

        if not roots:
            raise RuntimeError('root not found')

        return roots

    def __detach_root(self, root: RootNode) -> RootNode:
        entry_point = root.out_edges[0]
        return self.__detach_node_as_sub([root], entry_point)

    def __detach_node_as_sub(self, roots: List[RootNode], entry_point: Node) -> RootNode:
        name = entry_point.name if not isinstance(entry_point, EntryPointNode) else entry_point.entry_label
        vardefs = roots[0].vardefs
        for root in roots[1:]:
            vardefs += root.vardefs
        new_root = RootNode(f'sub_{name}', list(set(vardefs)))
        new_root.add_out_edge(entry_point)

        for caller in entry_point.in_edges[:]:
            self.__detach_nodes_with_call(caller, entry_point, new_root.name)

        entry_point.in_edges = [new_root]
        self.nodes[new_root.name] = new_root
        self.roots.append(new_root)

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
                self.__detach_nodes_with_call(node, label, label.entry_label)

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

    def __dominates(self, dominator: Node, dominatee: Node, tree: Dict[Node, Node]) -> bool:
        if dominator == dominatee:
            return True

        node = dominatee
        while node != tree[node]:
            if tree[node] == dominator:
                return True
            node = tree[node]

        return False

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

    def __split_loops_helper(self, node: Node, active: Set[Node], visited: Set[Node], replacement_nodes: Dict[Node, Node]) -> None:
        active.add(node)

        for nxt in node.out_edges[:]:
            if nxt not in visited and nxt not in active:
                self.__split_loops_helper(nxt, active, visited, replacement_nodes)
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
                            while_node = self.__inject_while(nxt, loop_cond, A, B)
                            active.add(while_node)
                            replacement_nodes[nxt] = while_node
                            break
                elif len(node.out_edges) == 1 and len(nxt.out_edges) == 1:
                    # check for while true pattern
                    if self.__is_contained_subgraph(nxt.out_edges[0], nxt):
                        A = nxt.out_edges[0]
                        node.del_out_edge(nxt)
                        inf_terminal = DeadendTerminalNode(f'infloop!{nxt.name}')
                        nxt.add_out_edge(inf_terminal)
                        inf_terminal.add_in_edge(nxt)
                        self.nodes[inf_terminal.name] = inf_terminal

                        while_node = self.__inject_while(nxt, ConstPredicate(True), A, inf_terminal)
                        active.add(while_node)
                        replacement_nodes[nxt] = while_node
                        break
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
                            del self.nodes[node.name]

                            if A.name in node.cases:
                                loop_cond = QueryPredicate(node.query, node.params, node.cases[A.name])
                            else:
                                assert B.name in node.cases and len(node.cases) == 1
                                loop_cond = ~QueryPredicate(node.query, node.params, node.cases[B.name])
                            do_while_node = DoWhileNode(f'dw!{nxt.name}', loop_cond, nxt, B)

                            for in_edge in nxt.in_edges:
                                in_edge.reroute_out_edge(nxt, do_while_node)
                                do_while_node.add_in_edge(in_edge)
                                nxt.del_in_edge(in_edge)
                            do_while_node.add_out_edge(A)
                            A.reroute_in_edge(node, do_while_node)
                            do_while_node.add_out_edge(B)
                            B.reroute_in_edge(node, do_while_node)
                            self.nodes[do_while_node.name] = do_while_node
                            active.add(do_while_node)
                            replacement_nodes[nxt] = do_while_node
                            break

                # doesn't match our loop patterns, so rewrite with goto
                entrypoint = self.__convert_node_to_entrypoint(nxt, nxt.name)
                self.__detach_nodes_with_call(node, entrypoint, entrypoint.entry_label)

                active.add(entrypoint)
                replacement_nodes[nxt] = entrypoint

        active.remove(node)
        visited.add(node)
        if node in replacement_nodes:
            active.remove(replacement_nodes[node])
            visited.add(replacement_nodes[node])

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
                    (node.out_edges or not isinstance(node, (ActionNode, TerminalNode, NoopNode, EntryPointNode, SubflowNode))):
                self.__detach_node_as_sub(self.__find_root(node), node)
            elif isinstance(node, GroupNode):
                l = node.nodes[:]
                l.remove(node.root)
                self.__extract_reused_blocks(l)
                node.recalculate_group()

    def __remove_redundant_entrypoints(self) -> None:
        remapped_entrypoints: Dict[str, str] = {}
        new_roots = []
        for root in self.roots:
            new_root = root
            for node in self.__find_postorder(root):
                if isinstance(node, (RootNode, EntryPointNode)) and len(node.out_edges) == 1 \
                        and isinstance(node.out_edges[0], EntryPointNode):
                    new_parent: Node
                    if isinstance(node, RootNode):
                        new_root = RootNode(node.name, node.vardefs)
                        new_parent = new_root
                        remapped_entrypoints[node.out_edges[0].entry_label] = node.name
                    else:
                        new_parent = EntryPointNode(node.name, node.entry_label)
                        remapped_entrypoints[node.out_edges[0].entry_label] = node.entry_label
                    self.__merge_coupled_nodes(node, node.out_edges[0], new_parent)
            new_roots.append(new_root)
        self.roots = new_roots
        for root in self.roots:
            for node in self.__find_postorder(root):
                if isinstance(node, SubflowNode):
                    if node.called_root_name in remapped_entrypoints:
                        node.called_root_name = remapped_entrypoints[node.called_root_name]

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

    def import_actors(self, actor_data: Dict[str, Any]) -> None:
        for actor_name, d in actor_data.items():
            if actor_name not in self.actors:
                self.actors[actor_name] = Actor(actor_name)
            for action, info in d['actions'].items():
                self.actors[actor_name].register_action(Action(
                    actor_name,
                    action,
                    [Param(name, Type(type_)) for name, type_ in info['params'].items()],
                    info.get('conversion', None),
                ))
            for query, info in d['queries'].items():
                self.actors[actor_name].register_query(Query(
                    actor_name,
                    query,
                    [Param(name, Type(type_)) for name, type_ in info['params'].items()],
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

    def restructure(self) -> None:
        self.__separate_overlapping_flows()
        self.__add_terminal_nodes()

        old_nodes = None
        while old_nodes != set(self.nodes.values()):
            old_nodes = set(self.nodes.values())

            self.__split_loops()
            self.__convert_switch_to_if()
            self.__collapse_andor()
            self.__collapse_if()
            self.__collapse_cases()
            self.__extract_reused_blocks()
            self.__remove_redundant_entrypoints()

            self.__simplify_all()

    def get_dot(self, search_from_roots: bool = False) -> str:
        # search_from_roots = True may be more useful for debugging in some cases
        if not search_from_roots:
            return f'digraph {self.name}' + '{ compound=true; ' + ''.join(n.get_dot() for n in self.nodes.values()) + '}'
        else:
            return f'digraph {self.name}' + '{ compound=true; ' + ''.join(n.get_dot() for r in self.roots for n in self.__find_postorder(r)) + '}'

