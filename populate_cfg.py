from __future__ import annotations

from typing import Any, Dict

import evfl

from actors import Param, Action, Query, Actor
from nodes import Node, RootNode, ActionNode, SwitchNode, ForkNode, JoinNode, SubflowNode
from cfg import CFG


def read(data: bytes, actor_data: Dict[str, Any]) -> CFG:
    flow = evfl.EventFlow()
    flow.read(data)

    flowchart = flow.flowchart

    cfg = CFG(flowchart.name)
    cfg.import_actors(actor_data)
    for r in flowchart.actors:
        cfg.add_actor(Actor(r.identifier.name))

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
            cfg.add_node(join_node)

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

        cfg.add_node(node)

    for name, node in cfg.nodes.items():
        if isinstance(node, (ActionNode, SubflowNode, JoinNode)):
            if node.nxt is not None:
                cfg.add_edge(name, node.nxt)
        elif isinstance(node, SwitchNode):
            for out_name in node.cases.keys():
                cfg.add_edge(name, out_name)
        elif isinstance(node, ForkNode):
            for out_name in node.forks:
                cfg.add_edge(name, out_name)
        else:
            raise RuntimeError('bad node type')

    for entry_point in flowchart.entry_points:
        node = RootNode(entry_point.name)
        cfg.add_node(node)
        if entry_point.main_event.v is not None:
            main_event = entry_point.main_event.v.name
            cfg.add_edge(node.name, main_event)

    return cfg

