from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Tuple, Union

from bitstring import ConstBitStream

from .logger import LOG

from .actors import Param, Action, Query, Actor
from .nodes import Node, RootNode, ActionNode, SwitchNode, ForkNode, JoinNode, SubflowNode
from .datatype import ActorMarker, AnyType, BoolType, FloatType, IntType, Type, Argument, infer_types
from .cfg import CFG

@dataclass
class BFEVFLActor:
    name: str
    secondary_name: str
    argument_name: str
    actions: List[Tuple[Action, bool]]
    queries: List[Tuple[Query, bool]]

@dataclass
class BFEVFL:
    filename: str = ''
    flowchart_name: str = ''
    actors: List[BFEVFLActor] = field(default_factory=list)
    nodes: List[Node] = field(default_factory=list)
    roots: List[RootNode] = field(default_factory=list)
    edges: List[Tuple[str, str]] = field(default_factory=list)

class read_at_offset:
    def __init__(self, bs: ConstBitStream, offset: int, offset_in_bits: bool = False, restore: bool = True) -> None:
        self.bs = bs
        self.offset = offset * (1 if offset_in_bits else 8)
        self.old_pos = 0
        self.restore = restore

    def __enter__(self) -> None:
        self.old_pos = self.bs.pos
        self.bs.pos = self.offset

    def __exit__(self, *args: Any) -> None:
        if self.restore:
            self.bs.pos = self.old_pos

def check_and_update_action(actor: BFEVFLActor, action_index: int, params: Dict[str, Any]):
    ptypes = infer_types(params)
    actor_name = (actor.name, actor.secondary_name, actor.argument_name)
    if not actor.actions[action_index][1]:
        actor.actions[action_index] = (Action(actor_name, actor.actions[action_index][0].name,
                [Param(n, t) for n, t in ptypes.items()], False), True)
    else:
        action = actor.actions[action_index][0]
        assert len(action.params) == len(ptypes)
        for i, param in enumerate(action.params):
            pname, ptype = param
            if ptypes[pname] != ptype:
                action.params[i] = Param(pname, AnyType)

def check_and_update_query(actor: BFEVFLActor, query_index: int, params: Dict[str, Any], rvs: Iterable[int]):
    ptypes = infer_types(params)
    actor_name = (actor.name, actor.secondary_name, actor.argument_name)
    if not actor.queries[query_index][1]:
        rt = Type(f'int{max(max(rvs) + 1 if rvs else 0, 2)}')
        actor.queries[query_index] = (Query(actor_name, actor.queries[query_index][0].name,
                [Param(n, t) for n, t in ptypes.items()], False, rt), True)
    else:
        query = actor.queries[query_index][0]
        assert len(query.params) == len(ptypes)
        for i, param in enumerate(query.params):
            pname, ptype = param
            if ptypes[pname] != ptype:
                query.params[i] = Param(pname, AnyType)
        mx = max(rvs) + 1 if rvs else 0
        if int(query.rv.type[3:]) < mx:
            query.rv = Type(f'int{max(mx, 2)}')

def _load_str(bs: ConstBitStream, offset: int, len_skipped: bool = False, **kwargs: Any) -> str:
    with read_at_offset(bs, offset - (2 if len_skipped else 0), **kwargs):
        length = bs.read('uintle:16')
        return bs.read(f'bytes:{length}').decode('utf-8')

def _load_dictionary(bs: ConstBitStream, offset: int, **kwargs: Any) -> List[str]:
    values = []

    with read_at_offset(bs, offset, **kwargs):
        assert bs.read('bytes:4') == b'DIC ' # magic
        num_entries = bs.read('uintle:32')

        # skip root
        bs.read('pad:128')

        for i in range(num_entries):
            bs.read('pad:64') # skip search stuff, just fetch string
            values.append(_load_str(bs, bs.read('uintle:64')))

        return values

def _load_container_item(bs: ConstBitStream, offset: int, **kwargs: Any) -> Any:
    rv: Any
    with read_at_offset(bs, offset, **kwargs):
        data_type = bs.read('uintle:16')
        num_items = bs.read('uintle:16')
        bs.read('pad:32') # padding
        dict_ptr = bs.read('uintle:64')

        if data_type == 0: # Argument
            assert num_items == 1
            rv = Argument(_load_str(bs, bs.read('uintle:64')))
        elif data_type == 1: # Container
            dict_ = _load_dictionary(bs, dict_ptr)
            assert len(dict_) == num_items

            rv = OrderedDict()
            for name in dict_:
                rv[name] = _load_container_item(bs, bs.read('uintle:64'))
        elif data_type in (2, 7): # Int
            assert num_items == 1 or data_type > 6

            rv = [bs.read('intle:32') for _ in range(num_items)]
            bs.read(f'pad:{64 - 32 * num_items % 64}')
            if data_type <= 6:
                rv = rv[0]
        elif data_type in (3, 8): # Bool
            assert num_items == 1 or data_type > 6

            rv = [bool(bs.read('uintle:8')) for _ in range(num_items)]
            bs.read(f'pad:{64 - 8 * num_items % 64}')
            if data_type <= 6:
                rv = rv[0]
        elif data_type in (4, 9): # Float
            assert num_items == 1 or data_type > 6

            rv = [bs.read('floatle:32') for _ in range(num_items)]
            bs.read(f'pad:{64 - 32 * num_items % 64}')
            if data_type <= 6:
                rv = rv[0]
        elif data_type in (5, 10): # String
            assert num_items == 1 or data_type > 6

            rv = [_load_str(bs, bs.read('uintle:64')) for _ in range(num_items)]
            if data_type <= 6:
                rv = rv[0]
        elif data_type == (6, 11): # wstring
            raise RuntimeError(f'wstring unsupported')
        elif data_type == 12: # Actor identifier
            assert num_items == 2
            rv = [_load_str(bs, bs.read('uintle:64')) for _ in range(num_items)]
            return ActorMarker(*rv)
        else:
            raise ValueError(f'bad data type: {data_type}')

        return rv

def _dereference(bs: ConstBitStream, ptr: int, data_type: str = 'uintle:64', **kwargs: Any) -> Any:
    with read_at_offset(bs, ptr, **kwargs):
        return bs.read(data_type)

def _load_actors(bs: ConstBitStream, offset: int, num_actors: int, **kwargs: Any) -> List[BFEVFLActor]:
    actors: List[BFEVFLActor] = []
    with read_at_offset(bs, offset, **kwargs):
        for _ in range(num_actors):
            name = _load_str(bs, bs.read('uintle:64'))
            secondary_name = _load_str(bs, bs.read('uintle:64'))
            argument_name = _load_str(bs, bs.read('uintle:64'))
            actions_ptr = bs.read('uintle:64')
            queries_ptr = bs.read('uintle:64')
            parameters_ptr = bs.read('uintle:64')
            parameters = _load_container_item(bs, parameters_ptr) if parameters_ptr else None
            num_actions = bs.read('uintle:16')
            num_queries = bs.read('uintle:16')
            ep_index = bs.read('uintle:16')
            bs.read('pad:16')

            if argument_name:
                # not really a safe assumption..?
                assert '(' in secondary_name and secondary_name.split('(')[1][:-1] == argument_name
                secondary_name = secondary_name.split('(')[0]
            assert parameters is None
            if ep_index != 0xffff:
                # this only marks which entrypoint the actor is used? in
                # not really important for decompile
                pass

            if actions_ptr:
                with read_at_offset(bs, actions_ptr):
                    actions = [(Action((name, secondary_name, argument_name), _load_str(bs, bs.read('uintle:64'))[15:], [], False), False) for _ in range(num_actions)]
            else:
                actions = []

            if queries_ptr:
                with read_at_offset(bs, queries_ptr):
                    queries = [(Query((name, secondary_name, argument_name), _load_str(bs, bs.read('uintle:64'))[14:], [], False, IntType), False) for _ in range(num_queries)]
            else:
                queries = []

            actor = BFEVFLActor(name, secondary_name, argument_name, actions, queries)
            actors.append(actor)
    return actors

def _load_events(bs: ConstBitStream, offset: int, num_events: int, actors: List[BFEVFLActor], **kwargs: Any) -> List[Node]:
    names: List[str] = []
    events: List[Node] = []
    join_nodes: Dict[int, JoinNode] = {}

    with read_at_offset(bs, offset, **kwargs):
        for _ in range(num_events):
            names.append(_load_str(bs, bs.read('uintle:64')))
            bs.read('pad:256')

    with read_at_offset(bs, offset, **kwargs):
        for index in range(num_events):
            name = names[index]
            bs.read('pad:64')
            event_type = bs.read('uintle:8')
            bs.read('pad:8') # padding

            if event_type == 0: # Action
                event_index = bs.read('uintle:16')
                next_ = names[event_index] if event_index != 0xFFFF else None
                actor_index = bs.read('uintle:16')
                actor_action_index = bs.read('uintle:16')
                container_ptr = bs.read('uintle:64')
                bs.read('pad:128')

                if container_ptr:
                    container = _load_container_item(bs, container_ptr)
                    assert isinstance(container, dict)
                else:
                    container = {}

                actor = actors[actor_index]
                check_and_update_action(actor, actor_action_index, container)

                events.append(ActionNode(name, actor.actions[actor_action_index][0], container, next_))
            elif event_type == 1: # Switch
                num_cases = bs.read('uintle:16')
                actor_index = bs.read('uintle:16')
                actor_query_index = bs.read('uintle:16')
                container_ptr = bs.read('uintle:64')
                swcase_ptr = bs.read('uintle:64')
                bs.read('pad:64')

                if container_ptr:
                    container = _load_container_item(bs, container_ptr)
                    assert isinstance(container, dict)
                else:
                    container = {}

                cases: Dict[int, str] = OrderedDict()
                if swcase_ptr:
                    with read_at_offset(bs, swcase_ptr):
                        for _ in range(num_cases):
                            # value, event index
                            value = bs.read('uintle:32')
                            event_index = bs.read('uintle:16')
                            cases[value] = names[event_index]
                            bs.read('pad:16')

                actor = actors[actor_index]
                check_and_update_query(actor, actor_query_index, container, cases.keys())
                query = actor.queries[actor_query_index][0]

                switch_node = SwitchNode(name, query, container)
                for value, ev in cases.items():
                    switch_node.add_case(ev, value)
                events.append(switch_node)
            elif event_type == 2: # fork
                num_forks = bs.read('uintle:16')
                join_index = bs.read('uintle:16')
                bs.read('pad:16')
                fork_array_ptr = bs.read('uintle:64')
                bs.read('pad:128')

                forks: List[str] = []
                if fork_array_ptr:
                    with read_at_offset(bs, fork_array_ptr):
                        forks = [names[bs.read('uintle:16')] for _ in range(num_forks)]

                join_node = JoinNode(names[join_index], None) if join_index > index else events[join_index]
                assert isinstance(join_node, JoinNode)
                join_nodes[join_index] = join_node

                node = ForkNode(name, join_node, forks)
                node.add_out_edge(join_node)
                join_node.add_in_edge(node)

                events.append(node)
            elif event_type == 3: # join
                event_index = bs.read('uintle:16')
                next_ = names[event_index] if event_index != 0xFFFF else None
                bs.read(f'pad:224')

                if index not in join_nodes:
                    join_node = JoinNode(name, next_)
                else:
                    join_node_ = join_nodes[index]
                    assert isinstance(join_node_, JoinNode)
                    join_node = join_node_
                join_node.nxt = next_

                events.append(join_node)
            elif event_type == 4: # subflow
                event_index = bs.read('uintle:16')
                next_ = names[event_index] if event_index != 0xFFFF else None
                bs.read('pad:32')
                container_ptr = bs.read('uintle:64')
                flowchart_name = _load_str(bs, bs.read('uintle:64'))
                entrypoint_name = _load_str(bs, bs.read('uintle:64'))

                if container_ptr:
                    container = _load_container_item(bs, container_ptr)
                    assert isinstance(container, dict)
                else:
                    container = {}

                events.append(SubflowNode(name, flowchart_name, entrypoint_name, next_, container))
            else:
                raise ValueError(f'unknown event type {event_type}')
    return events

def _load_entrypoints(bs: ConstBitStream, offset: int, num_entrypoints: int, names: List[str], **kwargs: Any) -> List[Tuple[RootNode, int]]:
    roots: List[Tuple[RootNode, int]] = []
    with read_at_offset(bs, offset, **kwargs):
        for index in range(num_entrypoints):
            bs.read('pad:64') # subflow event index array ptr
            vardef_name_ptr = bs.read('uintle:64')
            vardef_ptr = bs.read('uintle:64')
            bs.read('pad:16') # subflow event index array length
            num_vardefs = bs.read('uintle:16')
            main_event_index = bs.read('uintle:16')
            bs.read('pad:16') # padding

            vardefs: List[RootNode.VarDef] = []
            if num_vardefs > 0:
                vardef_names = _load_dictionary(bs, vardef_name_ptr)
                with read_at_offset(bs, vardef_ptr):
                    for name in vardef_names:
                        initial_value_raw = bs.read('bits:64')
                        assert bs.read('uintle:16') == 1 # number of items
                        type_enum = bs.read('uintle:16')

                        type_: Type
                        initial_value: Union[int, bool, float]
                        if type_enum == 2:
                            type_ = IntType
                            initial_value = initial_value_raw.read('intle:32')
                        elif type_enum == 3:
                            type_ = BoolType
                            initial_value = bool(initial_value_raw.read('uintle:8'))
                        elif type_enum == 4:
                            type_ = FloatType
                            initial_value = initial_value_raw.read('floatle:32')
                        else:
                            raise RuntimeError(f'unsupported type for vardef: {type_enum}')
                        bs.read('pad:32') # padding

                        vardefs.append(RootNode.VarDef(name, type_, initial_value))

            ep_name = names[index]
            node = RootNode(ep_name, vardefs)
            roots.append((node, main_event_index))
    return roots

def parse_bfevfl(data: bytes) -> BFEVFL:
    # format details: https://zeldamods.org/wiki/BFEVFL

    bs = ConstBitStream(bytes=data)
    bfevfl = BFEVFL()

    # header
    assert bs.read('bytes:8') == b'BFEVFL\0\0' # magic
    assert bs.read('uintle:16') == 0x0300 # version
    assert bs.read('uintle:8') == 0
    bs.read('pad:8')
    assert bs.read('uintle:16') == 0xFEFF # BOM, assume little endian
    alignment = 1 << bs.read('uintle:8')
    bs.read('pad:8')
    file_name_offset = bs.read('uintle:32')
    bs.read('pad:16') # is relocated flag
    first_block_offset = bs.read('uintle:16')
    reloc_table_offset = bs.read('uintle:32')
    assert bs.read('uintle:32') == len(data) # file size
    assert bs.read('uintle:16') == 1 # num flowcharts
    assert bs.read('uintle:16') == 0 # num timelines
    bs.read('pad:32')
    flowchart_array_ptr = bs.read('uintle:64')
    bs.read('pad:64') # flowchart name dict
    bs.read('pad:128') # timeline array ptr/name dict

    bfevfl.filename = _load_str(bs, file_name_offset, True)

    bs.pos = flowchart_array_ptr * 8
    bs.pos = bs.read('uintle:64') * 8 # flowchart_array[0]

    assert bs.read('bytes:4') == b'EVFL' # magic
    bs.read('pad:32') # string pool offset
    bs.read('pad:64') # padding
    num_actors = bs.read('uintle:16')
    bs.read('pad:32') # num_actions, num_queries
    num_events = bs.read('uintle:16')
    num_entrypoints = bs.read('uintle:16')
    bs.read('pad:48') # padding
    bfevfl.flowchart_name = _load_str(bs, bs.read('uintle:64'))
    actors_ptr = bs.read('uintle:64')
    events_ptr = bs.read('uintle:64')
    entrypoint_dict_ptr = bs.read('uintle:64')
    entrypoints_ptr = bs.read('uintle:64')

    bfevfl.actors = _load_actors(bs, actors_ptr, num_actors)
    entrypoint_dict = _load_dictionary(bs, entrypoint_dict_ptr)
    bfevfl.nodes = _load_events(bs, events_ptr, num_events, bfevfl.actors)

    for node in bfevfl.nodes:
        if isinstance(node, (ActionNode, SubflowNode, JoinNode)):
            if node.nxt is not None:
                bfevfl.edges.append((node.name, node.nxt))
        elif isinstance(node, SwitchNode):
            for out_name in node.cases.keys():
                bfevfl.edges.append((node.name, out_name))
        elif isinstance(node, ForkNode):
            for out_name in node.forks:
                bfevfl.edges.append((node.name, out_name))

    for root, event_index in _load_entrypoints(bs, entrypoints_ptr, num_entrypoints, entrypoint_dict):
        bfevfl.roots.append(root)
        if event_index != 0xffff:
            bfevfl.edges.append((root.name, bfevfl.nodes[event_index].name))

    return bfevfl

def read(data: bytes, actions: Dict[str, Any], queries: Dict[str, Any]) -> CFG:
    bfevfl = parse_bfevfl(data)

    cfg = CFG(bfevfl.flowchart_name)
    cfg.import_functions([(r.name, r.secondary_name, r.argument_name) for r in bfevfl.actors], actions, queries)
    for r in bfevfl.actors:
        actor = cfg.actors[(r.name, r.secondary_name, r.argument_name)]
        for action, initialized in r.actions:
            if initialized and action.name not in actor.actions:
                LOG.warning(f'untyped action: {action}')
                actor.register_action(Action(action.actor_name, action.name, [], True))
        for query, initialized in r.queries:
            if initialized and query.name not in actor.queries:
                LOG.warning(f'untyped query: {query}')
                actor.register_query(Query(query.actor_name, query.name, [], True, query.rv))

    for node in bfevfl.nodes + bfevfl.roots: # type: ignore
        if isinstance(node, ActionNode):
            node.action = cfg.actors[node.action.actor_name].actions[node.action.name]
        elif isinstance(node, SwitchNode):
            node.query = cfg.actors[node.query.actor_name].queries[node.query.name]
        cfg.add_node(node)

    for src, dest in bfevfl.edges:
        cfg.add_edge(src, dest)

    cfg.prepare()
    return cfg

