from __future__ import annotations

from typing import Callable, Dict, List

from codegen import CodeGenerator, NodeCodeGenerator, PredicateCodeGenerator
from datatype import AnyType, BoolType, Type, Argument
from indent import indent
from nodes import *
from predicates import *

node_codegen: Dict[type, NodeCodeGenerator] = {}
pred_codegen: Dict[type, PredicateCodeGenerator] = {}

def node_generator(typ: type) -> Callable[[NodeCodeGenerator], NodeCodeGenerator]:
    def inner(f: NodeCodeGenerator) -> NodeCodeGenerator:
        node_codegen[typ] = f
        return f
    return inner

def pred_generator(typ: type) -> Callable[[PredicateCodeGenerator], PredicateCodeGenerator]:
    def inner(f: PredicateCodeGenerator) -> PredicateCodeGenerator:
        pred_codegen[typ] = f
        return f
    return inner

def node_generate_code(node: Node, indent_level: int = 0, generate_pass: bool = False) -> str:
    return node_codegen[type(node)](node, indent_level, generate_pass)

def pred_generate_code(pred: Predicate) -> str:
    return pred_codegen[type(pred)](pred)

class EVFLCodeGenerator(CodeGenerator):
    def generate_actor_annotation(self, actor_name: str, secondary_name: str) -> str:
        return f'@{actor_name}:{secondary_name}'

    def generate_code(self, node: Node, indent_level: int = 0, generate_pass: bool = False) -> str:
        return node_generate_code(node, indent_level, generate_pass)

@node_generator(RootNode)
def RootNode_generate_code(self_node: Node, indent_level: int = 0, generate_pass: bool = False) -> str:
    assert isinstance(self_node, RootNode)

    return f'{indent(indent_level)}flow {self_node.name}({", ".join(str(v) for v in self_node.vardefs)}):\n' + \
            '\n'.join(node_generate_code(n, indent_level + 1) for n in self_node.out_edges)

@node_generator(ActionNode)
def ActionNode_generate_code(self_node: Node, indent_level: int = 0, generate_pass: bool = False) -> str:
    assert isinstance(self_node, ActionNode)

    hint_s = ''.join(f'{indent(indent_level)}# {hint}\n' for hint in self_node.action.hint(self_node.params))
    return hint_s + f'{indent(indent_level)}{Action_format(self_node.action, self_node.params)}\n' + \
            '\n'.join(node_generate_code(n, indent_level) for n in self_node.out_edges)

@node_generator(SwitchNode)
def SwitchNode_generate_code(self_node: Node, indent_level: int = 0, generate_pass: bool = False) -> str:
    assert isinstance(self_node, SwitchNode)

    if len(self_node.cases) == 0:
        return f'{indent(indent_level)}if {Query_format(self_node.query, self_node.params, False)}:\n' + \
                node_generate_code(NoopNode(''), indent_level + 1, True)

    hint_s = ''.join(f'{indent(indent_level)}# {hint}\n' for hint in self_node.query.hint(self_node.params))
    if self_node.query.rv == BoolType:
        if len(self_node.cases) == 1:
            values = [*self_node.cases.values()][0]
            if len(values) == 2:
                # true/false branch are identical, no branch needed
                return node_generate_code(self_node.out_edges[0], indent_level)
            else:
                # if (not) X, else return -> invert unless goto
                assert self_node.terminal_node is not None

                if isinstance(self_node.terminal_node, NoopNode):
                    return hint_s + f'{indent(indent_level)}if {Query_format(self_node.query, self_node.params, not values[0])}:\n' + \
                            node_generate_code(self_node.out_edges[0], indent_level + 1, True)
                else:
                    not_s = '' if not values[0] else 'not '
                    return hint_s + f'{indent(indent_level)}if {not_s}{Query_format(self_node.query, self_node.params, values[0])}):\n' + \
                            node_generate_code(self_node.terminal_node, indent_level + 1, True) + \
                            node_generate_code(self_node.out_edges[0], indent_level)
        else:
            # T/F explicitly spelled out
            true_node: Node
            false_node: Node

            if self_node.cases[self_node.out_edges[0].name][0]:
                true_node, false_node = self_node.out_edges
            else:
                false_node, true_node = self_node.out_edges
            return hint_s + f'{indent(indent_level)}if {Query_format(self_node.query, self_node.params, False)}:\n' + \
                    node_generate_code(true_node, indent_level + 1, True) + \
                    f'{indent(indent_level)}else:\n' + \
                    node_generate_code(false_node, indent_level + 1, True)
    elif len(self_node.cases) == 1:
        # if [query] in (...), if [query] = X ... else return -> negate and return unless goto
        values = [*self_node.cases.values()][0]
        if len(values) == self_node.query.num_values:
            # all branches identical, no branch needd
            return node_generate_code(self_node.out_edges[0], indent_level)

        assert self_node.terminal_node is not None

        try:
            if isinstance(self_node.terminal_node, NoopNode):
                op = f'== {Type_format(self_node.query.rv, values[0])}' if len(values) == 1 else 'in (' + ', '.join(Type_format(self_node.query.rv, v) for v in values) + ')'

                return hint_s + f'{indent(indent_level)}if {Query_format(self_node.query, self_node.params, False)} {op}:\n' + \
                        node_generate_code(self_node.out_edges[0], indent_level + 1, True)
            else:
                not_op = f'!= {Type_format(self_node.query.rv, values[0])}' if len(values) == 1 else 'not in (' + ', '.join(Type_format(self_node.query.rv, v) for v in values) + ')'

                return hint_s + f'{indent(indent_level)}if {Query_format(self_node.query, self_node.params, False)} {not_op}:\n' + \
                        node_generate_code(self_node.terminal_node, indent_level + 1, True) + \
                        node_generate_code(self_node.out_edges[0], indent_level)
        except:
            print(self_node.query, values)
            raise
    else:
        # generic case:
        # switch [query]:
        #   case X, Y:
        #   default - return

        cases: List[str] = []
        for event, values in sorted(self_node.cases.items(), key=lambda x: min(x[1])):
            try:
                cases.append(
                        f'{indent(indent_level + 1)}case {", ".join(str(v) for v in values)}:\n' +
                        node_generate_code([e for e in self_node.out_edges if e.name == event][0], indent_level + 2, True)
                )
            except:
                print(self_node.query, self_node.cases.values())
                raise

        default = ''
        if sum(len(v) for v in self_node.cases.values()) < self_node.query.num_values:
            default = f'{indent(indent_level + 1)}default:\n{indent(indent_level + 2)}return\n'
        return hint_s + f'{indent(indent_level)}switch {Query_format(self_node.query, self_node.params, False)}\n' + \
                ''.join(cases) + default

@node_generator(ForkNode)
def ForkNode_generate_code(self_node: Node, indent_level: int = 0, generate_pass: bool = False) -> str:
    assert isinstance(self_node, ForkNode)

    s = f'{indent(indent_level)}fork:\n'
    for i, e in enumerate(e for e in self_node.out_edges if e != self_node.join_node):
        s += f'{indent(indent_level + 1)}branch{i}:\n{node_generate_code(e, indent_level + 2, True)}'
    return s + node_generate_code(self_node.join_node, indent_level)

@node_generator(JoinNode)
def JoinNode_generate_code(self_node: Node, indent_level: int = 0, generate_pass: bool = False) -> str:
    assert isinstance(self_node, JoinNode)

    return "\n".join(node_generate_code(e, indent_level) for e in self_node.out_edges)

@node_generator(SubflowNode)
def SubflowNode_generate_code(self_node: Node, indent_level: int = 0, generate_pass: bool = False) -> str:
    assert isinstance(self_node, SubflowNode)

    ns = f'{self_node.ns}::' if self_node.ns else ''
    # todo: type-checked subflow params
    # todo: hints
    param_s = ', '.join(f'{name}={Type_format(AnyType, value)}' for name, value in self_node.params.items())
    return f'{indent(indent_level)}run {ns}{self_node.called_root_name}({param_s})\n' + \
            '\n'.join(node_generate_code(e, indent_level) for e in self_node.out_edges)

@node_generator(TerminalNode)
def TerminalNode_generate_code(self_node: Node, indent_level: int = 0, generate_pass: bool = False) -> str:
    assert isinstance(self_node, TerminalNode)

    return f'{indent(indent_level)}return\n'

@node_generator(DeadendTerminalNode)
def DeadendTerminalNode_generate_code(self_node: Node, indent_level: int = 0, generate_pass: bool = False) -> str:
    assert isinstance(self_node, DeadendTerminalNode)

    return ''

@node_generator(NoopNode)
def NoopNode_generate_code(self_node: Node, indent_level: int = 0, generate_pass: bool = False) -> str:
    assert isinstance(self_node, NoopNode)

    if generate_pass:
        return f'{indent(indent_level)}pass\n'
    else:
        return ''

@node_generator(EntryPointNode)
def EntryPointNode_generate_code(self_node: Node, indent_level: int = 0, generate_pass: bool = False) -> str:
    assert isinstance(self_node, EntryPointNode)

    return f'{indent(0)}entrypoint {self_node.entry_label}:\n' + \
            '\n'.join(node_generate_code(e, indent_level) for e in self_node.out_edges)

@node_generator(GroupNode)
def GroupNode_generate_code(self_node: Node, indent_level: int = 0, generate_pass: bool = False) -> str:
    assert isinstance(self_node, GroupNode)

    return node_generate_code(self_node.root, indent_level) + \
            '\n'.join(node_generate_code(e, indent_level) for e in self_node.out_edges)

@node_generator(IfElseNode)
def IfElseNode_generate_code(self_node: Node, indent_level: int = 0, generate_pass: bool = False) -> str:
    assert isinstance(self_node, IfElseNode)

    code = ''
    for predicate, node in self_node.rules:
        hint_s = ''.join(f'{indent(indent_level)}# {hint}\n' for hint in predicate.hint())
        el_s = 'el' if code else ''
        code += hint_s
        code += indent(indent_level) + f'{el_s}if {pred_generate_code(predicate)}:\n' + \
                node_generate_code(node, indent_level + 1, True)
    if not isinstance(self_node.default, NoopNode):
        code += f'{indent(indent_level)}else:\n' + node_generate_code(self_node.default, indent_level + 1, True)
    return code

@node_generator(WhileNode)
def WhileNode_generate_code(self_node: Node, indent_level: int = 0, generate_pass: bool = False) -> str:
    assert isinstance(self_node, WhileNode)

    hint_s = ''.join(f'{indent(indent_level)}# {hint}\n' for hint in self_node.loop_cond.hint())
    code = hint_s
    code += f'{indent(indent_level)}while {pred_generate_code(self_node.loop_cond)}:\n'
    code += node_generate_code(self_node.loop_body, indent_level + 1, True)
    code += node_generate_code(self_node.loop_exit, indent_level)
    return code

@node_generator(DoWhileNode)
def DoWhileNode_generate_code(self_node: Node, indent_level: int = 0, generate_pass: bool = False) -> str:
    assert isinstance(self_node, DoWhileNode)

    hint_s = ''.join(f'{indent(indent_level)}# {hint}\n' for hint in self_node.loop_cond.hint())
    code = ''
    code += f'{indent(indent_level)}do:\n'
    code += node_generate_code(self_node.loop_body, indent_level + 1, True)
    code += hint_s
    code += f'{indent(indent_level)}while {pred_generate_code(self_node.loop_cond)}\n'
    code += node_generate_code(self_node.loop_exit, indent_level)
    return code

@pred_generator(ConstPredicate)
def ConstPredicate_generate_code(self_pred: Predicate) -> str:
    assert isinstance(self_pred, ConstPredicate)

    return Type_format(BoolType, self_pred.value)

@pred_generator(QueryPredicate)
def QueryPredicate_generate_code(self_pred: Predicate) -> str:
    assert isinstance(self_pred, QueryPredicate)

    if self_pred.query.rv == BoolType:
        if len(self_pred.values) == 1:
            return Query_format(self_pred.query, self_pred.params, self_pred.negated)
        else:
            return 'False' if self_pred.negated else 'True'
    else:
        if len(self_pred.values) == 1:
            op = '!=' if self_pred.negated else '=='
            try:
                return f'{Query_format(self_pred.query, self_pred.params, False)} {op} {Type_format(self_pred.query.rv, self_pred.values[0])}'
            except:
                print(self_pred.query, self_pred.values)
                raise
        else:
            op = 'not in' if self_pred.negated else 'in'
            try:
                vals_s = [Type_format(self_pred.query.rv, v) for v in self_pred.values]
            except:
                print(self_pred.query, self_pred.values)
                raise
            return f'{Query_format(self_pred.query, self_pred.params, False)} {op} ({", ".join(vals_s)})'

@pred_generator(NotPredicate)
def NotPredicate_generate_code(self_pred: Predicate) -> str:
    assert isinstance(self_pred, NotPredicate)

    return f'not ({pred_generate_code(self_pred.inner)})'

@pred_generator(AndPredicate)
def AndPredicate_generate_code(self_pred: Predicate) -> str:
    assert isinstance(self_pred, AndPredicate)

    return ' and '.join([f'({pred_generate_code(inner)})' for inner in self_pred.inners])

@pred_generator(OrPredicate)
def OrPredicate_generate_code(self_pred: Predicate) -> str:
    assert isinstance(self_pred, OrPredicate)

    return ' or '.join([f'({pred_generate_code(inner)})' for inner in self_pred.inners])

def Action_format(action: Action, params: Dict[str, Any]) -> str:
    conversion = action.conversion.replace('<.name>', f'{action.actor_name}.{action.name}')
    conversion = conversion.replace('<.actor>', f'{action.actor_name}')
    for p in action.params:
        try:
            assert p.name in params
            # some places do this with a string value instead of an Argument value
            if p.name == f'EntryVariableKeyInt_{params[p.name]}' or \
                p.name == f'EntryVariableKeyBool_{params[p.name]}' or \
                p.name == f'EntryVariableKeyFloat_{params[p.name]}':
                value = params[p.name]
            else:
                value = Type_format(p.type, params[p.name])
        except:
            print(action, p, params)
            raise
        conversion = conversion.replace(f'<<{p.name}>>', str(params[p.name]))
        conversion = conversion.replace(f'<{p.name}>', value)
    return conversion

def Query_format(query: Query, params: Dict[str, Any], negated: bool) -> str:
    if negated:
        conversion_used = query.neg_conversion
    else:
        conversion_used = query.conversion
    if not isinstance(conversion_used, str):
        pivot = params[conversion_used['key']]
        conversion = conversion_used['values'][pivot]
    else:
        conversion = conversion_used
    conversion = conversion.replace('<.name>', f'{query.actor_name}.{query.name}')
    conversion = conversion.replace('<.actor>', f'{query.actor_name}')
    for p in query.params:
        try:
            assert p.name in params
            # some places do this with a string value instead of an Argument value
            if p.name == f'EntryVariableKeyInt_{params[p.name]}' or \
                p.name == f'EntryVariableKeyBool_{params[p.name]}' or \
                p.name == f'EntryVariableKeyFloat_{params[p.name]}':
                value = params[p.name]
            else:
                value = Type_format(p.type, params[p.name])
        except:
            print(query, p, params)
            raise
        conversion = conversion.replace(f'<<{p.name}>>', str(params[p.name]))
        conversion = conversion.replace(f'<{p.name}>', value)
    return conversion

def Type_format(type_: Type, value: Any) -> str:
    if isinstance(value, Argument):
        return str(value)

    if type_.type.startswith('int'):
        assert isinstance(value, int)
        if type_.type != 'int':
            n = int(type_.type[3:])
            assert 0 <= value < n
        return repr(int(value))
    elif type_.type.startswith('enum'):
        assert not isinstance(value, bool) and isinstance(value, int)
        vals = type_.type[5:-1].split(',')
        assert 0 <= value < len(vals)
        return vals[value]
    elif type_.type == 'float':
        assert not isinstance(value, bool) and isinstance(value, (int, float))
        return repr(float(value))
    elif type_.type == 'str':
        assert isinstance(value, str)
        return repr(value)
    elif type_.type == 'bool':
        assert isinstance(value, bool) or (isinstance(value, int) and 0 <= value <= 1)
        return 'true' if value else 'false'
    elif type_.type == 'any':
        return repr(value)
    else:
        raise ValueError(f'bad type: {type_.type}')
