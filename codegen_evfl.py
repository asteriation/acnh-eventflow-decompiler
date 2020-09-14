from __future__ import annotations

from typing import Callable, Dict, List

from codegen import CodeGenerator
from datatype import AnyType, BoolType
from indent import indent
from nodes import *

NodeCodeGenerator = Callable[[Node, int, bool], str]

codegen: Dict[type, NodeCodeGenerator] = {}

def generator(typ: type) -> Callable[[NodeCodeGenerator], NodeCodeGenerator]:
    def inner(f: NodeCodeGenerator) -> NodeCodeGenerator:
        codegen[typ] = f
        return f
    return inner

def generate_code(node: Node, indent_level: int = 0, generate_pass: bool = False) -> str:
    return codegen[type(node)](node, indent_level, generate_pass)

class EVFLCodeGenerator(CodeGenerator):
    def generate_code(self, node: Node, indent_level: int = 0, generate_pass: bool = False) -> str:
        return generate_code(node, indent_level, generate_pass)

@generator(RootNode)
def RootNode_generate_code(self_node: Node, indent_level: int = 0, generate_pass: bool = False) -> str:
    assert isinstance(self_node, RootNode)

    return f'{indent(indent_level)}flow {self_node.name}({", ".join(str(v) for v in self_node.vardefs)}):\n' + \
            '\n'.join(generate_code(n, indent_level + 1) for n in self_node.out_edges)

@generator(ActionNode)
def ActionNode_generate_code(self_node: Node, indent_level: int = 0, generate_pass: bool = False) -> str:
    assert isinstance(self_node, ActionNode)

    hint_s = ''.join(f'{indent(indent_level)}# {hint}\n' for hint in self_node.action.hint(self_node.params))
    return hint_s + f'{indent(indent_level)}{self_node.action.format(self_node.params)}\n' + \
            '\n'.join(generate_code(n, indent_level) for n in self_node.out_edges)

@generator(SwitchNode)
def SwitchNode_generate_code(self_node: Node, indent_level: int = 0, generate_pass: bool = False) -> str:
    assert isinstance(self_node, SwitchNode)

    if len(self_node.cases) == 0:
        return f'{indent(indent_level)}if {self_node.query.format(self_node.params, False)}:\n' + \
                generate_code(NoopNode(''), indent_level + 1, True)

    hint_s = ''.join(f'{indent(indent_level)}# {hint}\n' for hint in self_node.query.hint(self_node.params))
    if self_node.query.rv == BoolType:
        if len(self_node.cases) == 1:
            values = [*self_node.cases.values()][0]
            if len(values) == 2:
                # true/false branch are identical, no branch needed
                return generate_code(self_node.out_edges[0], indent_level)
            else:
                # if (not) X, else return -> invert unless goto
                assert self_node.terminal_node is not None

                if isinstance(self_node.terminal_node, NoopNode):
                    return hint_s + f'{indent(indent_level)}if {self_node.query.format(self_node.params, not values[0])}:\n' + \
                            generate_code(self_node.out_edges[0], indent_level + 1, True)
                else:
                    not_s = '' if not values[0] else 'not '
                    return hint_s + f'{indent(indent_level)}if {not_s}{self_node.query.format(self_node.params, values[0])}):\n' + \
                            generate_code(self_node.terminal_node, indent_level + 1, True) + \
                            generate_code(self_node.out_edges[0], indent_level)
        else:
            # T/F explicitly spelled out
            true_node: Node
            false_node: Node

            if self_node.cases[self_node.out_edges[0].name][0]:
                true_node, false_node = self_node.out_edges
            else:
                false_node, true_node = self_node.out_edges
            return hint_s + f'{indent(indent_level)}if {self_node.query.format(self_node.params, False)}:\n' + \
                    generate_code(true_node, indent_level + 1, True) + \
                    f'{indent(indent_level)}else:\n' + \
                    generate_code(false_node, indent_level + 1, True)
    elif len(self_node.cases) == 1:
        # if [query] in (...), if [query] = X ... else return -> negate and return unless goto
        values = [*self_node.cases.values()][0]
        if len(values) == self_node.query.num_values:
            # all branches identical, no branch needd
            return generate_code(self_node.out_edges[0], indent_level)

        assert self_node.terminal_node is not None

        try:
            if isinstance(self_node.terminal_node, NoopNode):
                op = f'== {self_node.query.rv.format(values[0])}' if len(values) == 1 else 'in (' + ', '.join(self_node.query.rv.format(v) for v in values) + ')'

                return hint_s + f'{indent(indent_level)}if {self_node.query.format(self_node.params, False)} {op}:\n' + \
                        generate_code(self_node.out_edges[0], indent_level + 1, True)
            else:
                not_op = f'!= {self_node.query.rv.format(values[0])}' if len(values) == 1 else 'not in (' + ', '.join(self_node.query.rv.format(v) for v in values) + ')'

                return hint_s + f'{indent(indent_level)}if {self_node.query.format(self_node.params, False)} {not_op}:\n' + \
                        generate_code(self_node.terminal_node, indent_level + 1, True) + \
                        generate_code(self_node.out_edges[0], indent_level)
        except:
            print(self_node.query, values)
            raise
    else:
        # generic case:
        # f{name} = [query]
        # if/elif checks on f{name}, else implicit
        vname = f'f{self_node.name}'

        cases: List[str] = []
        for event, values in sorted(self_node.cases.items(), key=lambda x: min(x[1])):
            try:
                else_s = 'el' if cases else ''
                op_s = f'== {self_node.query.rv.format(values[0])}' if len(values) == 1 else 'in (' + ', '.join(self_node.query.rv.format(v)  for v in sorted(values)) + ')'
                cases.append(
                        f'{indent(indent_level)}{else_s}if {vname} {op_s}:\n' +
                        generate_code([e for e in self_node.out_edges if e.name == event][0], indent_level + 1, True)
                )
            except:
                print(self_node.query, self_node.cases.values())
                raise

        return hint_s + f'{indent(indent_level)}{vname} = {self_node.query.format(self_node.params, False)}\n' + ''.join(cases)

@generator(ForkNode)
def ForkNode_generate_code(self_node: Node, indent_level: int = 0, generate_pass: bool = False) -> str:
    assert isinstance(self_node, ForkNode)

    s = f'{indent(indent_level)}fork:\n'
    for i, e in enumerate(e for e in self_node.out_edges if e != self_node.join_node):
        s += f'{indent(indent_level + 1)}branch{i}:\n{generate_code(e, indent_level + 2, True)}'
    return s + generate_code(self_node.join_node, indent_level)

@generator(JoinNode)
def JoinNode_generate_code(self_node: Node, indent_level: int = 0, generate_pass: bool = False) -> str:
    assert isinstance(self_node, JoinNode)

    return "\n".join(generate_code(e, indent_level) for e in self_node.out_edges)

@generator(SubflowNode)
def SubflowNode_generate_code(self_node: Node, indent_level: int = 0, generate_pass: bool = False) -> str:
    assert isinstance(self_node, SubflowNode)

    ns = f'{self_node.ns}::' if self_node.ns else ''
    # todo: type-checked subflow params
    # todo: hints
    param_s = ', '.join(f'{name}={AnyType.format(value)}' for name, value in self_node.params.items())
    return f'{indent(indent_level)}run {ns}{self_node.called_root_name}({param_s})\n' + '\n'.join(generate_code(e, indent_level) for e in self_node.out_edges)

@generator(TerminalNode)
def TerminalNode_generate_code(self_node: Node, indent_level: int = 0, generate_pass: bool = False) -> str:
    assert isinstance(self_node, TerminalNode)

    return f'{indent(indent_level)}return\n'

@generator(DeadendTerminalNode)
def DeadendTerminalNode_generate_code(self_node: Node, indent_level: int = 0, generate_pass: bool = False) -> str:
    assert isinstance(self_node, DeadendTerminalNode)

    return ''

@generator(NoopNode)
def NoopNode_generate_code(self_node: Node, indent_level: int = 0, generate_pass: bool = False) -> str:
    assert isinstance(self_node, NoopNode)

    if generate_pass:
        return f'{indent(indent_level)}pass\n'
    else:
        return ''

@generator(EntryPointNode)
def EntryPointNode_generate_code(self_node: Node, indent_level: int = 0, generate_pass: bool = False) -> str:
    assert isinstance(self_node, EntryPointNode)

    return f'{indent(0)}entrypoint {self_node.entry_label}:\n' + '\n'.join(generate_code(e, indent_level) for e in self_node.out_edges)

@generator(GroupNode)
def GroupNode_generate_code(self_node: Node, indent_level: int = 0, generate_pass: bool = False) -> str:
    assert isinstance(self_node, GroupNode)

    return generate_code(self_node.root, indent_level) + \
            '\n'.join(generate_code(e, indent_level) for e in self_node.out_edges)

@generator(IfElseNode)
def IfElseNode_generate_code(self_node: Node, indent_level: int = 0, generate_pass: bool = False) -> str:
    assert isinstance(self_node, IfElseNode)

    code = ''
    for predicate, node in self_node.rules:
        hint_s = ''.join(f'{indent(indent_level)}# {hint}\n' for hint in predicate.hint())
        el_s = 'el' if code else ''
        code += hint_s
        code += indent(indent_level) + f'{el_s}if {predicate.generate_code()}:\n' + \
                generate_code(node, indent_level + 1, True)
    if not isinstance(self_node.default, NoopNode):
        code += f'{indent(indent_level)}else:\n' + generate_code(self_node.default, indent_level + 1, True)
    return code

@generator(WhileNode)
def WhileNode_generate_code(self_node: Node, indent_level: int = 0, generate_pass: bool = False) -> str:
    assert isinstance(self_node, WhileNode)

    hint_s = ''.join(f'{indent(indent_level)}# {hint}\n' for hint in self_node.loop_cond.hint())
    code = hint_s
    code += f'{indent(indent_level)}while {self_node.loop_cond.generate_code()}:\n'
    code += generate_code(self_node.loop_body, indent_level + 1, True)
    code += generate_code(self_node.loop_exit, indent_level)
    return code

@generator(DoWhileNode)
def DoWhileNode_generate_code(self_node: Node, indent_level: int = 0, generate_pass: bool = False) -> str:
    assert isinstance(self_node, DoWhileNode)

    hint_s = ''.join(f'{indent(indent_level)}# {hint}\n' for hint in self_node.loop_cond.hint())
    code = ''
    code += f'{indent(indent_level)}do:\n'
    code += generate_code(self_node.loop_body, indent_level + 1, True)
    code += hint_s
    code += f'{indent(indent_level)}while {self_node.loop_cond.generate_code()}\n'
    code += generate_code(self_node.loop_exit, indent_level)
    return code

