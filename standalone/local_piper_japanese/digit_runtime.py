"""Execute only the serialized source operations of OpenJTalk's digit phase."""
from .digit_memory import BINARY, CArray, CString, Ref, TypeSize, dereference, size_of
from .digit_nodes import NumberSequence, builtins
from .njd import NJDNode


class Scope:
    def __init__(self, parent=None):
        self.values = {}
        self.parent = parent

    def reference(self, name):
        if name in self.values:
            return Ref(self.values, name)
        if self.parent:
            return self.parent.reference(name)
        raise NameError(name)


class Flow(Exception):
    def __init__(self, kind, value=None):
        self.kind, self.value = kind, value


class Program:
    def __init__(self, definitions, globals_, functions):
        self.functions = functions
        self.global_scope = Scope()
        self.global_scope.values.update(builtins())
        self.literals = {}
        self.trace = None
        self.call_stack = []
        self._expressions = {
            'i': lambda node, scope: node[1],
            's': self.string, 'n': self.name, 'type': self.type,
            'bin': self.binary, 'unary': self.unary, 'post': self.postfix,
            'member': self.read_reference, 'index': self.read_reference,
            'call': self.call_expression, 'cond': self.conditional,
            'sizeof': self.sizeof, 'init': self.initializer,
        }
        self._statements = {
            'expr': self.expression_statement, 'decl': self.declaration,
            'block': self.block, 'if': self.if_statement,
            'for': self.for_statement, 'while': self.while_statement,
            'return': self.return_statement,
            'break': self.flow_statement, 'continue': self.flow_statement,
        }
        for name, value in definitions.items():
            self.global_scope.values[name] = None if name == 'NULL' else self.evaluate(value, self.global_scope)
        for statement in globals_:
            self.execute(statement, self.global_scope)

    def evaluate(self, node, scope):
        if node is None:
            return None
        return self._expressions[node[0]](node, scope)

    def execute(self, statement, scope):
        if statement is not None:
            return self._statements[statement[0]](statement, scope)

    def string(self, node, scope):
        value = node[1]
        if value not in self.literals:
            self.literals[value] = CString(value.encode())
        return self.literals[value]

    def name(self, node, scope):
        return scope.reference(node[1]).get()

    def type(self, node, scope):
        return TypeSize(node[1])

    def sizeof(self, node, scope):
        return size_of(self.evaluate(node[1], scope))

    def initializer(self, node, scope):
        return [self.evaluate(value, scope) for value in node[1]]

    def reference(self, node, scope):
        handlers = {
            'n': lambda: scope.reference(node[1]),
            'member': lambda: Ref(dereference(self.evaluate(node[1], scope)), node[2], True),
            'index': lambda: Ref(self.evaluate(node[1], scope), self.evaluate(node[2], scope)),
            'unary': lambda: self.evaluate(node[2], scope),
        }
        result = handlers[node[0]]()
        if not isinstance(result, Ref):
            raise TypeError(('not a reference', node, result))
        return result

    def read_reference(self, node, scope):
        return self.reference(node, scope).get()

    def conditional(self, node, scope):
        return self.evaluate(node[2] if self.evaluate(node[1], scope) else node[3], scope)

    def binary(self, node, scope):
        operation, left, right = node[1:]
        special = {
            '=': self.assign, '+=': self.assign, '-=': self.assign,
            '*=': self.assign, '/=': self.assign, '%=': self.assign, '|=': self.assign,
            ',': self.comma, '&&': self.logical_and, '||': self.logical_or,
        }.get(operation)
        if special:
            return special(node, scope)
        return BINARY[operation](self.evaluate(left, scope), self.evaluate(right, scope))

    def assign(self, node, scope):
        reference = self.reference(node[2], scope)
        value = self.evaluate(node[3], scope)
        if node[1] != '=':
            value = BINARY[node[1][:-1]](reference.get(), value)
        return reference.set(value)

    def comma(self, node, scope):
        self.evaluate(node[2], scope)
        return self.evaluate(node[3], scope)

    def logical_and(self, node, scope):
        return int(bool(self.evaluate(node[2], scope)) and bool(self.evaluate(node[3], scope)))

    def logical_or(self, node, scope):
        return int(bool(self.evaluate(node[2], scope)) or bool(self.evaluate(node[3], scope)))

    def unary(self, node, scope):
        operation, value = node[1:]
        if operation in ('++', '--'):
            return self.increment(value, scope, 1 if operation == '++' else -1, False)
        if operation == '&':
            return self.reference(value, scope).address()
        result = self.evaluate(value, scope)
        handlers = {
            '!': lambda: int(not result), '~': lambda: ~result,
            '+': lambda: +result, '-': lambda: -result,
            '*': lambda: result.get() if isinstance(result, Ref) else result[0],
        }
        return handlers[operation]()

    def postfix(self, node, scope):
        return self.increment(node[2], scope, 1 if node[1] == '++' else -1, True)

    def increment(self, node, scope, amount, previous):
        reference = self.reference(node, scope)
        before = reference.get()
        after = before + amount
        reference.set(after)
        return before if previous else after

    def call_expression(self, node, scope):
        target = node[1]
        if target[0] != 'n':
            raise ValueError(('indirect C call absent from pinned source', target))
        name = target[1]
        arguments = [self.evaluate(argument, scope) for argument in node[2]]
        return self.call(name, arguments) if name in self.functions else scope.reference(name).get()(*arguments)

    def call(self, name, arguments):
        parameters, statement = self.functions[name]
        if len(parameters) != len(arguments):
            raise TypeError((name, len(parameters), len(arguments)))
        scope = Scope(self.global_scope)
        scope.values.update((parameter[1][0], value) for parameter, value in zip(parameters, arguments))
        self.call_stack.append(name)
        try:
            self.execute(statement, scope)
        except Flow as result:
            if result.kind != 'return':
                raise
            return result.value
        finally:
            self.call_stack.pop()

    def expression_statement(self, statement, scope):
        self.evaluate(statement[1], scope)

    def declaration(self, statement, scope):
        kind, declarations = statement[1:]
        for name, pointers, dimensions, initial in declarations:
            value = self.evaluate(initial, scope) if initial is not None else None
            scope.values[name] = self.declared_value(kind, pointers, dimensions, value, scope)

    def declared_value(self, kind, pointers, dimensions, value, scope):
        if dimensions:
            lengths = [self.evaluate(length, scope) for length in dimensions]
            if kind == 'char' and not pointers:
                return CString(bytearray(lengths[0]))
            return self.array(kind, pointers, lengths, value)
        if value is not None or pointers:
            return value
        constructors = {'NJDNode': NJDNode, 'NJDNumberSequence': NumberSequence, 'int': int, 'size_t': int, 'char': int}
        return constructors[kind]()

    def array(self, kind, pointers, lengths, value):
        length = lengths[0] if lengths[0] is not None else len(value)
        if len(lengths) > 1:
            values = [self.array(kind, pointers, lengths[1:], item) for item in (value or [None] * length)]
            item_size = len(values[0]) * values[0].item_size
        else:
            values = list(value) if value is not None else [None if pointers else 0] * length
            item_size = 8 if pointers else 4
        return CArray(values, item_size)

    def block(self, statement, scope):
        child = Scope(scope)
        for nested in statement[1]:
            self.execute(nested, child)

    def if_statement(self, statement, scope):
        selected = statement[2] if self.evaluate(statement[1], scope) else statement[3]
        self.execute(selected, scope)

    def for_statement(self, statement, scope):
        scope = Scope(scope)
        self.execute(statement[1], scope)
        while statement[2] is None or self.evaluate(statement[2], scope):
            if self.iteration(statement[4], scope):
                break
            self.evaluate(statement[3], scope)

    def while_statement(self, statement, scope):
        while self.evaluate(statement[1], scope):
            if self.iteration(statement[2], scope):
                break

    def iteration(self, statement, scope):
        try:
            self.execute(statement, scope)
        except Flow as result:
            if result.kind == 'break':
                return True
            if result.kind != 'continue':
                raise
        return False

    def return_statement(self, statement, scope):
        raise Flow('return', self.evaluate(statement[1], scope))

    def flow_statement(self, statement, scope):
        raise Flow(statement[0])
