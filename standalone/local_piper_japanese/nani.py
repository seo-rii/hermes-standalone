"""Execute the original Japanese OneHotEncoder and 200-tree Nani model."""
from pathlib import Path
import struct
from standalone.local_piper_model import message, read_attribute, read_tensor

X_COLS = ('pos', 'pos_group1', 'pos_group2', 'pron', 'ctype', 'cform')
_FLOAT = struct.Struct('<f')


def _f32(value):
    return _FLOAT.unpack(_FLOAT.pack(value))[0]


def _attributes(node):
    output = {}
    for _, data in node.get(5, []):
        value = message(data)
        if value[20][0][1] == 8:
            output[value[1][0][1].decode()] = [text.decode() for _, text in value.get(9, [])]
        else:
            name, value = read_attribute(data)
            output[name] = value
    return output


def _graph(path):
    model = message(Path(path).read_bytes())
    if len(model.get(7, [])) != 1:
        raise ValueError('Japanese reading model requires a single graph')
    graph = message(model[7][0][1])
    nodes = []
    for _, raw in graph.get(1, []):
        node = message(raw)
        nodes.append({'op': node[4][0][1].decode(), 'inputs': [v.decode() for _, v in node.get(1, [])],
                      'outputs': [v.decode() for _, v in node.get(2, [])], 'attributes': _attributes(node)})
    weights = dict(read_tensor(value) for _, value in graph.get(5, []))
    return nodes, weights


class NaniModel:
    def __init__(self, directory):
        directory = Path(directory)
        self.categories = self._encoder(directory / 'nani_enc.onnx')
        self.width = sum(len(categories) for _, categories in self.categories)
        self.trees = self._forest(directory / 'nani_model.onnx')

    @staticmethod
    def _encoder(path):
        nodes, initializers = _graph(path)
        if len(nodes) != 14 or [node['op'] for node in nodes] != ['Gather'] * 6 + ['OneHotEncoder'] * 6 + ['Concat', 'Reshape']:
            raise ValueError('Unsupported Japanese categorical encoder graph')
        result = []
        for gather, encoder in zip(nodes[:6], nodes[6:12]):
            if gather['attributes'] != {'axis': 1} or gather['outputs'] != encoder['inputs']:
                raise ValueError('Unsupported Japanese encoder column mapping')
            index = initializers[gather['inputs'][1]].values
            if len(index) != 1 or not 0 <= index[0] < 6 or encoder['attributes'].get('zeros') != 1:
                raise ValueError('Unsupported Japanese encoder category policy')
            result.append((int(index[0]), tuple(encoder['attributes']['cats_strings'])))
        if nodes[12]['inputs'] != [node['outputs'][0] for node in nodes[6:12]] or nodes[12]['attributes'].get('axis') not in (-1, 2):
            raise ValueError('Unsupported Japanese encoder concatenation')
        shape = initializers[nodes[13]['inputs'][1]].values
        if tuple(shape) != (-1, sum(len(categories) for _, categories in result)):
            raise ValueError('Unsupported Japanese encoder output reshape')
        return tuple(result)

    @staticmethod
    def _forest(path):
        graph, _ = _graph(path)
        if len(graph) != 1 or graph[0]['op'] != 'TreeEnsembleClassifier':
            raise ValueError('Unsupported Japanese reading model graph')
        attributes = graph[0]['attributes']
        if attributes['classlabels_int64s'] != [0, 1] or attributes['post_transform'] != 'NONE':
            raise ValueError('Unsupported Japanese reading classifier output')
        trees = {}
        names = ('nodes_treeids', 'nodes_nodeids', 'nodes_featureids', 'nodes_values', 'nodes_truenodeids', 'nodes_falsenodeids', 'nodes_modes', 'nodes_missing_value_tracks_true')
        for tree, node, feature, threshold, yes, no, mode, missing in zip(*(attributes[name] for name in names)):
            if mode not in ('LEAF', 'BRANCH_LEQ') or missing:
                raise ValueError('Unsupported Japanese reading tree branch')
            trees.setdefault(tree, {})[node] = (feature, threshold, yes, no, mode)
        leaves = {}
        names = ('class_treeids', 'class_nodeids', 'class_ids', 'class_weights')
        for tree, node, kind, weight in zip(*(attributes[name] for name in names)):
            leaves.setdefault((tree, node), [0.0, 0.0])[kind] = weight
        return tuple((tree, nodes, leaves) for tree, nodes in sorted(trees.items()))

    def encode(self, features):
        if len(features) != 6 or any(not isinstance(value, str) for value in features):
            raise ValueError('Japanese reading model requires six categorical strings')
        return [float(features[index] == category) for index, categories in self.categories for category in categories]

    def probabilities(self, encoded):
        if len(encoded) != self.width:
            raise ValueError('Japanese reading model has an unexpected feature width')
        scores = [0.0, 0.0]
        for tree, nodes, leaves in self.trees:
            node = 0
            for _ in range(len(nodes) + 1):
                feature, threshold, yes, no, mode = nodes[node]
                if mode == 'LEAF':
                    break
                node = yes if encoded[feature] <= threshold else no
            else:
                raise ValueError('Invalid cyclic Japanese reading tree')
            first, second = leaves[(tree, node)]
            scores[0] = _f32(scores[0] + first)
            scores[1] = _f32(scores[1] + second)
        return scores

    def predict(self, following):
        if following is None:
            return 0
        scores = self.probabilities(self.encode([following[column] for column in X_COLS]))
        return int(scores[1] > scores[0])
