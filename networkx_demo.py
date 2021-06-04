import networkx as nx

data = {
    'example.com': {
        'amazon.com': {
            'weight': 2,
        },
    },
    'wikipedia.org': {
        'amazon.com': {
            'weight': 5,
        },
        'example.com': {
            'weight': 1,
        },
    },
    'amazon.com': {
        'wikipedia.org': {
            'weight': 10,
        }
    }
}

graph = nx.DiGraph(data)
print(nx.pagerank(graph, personalization={'amazon.com': 1, 'wikipedia.org': 0.2}))
