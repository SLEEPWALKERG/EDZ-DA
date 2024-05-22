from collections import defaultdict, deque

def topological_sort(graph):
    in_degree = defaultdict(int)
    for node in graph:
        for neighbor in graph[node]:
            in_degree[neighbor] += 1

    queue = deque([node for node in graph if in_degree[node] == 0])
    result = []

    while queue:
        current_node = queue.popleft()
        result.append(current_node)

        for neighbor in graph[current_node]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    if len(result) == len(graph):
        return result
    else:
        # Graph contains a cycle, no topological order exists
        return None

def determine_fill_order(inference_relations):
    domain_graph = defaultdict(list)

    # Construct the domain graph
    for target, source in inference_relations.items():
        source_domain = source.split('-')[0]
        target_domain = target.split('-')[0]
        if source_domain != target_domain:
            domain_graph[source_domain].append(target_domain)

    # Perform topological sort
    domain_fill_order = topological_sort(domain_graph)

    return domain_fill_order

# 示例用法
# inference_relations = {
#     'domain1-slot1': 'domain2-slot1',
#     'domain1-slot2': 'domain2-slot2',
#     'domain2-slot1': 'domain3-slot1',
#     'domain2-slot2': 'domain3-slot2',
# }
#
# domain_fill_order = determine_fill_order(inference_relations)
# print("Domain 的填充顺序:", domain_fill_order)
