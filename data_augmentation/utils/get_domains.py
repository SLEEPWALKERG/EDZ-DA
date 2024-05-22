def count_domains(belief_state):
    domains = set()
    for ds, v in belief_state.items():
        d, s = ds.split('-')
        domains.add(d)
    return list(domains)
