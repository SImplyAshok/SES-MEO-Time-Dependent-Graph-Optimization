import pulp
from .config import SOURCE,SATELLITES,GATEWAYS,TIME,DT,HANDOVER_COST

def route(k,E,x):
    selected=[(u,v) for u,v in E[k] if pulp.value(x[(u,v,k)])>.5]
    succ={}
    for u,v in selected:
        if u in succ: raise RuntimeError(f"Multiple outgoing edges from {u} at k={k}")
        succ[u]=v
    r=[SOURCE]; cur=SOURCE; seen=set()
    while cur not in GATEWAYS:
        if cur in seen: raise RuntimeError(f"Cycle detected at k={k}")
        seen.add(cur)
        if cur not in succ: raise RuntimeError(f"Route terminated at {cur}, k={k}")
        cur=succ[cur]; r.append(cur)
    return r

def extract(E,L,x,h):
    serving={}; routes={}; costs={}; hand={}
    for k in TIME:
        ss=[s for s in SATELLITES if (SOURCE,s,k) in x and pulp.value(x[(SOURCE,s,k)])>.5]
        if len(ss)!=1: raise RuntimeError(f"Invalid serving satellite at k={k}: {ss}")
        serving[k]=ss[0]
        routes[k]=route(k,E,x)
        costs[k]=sum(L[(u,v,k)] for u,v in zip(routes[k][:-1],routes[k][1:]))
    for k in range(1,len(TIME)): hand[k]=int(round(pulp.value(h[k])))
    return serving,routes,costs,hand

def report(serving,routes,costs,hand):
    lines=["="*72,"OPTIMAL ONE-SHOT TEMPORAL MEO SOLUTION","="*72]
    for k in TIME:
        ho=0 if k==0 else hand[k]
        lines += [f"\nt = {k*DT:4d} s",
                  f"  Serving satellite : {serving[k]}",
                  f"  Route             : {' -> '.join(routes[k])}",
                  f"  Routing cost      : {costs[k]:.4f}",
                  f"  Handover          : {ho}"]
    tr=sum(costs.values()); th=HANDOVER_COST*sum(hand.values())
    lines += ["","="*72,"COST BREAKDOWN","="*72,
              f"Total routing cost  = {tr:.4f}",
              f"Total handover cost = {th:.4f}",
              f"Total objective     = {tr+th:.4f}",
              f"Number of handovers = {sum(hand.values())}",
              "","Serving sequence:",
              " -> ".join(serving[k] for k in TIME)]
    return "\n".join(lines)
