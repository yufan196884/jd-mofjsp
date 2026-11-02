#!/usr/bin/env python3
"""CPU-only, deterministic two-objective heuristic for Step-2 MOFJSP instances."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
import random
import tempfile
from pathlib import Path

def intervals(x):
    p=x["energy_profile"]; H=x["horizon"]
    if isinstance(p,dict):
        w=p["interval_ticks"]
        if not isinstance(w,int) or isinstance(w,bool) or w<=0: raise ValueError("bad profile width")
        rows=[(i*w,min((i+1)*w,H),*pair) for i,pair in enumerate(p["price_solar"])]
    else:
        rows=[(a["start"],a["end"],a["price_ct_per_kwh"],a["solar_kw"]) for a in p]
    t=0
    for a,b,price,solar in rows:
        if a!=t or b<=a or b>H or not math.isfinite(price) or not math.isfinite(solar) or solar<0:
            raise ValueError("invalid energy profile")
        t=b
    if t!=H: raise ValueError("profile does not cover horizon")
    return rows

def check_instance(x):
    if x["format_version"] not in ("jd-mofjsp-step2-v1","jd-mofjsp-step2-v2"): raise ValueError("unsupported format")
    if x["ticks_per_hour"]!=100 or x["horizon"]<=0: raise ValueError("invalid time units/horizon")
    J=x["jobs"]; M=x["machines"]
    if not J or not M or len(J)!=x["num_jobs"] or len(M)!=x["num_machines"]: raise ValueError("invalid job/machine counts")
    if sum(len(j["operations"]) for j in J)!=x["num_operations"]: raise ValueError("invalid operation count")
    for m, machine in enumerate(M):
        if machine["machine"]!=m or not math.isfinite(machine["standby_kw"]) or machine["standby_kw"]<0:
            raise ValueError("invalid machine")
    for j,job in enumerate(J):
        if job["job"]!=j or not job["operations"]: raise ValueError("invalid job")
        for k,op in enumerate(job["operations"]):
            if op["operation"]!=k or not op["options"]: raise ValueError("invalid operation")
            seen=set()
            for o in op["options"]:
                m=o["machine"]
                if (not isinstance(m,int) or isinstance(m,bool) or m in seen or not 0<=m<len(M)
                    or not isinstance(o["duration"],int) or o["duration"]<=0
                    or not math.isfinite(o["active_kw"]) or o["active_kw"]<0):
                    raise ValueError("invalid option")
                seen.add(m)
    intervals(x)

def decode(x,actions,step=25,max_delay=2400):
    """Decode job dispatches in order, rejecting invalid/incomplete schedules."""
    jobs=x["jobs"]
    if len(actions)!=x["num_operations"]: raise ValueError("incorrect number of actions")
    cursor=[0]*len(jobs); ready_j=[0]*len(jobs); ready_m=[0]*len(x["machines"])
    schedule=[]
    for a in actions:
        if not isinstance(a,dict) or set(a)!= {"job","machine","delay"}: raise ValueError("bad action fields")
        j,m,d=a["job"],a["machine"],a["delay"]
        if type(j)!=int or not 0<=j<len(jobs) or cursor[j]>=len(jobs[j]["operations"]): raise ValueError("bad job")
        if type(m)!=int or type(d)!=int or d<0 or d>max_delay or d%step: raise ValueError("bad machine/delay")
        k=cursor[j]; opts={o["machine"]:o for o in jobs[j]["operations"][k]["options"]}
        if m not in opts: raise ValueError("incompatible machine")
        opt=opts[m]; start=max(ready_j[j],ready_m[m])+d; end=start+opt["duration"]
        if end>x["horizon"]: raise ValueError("exceeds horizon")
        schedule.append((j,k,m,start,end,opt["active_kw"]))
        ready_j[j]=ready_m[m]=end; cursor[j]+=1
    if any(cursor[j]!=len(jobs[j]["operations"]) for j in range(len(jobs))): raise ValueError("incomplete schedule")
    # Independent precedence and machine-nonoverlap checks.
    for idx in (0,2):
        for i in range(len(jobs) if idx==0 else len(x["machines"])):
            subset=[o for o in schedule if o[idx]==i]
            subset.sort(key=(lambda o:o[1]) if idx==0 else (lambda o:o[3]))
            if any(a[4]>b[3] for a,b in zip(subset,subset[1:])): raise ValueError("overlap/precedence")
    return schedule,max(o[4] for o in schedule)

def energy_cost(x,schedule,end,profile):
    """EUR: max(0, demand_kW - solar_kW)*ct/kWh*ticks / (100*ticks_per_hour)."""
    idle=sum(m["standby_kw"] for m in x["machines"])
    changes={}
    for j,k,m,a,b,power in schedule:
        delta=power-x["machines"][m]["standby_kw"]
        changes[a]=changes.get(a,0)+delta; changes[b]=changes.get(b,0)-delta
    cuts=sorted({0,end}|{t for t in changes if 0<=t<=end}|{b for a,b,p,s in profile if b<end})
    kw=idle; cost=0.0; i=0
    for a,b in zip(cuts,cuts[1:]):
        kw+=changes.get(a,0.0)
        while i+1<len(profile) and a>=profile[i][1]: i+=1
        price,solar=profile[i][2:]
        cost+=max(0.0,kw-solar)*price*(b-a)/(100*x["ticks_per_hour"])
    return cost

def actions(plan):
    order,machine,delay=plan; cursor=[0]*len(machine); out=[]
    for j in order:
        k=cursor[j]; out.append({"job":j,"machine":machine[j][k],"delay":delay[j][k]}); cursor[j]+=1
    return out

def evaluate(x,plan,profile,step,max_delay):
    try: sched,end=decode(x,actions(plan),step,max_delay)
    except ValueError: return None
    return (end,energy_cost(x,sched,end,profile))

def make_plan(x,rng,obj,restart,step,max_delay):
    machine=[]; delay=[]
    for job in x["jobs"]:
        mm=[]
        for op in job["operations"]:
            opts=op["options"]
            if restart%4==3: o=rng.choice(opts)
            elif obj=="makespan" or restart%4==0: o=min(opts,key=lambda v:(v["duration"],v["machine"]))
            else: o=min(opts,key=lambda v:(v["active_kw"]*v["duration"],v["duration"]))
            mm.append(o["machine"])
        machine.append(mm); delay.append([0]*len(mm))
    order=[j for j,job in enumerate(x["jobs"]) for _ in job["operations"]]
    if restart%3==0: rng.shuffle(order)
    elif restart%3==1: order.sort(key=lambda j:(len(x["jobs"][j]["operations"]),j))
    else: order.sort(key=lambda j:(-len(x["jobs"][j]["operations"]),j))
    if obj=="cost" and restart>=2 and max_delay:
        delay[order[0]][0]=rng.randrange(max_delay//step+1)*step
    return (order,machine,delay)

def mutate(plan,x,rng,obj,step,max_delay):
    order,ma,de=plan
    out=([*order],[*map(list,ma)],[*map(list,de)])
    modes=["swap","machine","swap","machine"]
    if obj=="cost" and max_delay: modes+=["delay","delay","delay"]
    mode=rng.choice(modes)
    if mode=="swap":
        if len(order)>1:
            a,b=rng.sample(range(len(order)),2);out[0][a],out[0][b]=out[0][b],out[0][a]
    else:
        j=rng.randrange(len(ma));k=rng.randrange(len(ma[j]))
        if mode=="machine":
            out[1][j][k]=rng.choice(x["jobs"][j]["operations"][k]["options"])["machine"]
        else:
            old=de[j][k]
            if rng.random()<0.22: new=rng.randrange(max_delay//step+1)*step
            else: new=max(0,min(max_delay,old+rng.choice([-32,-16,-4,-1,1,4,16,32])*step))
            out[2][j][k]=new
    return out

def optimize(x,obj,seed,iterations,restarts,step,max_delay):
    rng=random.Random(seed); profile=intervals(x)
    idx=0 if obj=="makespan" else 1
    best=None; best_metrics=None
    for r in range(restarts):
        current=make_plan(x,rng,obj,r,step,max_delay)
        metrics=evaluate(x,current,profile,step,max_delay)
        if metrics is None: continue
        if best_metrics is None or metrics[idx]<best_metrics[idx]: best,best_metrics=current,metrics
        budget=iterations//restarts+int(r<iterations%restarts)
        temperature=max(abs(metrics[idx])*0.06,0.1)
        for n in range(budget):
            candidate=mutate(current,x,rng,obj,step,max_delay)
            trial=evaluate(x,candidate,profile,step,max_delay)
            if trial is None: continue
            if trial[idx]<best_metrics[idx]: best,best_metrics=candidate,trial
            delta=trial[idx]-metrics[idx]
            temp=temperature*(0.01**(n/max(1,budget-1)))
            if delta<=0 or rng.random()<math.exp(-delta/max(temp,1e-12)):
                current,metrics=candidate,trial
    if best is None: raise ValueError("no feasible schedule found")
    return best,best_metrics

def solve(x,seed=2026,iterations=400,restarts=8,delay_step=25,max_delay=2400):
    check_instance(x)
    if iterations<0 or restarts<1 or delay_step<1 or max_delay<0 or max_delay%delay_step:
        raise ValueError("invalid search/delay settings")
    rowseed=int.from_bytes(hashlib.sha256((str(seed)+":"+x["instance_id"]).encode()).digest()[:8],"big")
    tp,tm=optimize(x,"makespan",rowseed,iterations,restarts,delay_step,max_delay)
    cp,cm=optimize(x,"cost",rowseed^0x9E3779B97F4A7C15,iterations,restarts,delay_step,max_delay)
    if cm[0]<tm[0]: tp,tm=cp,cm
    if tm[1]<cm[1]: cp,cm=tp,tm
    def pack(plan,metrics):
        aa=actions(plan); sched,end=decode(x,aa,delay_step,max_delay)
        cost=energy_cost(x,sched,end,intervals(x))
        assert end==metrics[0] and math.isclose(cost,metrics[1],abs_tol=1e-7)
        return {"actions":aa,"metrics":{"makespan_ticks":end,"energy_cost_eur":round(cost,8),
                "makespan_reward_unscaled":-end,"cost_reward_unscaled":round(-cost,8)}}
    return {"instance_id":x["instance_id"],"makespan":pack(tp,tm),"cost":pack(cp,cm)}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--instances",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--seed",type=int,default=2026)
    p.add_argument("--iterations",type=int,default=400)
    p.add_argument("--restarts",type=int,default=8)
    p.add_argument("--delay-step",type=int,default=25)
    p.add_argument("--max-delay",type=int,default=2400)
    p.add_argument("--limit",type=int,default=None)
    a=p.parse_args()
    if a.instances.resolve()==a.output.resolve(): p.error("output must differ from input")
    if a.limit is not None and a.limit<1: p.error("--limit must be positive")
    a.output.parent.mkdir(parents=True,exist_ok=True)
    tmp=None; count=0
    try:
        with tempfile.NamedTemporaryFile("w",encoding="utf-8",newline="\n",dir=a.output.parent,
                prefix="._mofjsp_",suffix=".jsonl",delete=False) as out:
            tmp=Path(out.name)
            for line_no,line in enumerate(a.instances.open(encoding="utf-8"),1):
                if not line.strip(): continue
                if a.limit is not None and count>=a.limit: break
                try:
                    row=solve(json.loads(line),a.seed,a.iterations,a.restarts,a.delay_step,a.max_delay)
                except (ValueError,KeyError,TypeError) as exc:
                    raise ValueError(f"input line {line_no}: {exc}") from exc
                row["input_line"]=line_no
                out.write(json.dumps(row,sort_keys=True,allow_nan=False)+"\n")
                count+=1
                if count%100==0: print(f"Solved {count} instances",flush=True)
        if count==0: raise ValueError("input has no instances")
        os.replace(tmp,a.output); tmp=None
        print(f"Wrote {count} rows with two solutions each: {a.output}")
    finally:
        if tmp is not None: tmp.unlink(missing_ok=True)

if __name__=="__main__": main()
