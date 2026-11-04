"""CPU-only unit checks: python -m unittest -v test_solve_mofjsp_step2.py"""
import unittest
from solve_mofjsp_step2 import decode, energy_cost, intervals, solve

def fixture():
    return {
        "instance_id":"tiny","format_version":"jd-mofjsp-step2-v2",
        "num_jobs":2,"num_machines":2,"num_operations":2,
        "horizon":400,"ticks_per_hour":100,
        "machines":[{"machine":0,"standby_kw":1},{"machine":1,"standby_kw":1}],
        "jobs":[
            {"job":0,"operations":[{"operation":0,"options":[
                {"machine":0,"duration":100,"active_kw":10},
                {"machine":1,"duration":150,"active_kw":2}]}]},
            {"job":1,"operations":[{"operation":0,"options":[
                {"machine":0,"duration":100,"active_kw":10},
                {"machine":1,"duration":150,"active_kw":2}]}]}],
        "energy_profile":{"interval_ticks":100,"price_solar":[[10,0],[10,0],[50,0],[10,0]]}}

class Tests(unittest.TestCase):
    def test_standby_cost(self):
        x=fixture()
        ss,t=decode(x,[{"job":0,"machine":0,"delay":0},{"job":1,"machine":1,"delay":0}])
        self.assertEqual(t,150)
        self.assertAlmostEqual(energy_cost(x,ss,t,intervals(x)),1.35)

    def test_solar(self):
        x=fixture(); x["energy_profile"]["price_solar"][0]=[-10,100]
        ss,t=decode(x,[{"job":0,"machine":0,"delay":0},{"job":1,"machine":1,"delay":0}])
        self.assertAlmostEqual(energy_cost(x,ss,t,intervals(x)),0.15)

    def test_invalid_actions(self):
        x=fixture(); a=[{"job":0,"machine":0,"delay":0},{"job":1,"machine":1,"delay":0}]
        for bad in (a[:1],[a[0],a[0]],[dict(a[0],delay=13),a[1]],
                    [dict(a[0],machine=20),a[1]]):
            with self.subTest(bad=bad),self.assertRaises(ValueError):
                decode(x,bad)

    def test_optimization_and_reproducibility(self):
        x=fixture()
        a=solve(x,iterations=80,restarts=4,max_delay=200)
        self.assertEqual(a,solve(x,iterations=80,restarts=4,max_delay=200))
        for name in ("makespan","cost"):
            schedule,end=decode(x,a[name]["actions"],max_delay=200)
            metrics=a[name]["metrics"]
            self.assertEqual(end,metrics["makespan_ticks"])
            self.assertAlmostEqual(energy_cost(x,schedule,end,intervals(x)),metrics["energy_cost_eur"])
        self.assertLessEqual(a["makespan"]["metrics"]["makespan_ticks"],a["cost"]["metrics"]["makespan_ticks"])
        self.assertLessEqual(a["cost"]["metrics"]["energy_cost_eur"],a["makespan"]["metrics"]["energy_cost_eur"])

    def test_v1(self):
        x=fixture()
        pairs=x["energy_profile"]["price_solar"]
        x["format_version"]="jd-mofjsp-step2-v1"
        x["energy_profile"]=[{"start":100*i,"end":100*(i+1),
             "price_ct_per_kwh":p,"solar_kw":s} for i,(p,s) in enumerate(pairs)]
        self.assertEqual(solve(x,iterations=10,restarts=2)["instance_id"],"tiny")

if __name__=="__main__": unittest.main()
