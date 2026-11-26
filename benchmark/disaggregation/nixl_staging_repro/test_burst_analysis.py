import unittest

from burst_analysis import analyse, endpoints, rank_interval, summarize
from plan import batches, LEGACY


class Tests(unittest.TestCase):
    def test_latest_b_event_and_order_bounds(self):
        def row(ready, last_b, bounds, start, http):
            return dict(prefill_a_ns=ready,last_decode_b_ns=last_b,last_decode_a_bounds_ns=bounds,
                        start_a_ns=start,http_response_a_ns=http)
        # The latest event on B is the first row. Both anchors use the first
        # timestamp across the full wave, independently of scheduling order.
        result=endpoints([row(110,9000,[195,209],90,240),row(100,8998,[198,220],70,255)])
        self.assertEqual(result["first_prefill_to_last_decode"],(98,109))
        self.assertEqual(result["first_start_to_last_http"],(185,185))
        stat=summarize([((9000,10000),(5000+i,6000+i)) for i in range(24)])
        self.assertEqual(stat['paired_median_delta_us'],[-4.9885,-2.9885])
        self.assertEqual(stat["ranks_one_based"],[7,18])
        self.assertEqual(stat["median_delta_order_stat_interval_us"],[-4.994,-2.983])
        self.assertEqual(stat["definitely_faster"],24)
        self.assertIsNone(rank_interval(4))

    def test_pair_rooms_by_identity_and_skip_unqualified_cells(self):
        plan=dict(LEGACY,lengths=[64],repeats=6)
        runs={}
        for order in plan['orders']:
            for arm in order:
                measurements=[]
                for batch in batches(plan):
                    for room,_,depth,rep,_ in batch:
                        if depth==4 and rep>=0:
                            time=room*10000
                            delay=(-2000 if arm.endswith('patched') else 0)
                            measurements.append(dict(room=room,cell='L64-D4',clock_issue=None,
                                start_a_ns=time,prefill_a_ns=time+1000,last_decode_b_ns=time+8_000_000,
                                last_decode_a_bounds_ns=[time+4000+delay,time+4100+delay],
                                http_response_a_ns=time+6000+delay))
                runs[arm]={'measurements':list(reversed(measurements)) if arm.endswith('patched') else measurements}
        source=dict(qualified=False,plan=plan,cases={'L64-D1':{'qualified':False},
                                                     'L64-D4':{'qualified':True,'runs':runs}})
        result=analyse(source)
        self.assertEqual(result['cases']['L64-D1'],{'qualified':False})
        for name, stats in result['cases']['L64-D4']['runs'].items():
            # Each wave's fourth room starts 30 us later than the first.
            # Gaps between one wave's final return and the next wave are idle.
            active_ns = 6 * (34_000 if name.endswith('patched') else 36_000)
            self.assertEqual(stats['active_http_duration_ns'], active_ns)
            self.assertEqual(stats['measured_requests'], 24)
            self.assertEqual(stats['measured_waves'], 6)
            self.assertAlmostEqual(stats['active_http_requests_per_s'], 24e9/active_ns)
        for order in result['cases']['L64-D4']['orders']:
            m=order['metrics']
            self.assertEqual(m['first_prefill_to_last_decode']['paired_median_delta_us'],[-2.1,-1.9])
            self.assertEqual(m['first_start_to_last_http']['paired_median_delta_us'],[-2,-2])
            self.assertEqual(len(order['waves']),6)
        runs['1-baseline']['measurements'][0]=runs['1-baseline']['measurements'][1]
        with self.assertRaisesRegex(ValueError,'duplicate, extra or missing'):
            analyse(source)
