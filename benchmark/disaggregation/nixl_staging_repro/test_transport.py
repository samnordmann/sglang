import unittest
from transport import attest


def example(target, lane='rc_mlx5/mlx5_0:1/path0', cfg='inter-node', desc='zero-copy'):
    tag='[timestamp] [host-a:41:0]   '
    return [tag+'| prefill-123 '+cfg+' cfg#2 |',tag+'| remote memory write by ucp_put*(multi) from cuda/GPU0 to cuda/dev['+str(target)+'] |',
            tag+'| Range | Description | Config |', tag+'| 0..4K | copy-in | rc_mlx5/mlx5_0:1/path0 |',
            tag+f'| 4097..inf | {desc} | {lane} |']


class Tests(unittest.TestCase):
    def test_two_valid_routes(self):
        rows=attest(example(5)+example(6),{8192,262144},41)
        self.assertEqual(len(rows),4)
        self.assertTrue(all(row['direct_rdma'] for row in rows))
        self.assertEqual({row['dst'] for row in rows},{'cuda/dev[5]','cuda/dev[6]'})

    def test_alternate_transport_is_identified(self):
        for lane,desc in [('tcp/eth0','zero-copy'),('rc_mlx5/mlx5_0:1/path0','staged zero-copy')]:
            rows=attest(example(5,lane,desc=desc)+example(6,lane,desc=desc),{8192},41)
            self.assertTrue(all(r['lane']==lane and not r['direct_rdma'] for r in rows))
        rows=attest(example(5)+example(6),{2048},41)
        self.assertTrue(all(not r['direct_rdma'] for r in rows))

    def test_reject_missing_or_unrelated_evidence(self):
        cases=[(example(5),8192,41),(example(5)+example(6),8192,29),

               (example(5)+example(6,'cuda_ipc/cuda'),8192,41),
               (example(5)+example(6,cfg='intra-node'),8192,41)]
        for lines,size,pid in cases:
            with self.subTest(size=size,pid=pid),self.assertRaises(ValueError):
                attest(lines,{size},pid)


if __name__=='__main__':
    unittest.main()
