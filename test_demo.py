import copy
import json
import unittest
import server as app


class DemoTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        app.init_db(':memory:')

    def setUp(self):
        app.DB.execute('SAVEPOINT testcase')

    def tearDown(self):
        app.DB.execute('ROLLBACK TO testcase')
        app.DB.execute('RELEASE testcase')

    def test_public_pool_and_presets(self):
        overview = app.overview()
        self.assertEqual(overview['pool_count'], 100000)
        self.assertEqual(len(overview['layers']), 3)
        self.assertEqual([e['users'] for e in overview['experiments']], [100000]*3)
        self.assertEqual(app.query('SELECT uid,ut FROM users WHERE experiment_id=1'), app.query('SELECT uid,ut FROM users WHERE experiment_id=2'))

    def test_layer_capacity_and_validation(self):
        eid = app.mutate('create', {'name': 'same-layer', 'layer_id': 'recommendation', 'percent': 50, 'metric_ids': ['click_rate']})['id']
        config = app.experiment(eid)['config']
        self.assertEqual(config['traffic'][0]['pieces'], [{'begin': 200, 'length': 500}])
        with self.assertRaises(ValueError):
            app.mutate('create', {'name': 'overflow', 'layer_id': 'recommendation', 'percent': 31})
        for percent in (0, 101, 1.5, True):
            with self.assertRaises(ValueError):
                app.plan_config('bad', 'ranking', percent, ['click_rate'])
        with self.assertRaises(ValueError):
            app.mutate('create', {'name':'empty-metrics', 'metric_ids':[]})
        with self.assertRaises(ValueError):
            app.plan_config('bad', 'ranking', 10, [])
        with self.assertRaises(ValueError):
            app.plan_config('bad', 'invalid', 10, ['click_rate'])

    def test_same_layer_disjoint_cross_layer_independent(self):
        app.mutate('setup', {'experiment': 2, 'layer_id': 'recommendation', 'percent': 20, 'metric_ids': ['click_rate']})
        configs = [app.experiment(i)['config'] for i in (1, 2, 3)]
        group_a, group_b, group_c = [set(uid for uid in range(100001, 110001) if app.split(uid, 0, c)[0]>0) for c in configs]
        self.assertFalse(group_a & group_b)
        self.assertTrue(group_a & group_c)
        self.assertEqual(configs[0]['layer_salt'], configs[1]['layer_salt'])
        self.assertNotEqual(configs[0]['layer_salt'], configs[2]['layer_salt'])

    def test_batch_models_reproducibility_and_stable_profiles(self):
        app.mutate('start', {'experiment': 1})
        profiles = app.query('SELECT * FROM profiles WHERE experiment_id=1')
        users = app.query('SELECT * FROM users WHERE experiment_id=1')
        first = app.run_batch(1, 1234)
        second = app.run_batch(1, 1234)
        third = app.run_batch(1, 5678)
        def values(bid):
            return app.query('SELECT uid,metric,vid,value FROM observations WHERE batch=?', (bid,))
        rows = values(first)
        self.assertEqual(rows, values(second))
        self.assertNotEqual(rows, values(third))
        self.assertEqual(profiles, app.query('SELECT * FROM profiles WHERE experiment_id=1'))
        self.assertEqual(users, app.query('SELECT * FROM users WHERE experiment_id=1'))
        for row in rows:
            if row['metric']=='click_rate':
                self.assertTrue(0 <= row['value'] <= 100)
                self.assertEqual(row['value'] % 5, 0)
            elif row['metric']=='play_count':
                self.assertGreaterEqual(row['value'], 0)
                self.assertEqual(row['value'], int(row['value']))
            else:
                self.assertGreater(row['value'], 0)
        batch = app.query('SELECT schema_versions FROM batches WHERE id=?', (first,))[0]
        self.assertEqual(set(json.loads(batch['schema_versions'])), {'click_rate','watch_seconds','play_count'})

    def test_selected_metrics_and_state_lock(self):
        app.mutate('setup', {'experiment': 1, 'layer_id': 'ranking', 'percent': 5, 'metric_ids': ['play_count']})
        app.mutate('start', {'experiment': 1})
        with self.assertRaises(ValueError):
            app.mutate('setup', {'experiment': 1, 'layer_id': 'ranking', 'percent': 10, 'metric_ids': ['click_rate']})
        bid = app.run_batch(1, 2)
        self.assertEqual(app.query('SELECT DISTINCT metric FROM observations WHERE batch=?', (bid,)), [{'metric': 'play_count'}])
        old = app.query('SELECT * FROM observations WHERE batch=?', (bid,))
        app.mutate('stop', {'experiment': 1})
        app.mutate('setup', {'experiment': 1, 'layer_id': 'experience', 'percent': 10, 'metric_ids': ['click_rate']})
        app.mutate('start', {'experiment': 1})
        self.assertEqual(old, app.query('SELECT * FROM observations WHERE batch=?', (bid,)))
        self.assertEqual(app.query('SELECT reason FROM revisions WHERE experiment_id=1 ORDER BY id DESC')[0]['reason'], 'recompute')
        app.mutate('stop', {'experiment': 1})
        with self.assertRaises(ValueError):
            app.run_batch(1, 3)
        with self.assertRaises(ValueError):
            app.mutate('preview', {'experiment': 1, 'mode': 'generate', 'count': 5})

    def test_profile_distribution_units(self):
        app.mutate('start', {'experiment': 1})
        result = app.distribution({'experiment': 1, 'source': 'profile', 'metric': 'click_rate'})
        n = len(app.query('SELECT uid FROM users WHERE experiment_id=1 AND code>0'))
        self.assertEqual(sum(g['n'] for g in result['groups']), n)
        self.assertTrue(all(0 <= v <= 100 for g in result['groups'] for v in g['values']))
        self.assertTrue(app.distribution({'experiment': 1, 'metric': 'quality_score', 'source': 'profile'})['shared'])
        self.assertEqual(app.stats([1, 2, 3, 4])['p50'], 2.5)

    def test_metric_selection_does_not_change_assignment_fingerprint(self):
        c = app.experiment(1)['config']
        changed = copy.deepcopy(c)
        changed['metric_ids'] = ['play_count']
        self.assertEqual(app.fingerprint(c), app.fingerprint(changed))

    def test_murmur_reference_vectors(self):
        for text, expected in [('', 0), ('foo', -156908512), ('hello', 613153351), ('The quick brown fox jumps over the lazy dog', 776992547)]:
            self.assertEqual(app.murmur(text), expected)

    def test_split_ut_algorithm_and_lexical_order(self):
        c = app.experiment(1)['config']
        c['traffic'] = [{'pieces': [{'begin': 0, 'length': 1000}]}]
        c['vids'] = '100,11,2'
        for a in (0, 1):
            for b in (0, 1):
                c.update(layer_algorithm=a, flight_algorithm=b)
                for ut, key in [(-1, '123:'), (0, ':123:'), (8, '8:123:')]:
                    vid, bucket = app.split(123, ut, c)
                    self.assertEqual(vid, [100, 11, 2][app.hash_int(c['flight_salt'], key, b) % 3])
                    self.assertEqual(bucket, app.hash_int(c['layer_salt'], key, a) % 1000)
        self.assertEqual(app.split(0, 0, c)[0], -2)
        c['traffic'] = [{'pieces': [{'begin': bucket, 'length': 1}]}]
        self.assertGreater(app.split(123, 8, c)[0], 0)
        c['traffic'] = [{'pieces': [{'begin': bucket+1, 'length': 1}]}]
        self.assertEqual(app.split(123, 8, c)[0], -1)


if __name__ == '__main__':
    unittest.main()
