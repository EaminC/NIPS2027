"""partition-bias-lab v2 local demo. Python standard library only."""
import argparse
import csv
import hashlib
import io
import json
import math
import random
import sqlite3
import statistics
import threading
import uuid
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).parent
LOCK = threading.RLock()
DB = None
PREVIEWS = {}
DEFAULT_USERS = 100000
MAX_USERS = 100000


def user_count(value):
    if isinstance(value, bool) or not str(value).isascii() or not str(value).isdecimal():
        raise ValueError("人数必须为整数")
    count = int(value)
    if not 1 <= count <= MAX_USERS:
        raise ValueError("人数范围为 1–100,000")
    return count


def now():
    return datetime.now().isoformat(timespec='seconds')


def murmur(text, seed=0):
    data = text.encode('utf-8')
    h = seed & 0xffffffff
    def mix(k):
        k = k * 0xcc9e2d51 & 0xffffffff
        k = (k << 15 | k >> 17) & 0xffffffff
        return k * 0x1b873593 & 0xffffffff
    end = len(data) // 4 * 4
    for i in range(0, end, 4):
        h ^= mix(int.from_bytes(data[i:i + 4], 'little'))
        h = (h << 13 | h >> 19) & 0xffffffff
        h = (h * 5 + 0xe6546b64) & 0xffffffff
    if end < len(data):
        h ^= mix(int.from_bytes(data[end:], 'little'))
    h ^= len(data)
    h ^= h >> 16
    h = h * 0x85ebca6b & 0xffffffff
    h ^= h >> 13
    h = h * 0xc2b2ae35 & 0xffffffff
    h ^= h >> 16
    return h if h < 0x80000000 else h - 0x100000000


def hash_int(salt, key, algorithm):
    return murmur(key, murmur(salt)) if algorithm == 1 else murmur(key + salt)


def split(uid, ut, c):
    if uid <= 0:
        return -2, None
    if not all(str(c.get(k, '')).strip() for k in ('layer_salt', 'flight_salt', 'vids')) or not c.get('traffic'):
        return -3, None
    key = f'{uid}:' if ut < 0 else f':{uid}:' if ut == 0 else f'{ut}:{uid}:'
    bucket = hash_int(c['layer_salt'], key, c['layer_algorithm']) % 1000
    if not any(p['begin'] <= bucket < p['begin'] + p['length'] for obj in c['traffic'] for p in obj.get('pieces', [])):
        return -1, bucket
    vids = sorted(c['vids'].split(','))
    return int(vids[hash_int(c['flight_salt'], key, c['flight_algorithm']) % len(vids)]), bucket


def fingerprint(c):
    split_fields = {k: c[k] for k in ('layer_salt', 'layer_algorithm', 'flight_salt', 'flight_algorithm', 'traffic', 'vids')}
    return hashlib.sha256(json.dumps(split_fields, sort_keys=True, separators=(',', ':')).encode()).hexdigest()[:12]


def query(sql, args=()):
    return [dict(r) for r in DB.execute(sql, args)]


def experiment(eid):
    rows = query('SELECT * FROM experiments WHERE id=?', (eid,))
    if not rows:
        raise ValueError('实验不存在')
    e = rows[0]
    e['config'] = json.loads(e['config'])
    return e


LAYERS = [
    {'id': 'recommendation', 'name': '推荐策略层', 'description': '召回、内容新鲜度与推荐策略', 'salt': 'partition_bias_lab_recommendation_v1'},
    {'id': 'ranking', 'name': '排序模型层', 'description': '排序模型、特征与权重', 'salt': 'partition_bias_lab_ranking_v1'},
    {'id': 'experience', 'name': '产品体验层', 'description': '页面交互、样式与播放体验', 'salt': 'partition_bias_lab_experience_v1'},
]
METRICS = [
    {'id': 'click_rate', 'name': '点击率', 'unit': '%', 'mode': 'heterogeneous', 'version': 1,
     'family': 'binomial', 'profile_key': 'p', 'profile_scale': 100, 'profile_label': '用户点击概率',
     'description': '每位用户每批固定 20 次曝光，点击次数 / 20 × 100%；组均值即整体 CTR。'},
    {'id': 'watch_seconds', 'name': '人均观看时长', 'unit': '秒', 'mode': 'heterogeneous', 'version': 1,
     'family': 'lognormal', 'profile_key': 'mean', 'profile_scale': 1, 'profile_label': '用户期望观看时长',
     'description': '每位用户每批累计观看秒数；使用非负、右偏的对数正态分布模拟。'},
    {'id': 'play_count', 'name': '人均播放次数', 'unit': '次', 'mode': 'heterogeneous', 'version': 1,
     'family': 'poisson', 'profile_key': 'rate', 'profile_scale': 1, 'profile_label': '用户期望播放次数',
     'description': '每位用户每批的播放次数；使用 Poisson 计数模型，观测值均为非负整数。'},
]
LEGACY_METRICS = [
    {'id': 'watch_time', 'name': '观看时长（旧版）', 'unit': '秒', 'mode': 'heterogeneous', 'version': 1, 'mean': 42, 'sigma': 6.5, 'profile_key': 'mu', 'profile_scale': 1, 'profile_label': '用户均值', 'description': '旧版 Normal 模型，仅用于查看历史数据。'},
    {'id': 'quality_score', 'name': '内容质量（旧版）', 'unit': '分', 'mode': 'iid', 'version': 1, 'mean': 70, 'sigma': 10, 'description': '旧版共享 Normal 模型，仅用于查看历史数据。'},
]


def chosen_metrics(e):
    ids = e['config'].get('metric_ids', [m['id'] for m in LEGACY_METRICS])
    return [m for m in METRICS + LEGACY_METRICS if m['id'] in ids]


def ensure_profiles(eid):
    e = experiment(eid)
    for m in chosen_metrics(e):
        if m['mode'] == 'iid':
            continue
        sql = 'SELECT u.uid FROM users u WHERE u.experiment_id=? AND NOT EXISTS (SELECT 1 FROM profiles p WHERE p.experiment_id=u.experiment_id AND p.uid=u.uid AND p.metric=?)'
        if e['config'].get('pool'):
            sql += ' AND u.code>0'
        rows = []
        for u in query(sql, (eid, m['id'])):
            rng = random.Random(f'pool-profile:{u["uid"]}:{m["id"]}:2026')
            if m['id'] == 'click_rate':
                profile = {'p': rng.betavariate(4.8, 35.2), 'exposures': 20}
            elif m['id'] == 'watch_seconds':
                profile = {'mean': rng.lognormvariate(math.log(120)-.5*.65**2, .65), 'sigma_log': .5}
            elif m['id'] == 'play_count':
                profile = {'rate': rng.gammavariate(4, 2)}
            else:
                profile = {'mu': rng.gauss(42, 9), 'sigma': 6.5}
            rows.append((eid, u['uid'], m['id'], json.dumps(profile), 2026, 1))
        DB.executemany('INSERT OR IGNORE INTO profiles VALUES (?,?,?,?,?,?)', rows)


def layer_allocations(layer_id, exclude=None):
    allocations = []
    for row in query('SELECT id,name,status,config FROM experiments ORDER BY id'):
        c = json.loads(row['config'])
        if row['id'] != exclude and c.get('layer_id') == layer_id:
            for obj in c['traffic']:
                for piece in obj.get('pieces', []):
                    allocations.append(dict(experiment=row['id'], name=row['name'], status=row['status'], **piece))
    return allocations


def plan_config(name, layer_id, percent, metric_ids, exclude=None):
    layer = next((x for x in LAYERS if x['id'] == layer_id), None)
    if not layer:
        raise ValueError('请选择预置流量层')
    if isinstance(percent, bool) or not str(percent).isdecimal() or not 1 <= int(percent) <= 100:
        raise ValueError('流量比例必须为 1–100 的整数')
    if not isinstance(metric_ids, list) or not metric_ids or any(x not in [m['id'] for m in METRICS] for x in metric_ids):
        raise ValueError('请至少选择一个有效指标')
    length = int(percent)*10
    occupied = set()
    for piece in layer_allocations(layer_id, exclude):
        occupied.update(range(piece['begin'], piece['begin']+piece['length']))
    available = [i for i in range(1000) if i not in occupied]
    if len(available) < length:
        raise ValueError(f'该层仅剩 {len(available)/10:g}% 可分配流量，请降低比例或选择其他层')
    buckets = available[:length]
    pieces = []
    for bucket in buckets:
        if pieces and pieces[-1]['begin']+pieces[-1]['length'] == bucket:
            pieces[-1]['length'] += 1
        else:
            pieces.append({'begin': bucket, 'length': 1})
    return {'pool': True, 'layer_id': layer_id, 'metric_ids': list(dict.fromkeys(metric_ids)),
            'layer_salt': layer['salt'], 'layer_algorithm': 1,
            'flight_salt': 'flight_'+hashlib.sha256(name.encode()).hexdigest()[:12], 'flight_algorithm': 1,
            'traffic': [{'pieces': pieces}], 'vids': '1001,1002'}


def sync_pool_members(eid):
    # A shared 100k-person universe; experiment membership is just its assignment view.
    DB.execute('INSERT OR IGNORE INTO users SELECT ?,uid,ut,NULL,NULL,?,? FROM user_pool', (eid, '', now()))


def poisson(rng, rate):
    limit, product, k = math.exp(-rate), 1.0, 0
    while product > limit:
        product *= rng.random()
        k += 1
    return k-1


def assign(eid):
    e = experiment(eid)
    fp = fingerprint(e['config'])
    previous = query('SELECT * FROM revisions WHERE experiment_id=? ORDER BY id DESC LIMIT 1', (eid,))
    users = query('SELECT * FROM users WHERE experiment_id=?', (eid,))
    changed = not previous or previous[0]['fingerprint'] != fp
    missing = any(u['fingerprint'] != fp for u in users)
    if not changed and not missing:
        return
    reason = 'initial' if not previous else 'recompute' if changed else 'append'
    for u in users:
        if changed or u['fingerprint'] != fp:
            code, bucket = split(u['uid'], u['ut'], e['config'])
            DB.execute('UPDATE users SET code=?,bucket=?,fingerprint=? WHERE experiment_id=? AND uid=?', (code, bucket, fp, eid, u['uid']))
    snapshot = query('SELECT uid,ut,code,bucket FROM users WHERE experiment_id=? ORDER BY uid', (eid,))
    DB.execute('INSERT INTO revisions(experiment_id,reason,fingerprint,snapshot,created_at) VALUES(?,?,?,?,?)', (eid, reason, fp, json.dumps(snapshot), now()))


def run_batch(eid, seed):
    e = experiment(eid)
    if e['status'] != 'running':
        raise ValueError('请先启动实验，再生成 observation 批次')
    ensure_profiles(eid)
    metrics = chosen_metrics(e)
    rngs = {m['id']: random.Random(f'{seed}:{m["id"]}') for m in metrics}
    cur = DB.execute('INSERT INTO batches(experiment_id,seed,fingerprint,created_at,status,schema_versions) VALUES(?,?,?,?,?,?)', (eid, seed, fingerprint(e['config']), now(), 'complete', json.dumps({m['id']: m['version'] for m in metrics})))
    bid = cur.lastrowid
    profiles = {(r['uid'], r['metric']): json.loads(r['profile_json']) for r in query('SELECT * FROM profiles WHERE experiment_id=?', (eid,))}
    rows = []
    for u in query('SELECT uid,code FROM users WHERE experiment_id=? AND code>0 ORDER BY uid', (eid,)):
        for m in metrics:
            rng = rngs[m['id']]
            p = profiles.get((u['uid'], m['id']), {})
            if m['id'] == 'click_rate':
                value = sum(rng.random() < p['p'] for _ in range(p['exposures'])) * 100 / p['exposures']
            elif m['id'] == 'watch_seconds':
                sigma = p['sigma_log']
                value = rng.lognormvariate(math.log(p['mean'])-.5*sigma*sigma, sigma)
            elif m['id'] == 'play_count':
                value = poisson(rng, p['rate'])
            else:
                value = rng.gauss(p.get('mu', m['mean']), m['sigma'])
            rows.append((bid, eid, u['uid'], m['id'], u['code'], value))
    DB.executemany('INSERT INTO observations VALUES(?,?,?,?,?,?)', rows)
    return bid


def create(name, layer_id='recommendation', percent=10, metric_ids=None):
    c = plan_config(name, layer_id, percent, [m['id'] for m in METRICS] if metric_ids is None else metric_ids)
    eid = DB.execute('INSERT INTO experiments(name,status,config,created_at) VALUES(?,?,?,?)', (name, 'draft', json.dumps(c), now())).lastrowid
    sync_pool_members(eid)
    return eid


def migrate_pool():
    if query('SELECT value FROM settings WHERE key="pool_v1"'):
        return
    # Keep all existing identities; add deterministic synthetic users to reach exactly 100k.
    DB.execute('INSERT OR IGNORE INTO user_pool SELECT uid,MIN(ut) FROM users GROUP BY uid')
    count = DB.execute('SELECT COUNT(*) FROM user_pool').fetchone()[0]
    if count > DEFAULT_USERS:
        raise ValueError('现有唯一用户超过 100,000，无法自动迁移公共池')
    existing = {r['uid'] for r in query('SELECT uid FROM user_pool')}
    additions, uid = [], 100001
    while count+len(additions) < DEFAULT_USERS:
        if uid not in existing:
            additions.append((uid, 0))
        uid += 1
    DB.executemany('INSERT INTO user_pool VALUES(?,?)', additions)
    old = query('SELECT id FROM experiments ORDER BY id')
    for i, row in enumerate(old):
        e = experiment(row['id'])
        if e['config'].get('pool'):
            continue
        # Keep historical assignments/profiles/batches; migrate only current configuration.
        c = plan_config(e['name'], LAYERS[i % len(LAYERS)]['id'], 20, [m['id'] for m in METRICS], e['id'])
        DB.execute('UPDATE experiments SET config=?,status=? WHERE id=?', (json.dumps(c), 'draft' if e['status']=='draft' else 'stopped', e['id']))
        sync_pool_members(e['id'])
        DB.execute('UPDATE users SET ut=(SELECT ut FROM user_pool WHERE user_pool.uid=users.uid),code=NULL,bucket=NULL,fingerprint="" WHERE experiment_id=?', (e['id'],))
    if not old:
        create('远汀实验', 'recommendation', 20)
        create('青岚实验', 'ranking', 20)
        create('星砂实验', 'experience', 10)
    DB.execute('INSERT INTO settings VALUES("pool_v1","complete")')


def init_db(path):
    global DB
    DB = sqlite3.connect(path, check_same_thread=False)
    DB.row_factory = sqlite3.Row
    DB.executescript('''
    PRAGMA journal_mode=WAL;
    CREATE TABLE IF NOT EXISTS settings(key TEXT PRIMARY KEY,value TEXT);
    CREATE TABLE IF NOT EXISTS user_pool(uid INTEGER PRIMARY KEY,ut INTEGER);
    CREATE TABLE IF NOT EXISTS experiments(id INTEGER PRIMARY KEY,name TEXT,status TEXT,config TEXT,created_at TEXT);
    CREATE TABLE IF NOT EXISTS users(experiment_id INTEGER,uid INTEGER,ut INTEGER,code INTEGER,bucket INTEGER,fingerprint TEXT,created_at TEXT,PRIMARY KEY(experiment_id,uid));
    CREATE TABLE IF NOT EXISTS profiles(experiment_id INTEGER,uid INTEGER,metric TEXT,profile_json TEXT,seed INTEGER,revision INTEGER,PRIMARY KEY(experiment_id,uid,metric));
    CREATE TABLE IF NOT EXISTS revisions(id INTEGER PRIMARY KEY,experiment_id INTEGER,reason TEXT,fingerprint TEXT,snapshot TEXT,created_at TEXT);
    CREATE TABLE IF NOT EXISTS batches(id INTEGER PRIMARY KEY,experiment_id INTEGER,seed INTEGER,fingerprint TEXT,created_at TEXT,status TEXT,schema_versions TEXT);
    CREATE TABLE IF NOT EXISTS observations(batch INTEGER,experiment_id INTEGER,uid INTEGER,metric TEXT,vid INTEGER,value REAL);
    CREATE INDEX IF NOT EXISTS obs_selection ON observations(experiment_id,metric,batch);
    ''')
    # Existing salts are experimental data: preserve them across a product rename.
    # All experiments within a layer must continue sharing the same hash seed.
    for layer in LAYERS:
        saved = [json.loads(r['config']) for r in query('SELECT config FROM experiments')]
        matching = [c for c in saved if c.get('layer_id') == layer['id']]
        layer['salt'] = matching[0]['layer_salt'] if matching else f"partition_bias_lab_{layer['id']}_v1"
    with DB:
        migrate_pool()


def overview():
    result = []
    for r in query('SELECT id FROM experiments'):
        e = experiment(r['id'])
        eid = e['id']
        e['groups'] = query('SELECT code,COUNT(*) AS n FROM users WHERE experiment_id=? GROUP BY code', (eid,))
        e['users'] = sum(g['n'] for g in e['groups'])
        e['in_traffic'] = sum(g['n'] for g in e['groups'] if g['code'] and g['code'] > 0)
        e['profiles'] = DB.execute('SELECT COUNT(*) FROM profiles WHERE experiment_id=?', (eid,)).fetchone()[0]
        e['batches'] = query('SELECT * FROM batches WHERE experiment_id=? ORDER BY id DESC', (eid,))
        e['revisions'] = query('SELECT id,reason,fingerprint,created_at FROM revisions WHERE experiment_id=? ORDER BY id DESC', (eid,))
        e['fingerprint'] = fingerprint(e['config'])
        e['metric_ids'] = [m['id'] for m in chosen_metrics(e)]
        e['available_metric_ids'] = sorted(set(e['metric_ids']) | {r['metric'] for r in query('SELECT DISTINCT metric FROM observations WHERE experiment_id=?', (eid,))})
        e['percent'] = sum(p['length'] for obj in e['config']['traffic'] for p in obj.get('pieces', []))/10
        e['assigned'] = all(g['code'] is not None for g in e['groups'])
        result.append(e)
    return {'experiments': result, 'metrics': METRICS, 'legacy_metrics': LEGACY_METRICS, 'pool_count': DB.execute('SELECT COUNT(*) FROM user_pool').fetchone()[0], 'layers': [dict(layer, allocations=layer_allocations(layer['id'])) for layer in LAYERS]}


def rows_for(params):
    eid = int(params.get('experiment', 1))
    dataset = params.get('dataset', 'users')
    sqls = {
        'users': ('SELECT uid,ut,code AS vid,bucket,fingerprint,created_at FROM users WHERE experiment_id=?', [eid]),
        'profiles': ('SELECT p.uid,u.code AS vid,p.metric,1 AS schema_version,p.profile_json,p.seed,p.revision FROM profiles p JOIN users u ON p.uid=u.uid AND p.experiment_id=u.experiment_id WHERE p.experiment_id=?', [eid]),
        'observations': ('SELECT batch,uid,metric,vid,ROUND(value,4) AS value FROM observations WHERE experiment_id=?', [eid]),
        'batches': ('SELECT id AS batch,seed,fingerprint,status,schema_versions,created_at FROM batches WHERE experiment_id=?', [eid]),
        'history': ('SELECT id AS revision,reason,fingerprint,created_at FROM revisions WHERE experiment_id=?', [eid]),
    }
    if dataset not in sqls:
        raise ValueError('未知数据集')
    sql, args = sqls[dataset]
    rows = query(sql, args)
    for key in ('metric', 'batch', 'vid'):
        if params.get(key):
            rows = [r for r in rows if key not in r or str(r[key]) == str(params[key])]
    if params.get('uid'):
        rows = [r for r in rows if 'uid' in r and str(params['uid']) in str(r['uid'])]
    return rows


def stats(values):
    a = sorted(values)
    if not a:
        return {'n': 0}
    def percentile(q):
        i = (len(a) - 1) * q
        return a[int(i)] * (1 - i % 1) + a[math.ceil(i)] * (i % 1)
    return dict(n=len(a), mean=statistics.mean(a), std=statistics.stdev(a) if len(a) > 1 else 0, min=a[0], max=a[-1], p25=percentile(.25), p50=percentile(.5), p75=percentile(.75))


def distribution(p):
    eid = int(p.get('experiment', 1))
    metric = next((m for m in METRICS + LEGACY_METRICS if m['id'] == p.get('metric', 'click_rate')), None)
    if not metric:
        raise ValueError('未知指标')
    if p.get('source') == 'profile' and metric['mode'] == 'iid':
        return {'shared': True, 'metric': metric, 'groups': []}
    if p.get('source') == 'profile':
        rows = query('SELECT u.code AS vid,p.profile_json FROM profiles p JOIN users u ON p.uid=u.uid AND p.experiment_id=u.experiment_id WHERE p.experiment_id=? AND p.metric=? AND u.code>0', (eid, metric['id']))
        rows = [{'vid': r['vid'], 'value': json.loads(r['profile_json'])[metric['profile_key']] * metric.get('profile_scale', 1)} for r in rows]
    else:
        sql = 'SELECT vid,value FROM observations WHERE experiment_id=? AND metric=?'
        args = [eid, metric['id']]
        if p.get('batch'):
            sql += ' AND batch=?'
            args.append(int(p['batch']))
        rows = query(sql, args)
    groups = []
    for vid in sorted(set(r['vid'] for r in rows)):
        vals = [r['value'] for r in rows if r['vid'] == vid]
        groups.append({'vid': vid, 'values': vals, **stats(vals)})
    return {'shared': False, 'metric': metric, 'groups': groups}


def mutate(action, data):
    if action == 'create':
        name = str(data.get('name', '')).strip()
        if not name or len(name) > 80:
            raise ValueError('请输入 1–80 字符的实验名称')
        return {'id': create(name, data.get('layer_id', 'recommendation'), data.get('percent', 10), data.get('metric_ids'))}
    eid = int(data['experiment'])
    e = experiment(eid)
    if action == 'setup':
        if e['status'] == 'running':
            raise ValueError('请先停止实验，再调整流量和指标')
        c = plan_config(e['name'], data['layer_id'], data['percent'], data['metric_ids'], eid)
        DB.execute('UPDATE experiments SET config=? WHERE id=?', (json.dumps(c), eid))
    elif action == 'start':
        assign(eid)
        ensure_profiles(eid)
        DB.execute('UPDATE experiments SET status="running" WHERE id=?', (eid,))
    elif action == 'stop':
        DB.execute('UPDATE experiments SET status="stopped" WHERE id=?', (eid,))
    elif action == 'run':
        times, seed = int(data.get('times', 1)), int(data.get('seed', 2026))
        if not 1 <= times <= 10:
            raise ValueError('每次支持 1–10 个批次')
        return {'batches': [run_batch(eid, seed + i) for i in range(times)]}
    elif action == 'config':
        if e['config'].get('pool'):
            raise ValueError('公共池实验请通过流量层和比例配置分流')
        if e['status'] == 'running':
            raise ValueError('运行中分流配置已锁定，请先停止实验')
        c = data['config']
        if not c['layer_salt'].strip() or not c['flight_salt'].strip():
            raise ValueError('哈希盐不能为空')
        # Guava legacy murmur3_32 has known non-BMP string behavior; restrict demo salts.
        if not (c['layer_salt'].isascii() and c['flight_salt'].isascii()):
            raise ValueError('此 demo 的哈希盐仅支持 ASCII 字符')
        vids = c['vids'].split(',')
        if not vids or any(not v.isdecimal() or not 0 < int(v) <= 2147483647 for v in vids) or len(set(vids)) != len(vids):
            raise ValueError('vid 应为互不重复的正整数，以英文逗号分隔')
        if not isinstance(c['traffic'], list) or not c['traffic']:
            raise ValueError('trafficMap 必须为非空数组')
        for obj in c['traffic']:
            for piece in obj.get('pieces', []):
                if not all(isinstance(piece[k], int) for k in ('begin', 'length')) or not 0 <= piece['begin'] < 1000 or not 0 < piece['length'] <= 1000 - piece['begin']:
                    raise ValueError('流量区间必须在 0–999 范围内')
        if not any(obj.get('pieces') for obj in c['traffic']):
            raise ValueError('至少需要一个流量区间')
        DB.execute('UPDATE experiments SET config=? WHERE id=?', (json.dumps(c), eid))
    elif action == 'preview':
        if e['config'].get('pool'):
            raise ValueError('公共池固定为 100,000 位用户，请调整实验流量比例')
        raw = data.get('text', '').strip()
        if data.get('mode') == 'generate':
            count = user_count(data.get('count', DEFAULT_USERS))
            start = DB.execute('SELECT COALESCE(MAX(uid),100000)+1 FROM users WHERE experiment_id=?', (eid,)).fetchone()[0]
            incoming = [{'uid': start+i, 'ut': 0} for i in range(count)]
        elif raw.startswith('['):
            incoming = json.loads(raw)
        else:
            reader = csv.DictReader(io.StringIO(raw), delimiter='\t' if '\t' in raw.split('\n')[0] else ',')
            incoming = list(reader)
        if not incoming or len(incoming) > MAX_USERS:
            raise ValueError('请导入 1–100,000 行，表头为 uid,ut')
        existing = {r['uid'] for r in query('SELECT uid FROM users WHERE experiment_id=?', (eid,))}
        users = {}
        for row in incoming:
            uid, ut = int(row['uid']), int(row.get('ut', 0))
            if not 0 < uid <= 9223372036854775807 or not -2147483648 <= ut <= 2147483647:
                raise ValueError('uid 必须为正 long 整数，ut 必须为 int 整数')
            if uid not in existing:
                users[uid] = ut
        if len(existing) + len(users) > MAX_USERS:
            raise ValueError('每个实验最多支持 100,000 位用户，请减少追加数量')
        token = uuid.uuid4().hex
        PREVIEWS[token] = (eid, users, e['status'], fingerprint(e['config']), len(existing))
        return {'token': token, 'count': len(users), 'skipped': len(incoming) - len(users), 'sample': list(users.items())[:5]}
    elif action == 'append':
        preview = PREVIEWS.pop(data.get('token'), None)
        if not preview or preview[0] != eid:
            raise ValueError('预览已失效，请重新预览')
        _, users, status, fp, count = preview
        current_count = DB.execute('SELECT COUNT(*) FROM users WHERE experiment_id=?', (eid,)).fetchone()[0]
        if status != e['status'] or fp != fingerprint(e['config']) or count != current_count:
            raise ValueError('实验或名单已变化，请重新预览')
        DB.executemany('INSERT INTO users VALUES(?,?,?,?,?,?,?)', [(eid, uid, ut, None, None, '', now()) for uid, ut in users.items()])
        ensure_profiles(eid)
        if e['status'] == 'running':
            assign(eid)
        return {'count': len(users)}
    else:
        raise ValueError('未知操作')
    return {'ok': True}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass

    def send(self, value, status=200, content_type='application/json; charset=utf-8'):
        payload = value if isinstance(value, bytes) else json.dumps(value, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(payload)))
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self):
        try:
            url = urlparse(self.path)
            p = {k: v[0] for k, v in parse_qs(url.query).items()}
            with LOCK:
                if url.path == '/api/overview':
                    return self.send(overview())
                if url.path == '/api/data':
                    rows = rows_for(p)
                    page = max(0, int(p.get('page', 0)))
                    return self.send({'total': len(rows), 'rows': rows[page * 15:(page + 1) * 15]})
                if url.path == '/api/export':
                    rows = rows_for(p)
                    output = io.StringIO()
                    if rows:
                        writer = csv.DictWriter(output, fieldnames=rows[0].keys())
                        writer.writeheader()
                        writer.writerows(rows)
                    return self.send(output.getvalue().encode('utf-8-sig'), content_type='text/csv; charset=utf-8')
                if url.path == '/api/distribution':
                    return self.send(distribution(p))
                if url.path == '/api/diff':
                    eid = int(p['experiment'])
                    snapshots = []
                    for key in ('left', 'right'):
                        r = query('SELECT snapshot FROM revisions WHERE id=? AND experiment_id=?', (int(p[key]), eid))
                        if not r:
                            raise ValueError('修订不存在')
                        snapshots.append({u['uid']: u['code'] for u in json.loads(r[0]['snapshot'])})
                    a, b = snapshots
                    counts = {'未变': 0, '新进': 0, '退出': 0, '换 vid': 0, '两次均在实验外': 0, '新 uid': 0}
                    for uid in a.keys() | b.keys():
                        x, y = a.get(uid, -1), b.get(uid, -1)
                        key = '新 uid' if uid not in a else '两次均在实验外' if x <= 0 and y <= 0 else '新进' if x <= 0 < y else '退出' if y <= 0 < x else '未变' if x == y else '换 vid'
                        counts[key] += 1
                    return self.send(counts)
            name = {'/': 'index.html', '/app.js': 'app.js', '/style.css': 'style.css'}.get(url.path)
            if not name:
                return self.send({'error': 'Not found'}, 404)
            mime = {'html': 'text/html', 'js': 'text/javascript', 'css': 'text/css'}[name.split('.')[-1]]
            self.send((ROOT / 'static' / name).read_bytes(), content_type=mime + '; charset=utf-8')
        except (ValueError, KeyError, TypeError) as exc:
            self.send({'error': str(exc)}, 400)

    def do_POST(self):
        # Only same-origin browser mutations are accepted by the local server.
        origin = self.headers.get('Origin')
        if origin and origin != 'http://' + self.headers.get('Host', ''):
            return self.send({'error': 'Origin rejected'}, 403)
        try:
            length = int(self.headers.get('Content-Length', 0))
            if length > 10_000_000:
                raise ValueError('导入文件过大')
            data = json.loads(self.rfile.read(length))
            with LOCK, DB:
                result = mutate(self.path.removeprefix('/api/'), data)
            self.send(result)
        except (ValueError, KeyError, TypeError, sqlite3.Error) as exc:
            self.send({'error': str(exc)}, 400)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('--db', default=str(ROOT / 'partition-bias-lab.sqlite3'))
    args = parser.parse_args()
    init_db(args.db)
    print(f'partition-bias-lab demo: http://127.0.0.1:{args.port}', flush=True)
    ThreadingHTTPServer(('127.0.0.1', args.port), Handler).serve_forever()
