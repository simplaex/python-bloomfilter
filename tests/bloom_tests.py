import os
import uuid

import fvnbloom as bfjs


def test_add_test():
    total = 1000
    uuids = []

    bf = bfjs.create_empty(capacity=total, error_rate=0.01)

    for i in range(total):
        did = uuid.uuid4().hex
        uuids.append(did)
        bf.add(did)

    for did in uuids:
        assert did in bf


def test_union():
    total = 1000
    strings1 = []
    bf1 = bfjs.create_empty(capacity=2 * total, error_rate=0.01)

    for i in range(total):
        did = uuid.uuid4().hex
        strings1.append(did)
        bf1.add(did)

    strings2 = []
    bf2 = bfjs.create_empty(capacity=2 * total, error_rate=0.01)

    for i in range(total):
        did = uuid.uuid4().hex
        strings2.append(did)
        bf2.add(did)

    bf = bf1.union(bf2)

    for did in strings1 + strings2:
        assert did in bf


def test_save_load():
    total = 1000
    uuids = []

    bf = bfjs.create_empty(capacity=total, error_rate=0.01)

    for i in range(total):
        did = uuid.uuid4().hex
        uuids.append(did)
        bf.add(did)

    path = 'test.json.bloom'

    try:
        bf.save(path)
        bf_loaded = bfjs.load(path)

        for did in uuids:
            assert did in bf_loaded

    except:
        assert False
    finally:
        os.remove(path)


def test_error_rate():
    total = 10000
    uuids = []

    bf = bfjs.create_empty(capacity=total, error_rate=0.01)

    for i in range(total):
        did = uuid.uuid4().hex
        uuids.append(did)
        bf.add(did)

    fps = 0
    for i in range(total):
        did = uuid.uuid4().hex
        if did in bf:
            fps = fps + 1

    fpr = fps / total

    assert abs(fpr - 0.01) < 0.005