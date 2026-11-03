from gausify.pairs import orbit_pairs, orbit_times


def test_orbit_pairs_open():
    p = orbit_pairs(5, 2, False)
    assert p == [(0, 1), (0, 2), (1, 2), (1, 3), (2, 3), (2, 4), (3, 4)]


def test_orbit_pairs_closed_wraps_and_dedups():
    p = orbit_pairs(6, 2, True)
    # limit = 3: offsets 1, 2 (16/32/64 too far); wrap-around pairs sorted
    assert p == [(0, 1), (0, 2), (1, 2), (1, 3), (2, 3), (2, 4), (3, 4), (3, 5), (4, 5), (0, 4), (0, 5), (1, 5)]
    assert len(set(p)) == len(p)


def test_orbit_pairs_long_offsets():
    p = orbit_pairs(100, 8, True)
    assert (0, 16) in p and (0, 32) in p and (0, 64) not in p  # 64 > 50 = limit
    offs = {min(b - a, 100 - (b - a)) for a, b in p}
    assert offs == {1, 2, 3, 4, 5, 6, 7, 8, 16, 32}
    assert len(p) == 100 * 10


def test_orbit_pairs_window_beyond_16():
    p = orbit_pairs(200, 20, False)
    offs = sorted({b - a for a, b in p})
    assert offs == list(range(1, 21)) + [32, 64]


def test_orbit_times():
    names = ["00000.jpg", "00003.jpg", "00007.jpg"]
    assert orbit_times(names, {"00000.jpg": 0.0, "00003.jpg": 0.5, "00007.jpg": 1.0}) == [0.0, 0.5, 1.0]
    # missing or not increasing -> number in the name
    assert orbit_times(names, None) == [0, 3, 7]
    assert orbit_times(names, {"00000.jpg": 0.0, "00003.jpg": 0.0, "00007.jpg": 1.0}) == [0, 3, 7]
    assert orbit_times(["b.jpg", "a.jpg", "c.jpg"]) == [0, 1, 2]
    assert orbit_times(["3.png", "2.png"]) == [0, 1]
