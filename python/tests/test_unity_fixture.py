from tests.conftest import requires_unity

@requires_unity
def test_fixture_gives_exact_swarm_size(unity_worker):
    snap = unity_worker.snapshot(0)
    assert snap is not None
    assert snap["node"].shape[0] == 6, "fixture asked for exactly 6 robots"

@requires_unity
def test_fixture_reuses_the_player_and_resets_state(unity_players):
    w1 = unity_players.get(num_arenas=1, min_bots=6, max_bots=6)
    w1.step_count[0] = 999
    w2 = unity_players.get(num_arenas=1, min_bots=6, max_bots=6)
    assert w2 is w1, "same settings should reuse the same player"
    assert w2.step_count.get(0, 0) == 0, "state must be cleared between tests"

@requires_unity
def test_fixture_can_vary_arena_count(unity_players):
    w = unity_players.get(num_arenas=3, min_bots=4, max_bots=4)
    counts = [w.snapshot(k)["node"].shape[0] for k in range(3)]
    assert counts == [4, 4, 4], counts
