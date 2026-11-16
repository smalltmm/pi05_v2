import numpy as np
import pytest
from pi05_assists.handover_observation import DEFAULT_REFERENCE, HandoverObservation

@pytest.mark.parametrize('hz', [10, 100])
@pytest.mark.parametrize('reference', [DEFAULT_REFERENCE, DEFAULT_REFERENCE.with_name('items_handover_right_observation_20261113_075654.json'), DEFAULT_REFERENCE.with_name('items_handover_right_observation_20261113_092732.json'), DEFAULT_REFERENCE.with_name('items_handover_right_observation_20261113_100755.json')])
def test_observation_waits_for_transfer_then_only_overrides_right(hz, reference):
    helper = HandoverObservation(reference, 30, hz)
    state = np.zeros(14, dtype=np.float32)
    model = np.linspace(0, 0.5, 14, dtype=np.float32)
    np.testing.assert_array_equal(helper.apply(state, model, completed=False), model)
    assert helper.phase == 'idle'
    last = state[7:13].copy()
    for i in range(hz * 5):
        model[:7] += 0.001
        action = helper.apply(state, model, completed=i == 0)
        np.testing.assert_array_equal(action[:7], model[:7])
        assert np.max(np.abs(action[7:13] - last)) <= np.deg2rad(30) / hz + 1e-06
        assert action[13] == 1
        helper.commit(action)
        last = action[7:13].copy()
    assert helper.phase == 'holding'
    np.testing.assert_allclose(last, helper.goal)

def test_model_input_only_replaces_right_after_observation_starts():
    helper = HandoverObservation(DEFAULT_REFERENCE, 30, 100)
    state = np.arange(14, dtype=np.float32) / 20
    np.testing.assert_array_equal(helper.model_state(state), state)
    helper.apply(state, state, completed=True)
    actual = state.copy()
    for phase in ('moving', 'holding'):
        helper.phase = phase
        supplied = helper.model_state(actual)
        np.testing.assert_array_equal(supplied[:7], actual[:7])
        np.testing.assert_array_equal(supplied[7:], helper.model_right)
        np.testing.assert_array_equal(actual, state)
