import pytest

from semif_phase1.permutation import (
    aggregate_logits,
    align_logits,
    generate_permutations,
    permute_row,
)


ROW = {
    "id": "perm-test-1",
    "state": "Customer requested a password reset link.",
    "question": "Which action should be taken?",
    "options": [
        {"id": "opt_email", "description": "Send verification email."},
        {"id": "opt_sms", "description": "Send SMS code."},
        {"id": "opt_call", "description": "Initiate automated phone call."},
    ],
}


def test_generate_permutations_binary():
    perms = generate_permutations(2, max_perms=2)
    assert perms == [[0, 1], [1, 0]]


def test_generate_permutations_cyclic():
    perms = generate_permutations(3, max_perms=3)
    assert perms == [[0, 1, 2], [1, 2, 0], [2, 0, 1]]

    perms_limited = generate_permutations(3, max_perms=2)
    assert perms_limited == [[0, 1, 2], [1, 2, 0]]


def test_permute_row():
    perm = [2, 0, 1]
    p_row = permute_row(ROW, perm)
    assert p_row["id"] == ROW["id"]
    assert p_row["state"] == ROW["state"]
    assert p_row["question"] == ROW["question"]
    assert [opt["id"] for opt in p_row["options"]] == ["opt_call", "opt_email", "opt_sms"]

    with pytest.raises(ValueError, match="match options count"):
        permute_row(ROW, [0, 1])


def test_align_logits():
    # Options: [Opt0, Opt1]
    # Permutation: [1, 0] means Slot 0 has Opt1, Slot 1 has Opt0.
    # Suppose slot logits are [Slot0=9.0, Slot1=4.0].
    # Then Opt1 should get 9.0 and Opt0 should get 4.0 -> aligned: [4.0, 9.0]
    perm = [1, 0]
    slot_logits = [9.0, 4.0]
    aligned = align_logits(slot_logits, perm)
    assert aligned == [4.0, 9.0]


def test_aggregate_logits():
    run1 = [10.0, 4.0]
    run2 = [6.0, 8.0]
    mean_logits = aggregate_logits([run1, run2])
    assert mean_logits == [8.0, 6.0]


def test_position_bias_cancellation():
    """Mathematical verification: permutation ensembling cancels linear position bias.

    Suppose true semantic merit of Option 0 is 5.0, Option 1 is 4.0 (Opt0 should win).
    Suppose model has strong primacy bias: slot A always receives a +3.0 boost.

    - Order 1 [Opt0, Opt1]:
        Slot 0 (Opt0): 5.0 + 3.0 = 8.0
        Slot 1 (Opt1): 4.0 + 0.0 = 4.0
        Winner: Opt0 (8.0 vs 4.0)

    - Order 2 [Opt1, Opt0]:
        Slot 0 (Opt1): 4.0 + 3.0 = 7.0
        Slot 1 (Opt0): 5.0 + 0.0 = 5.0
        Winner: Opt1 (7.0 vs 5.0) -> FLIP! Without ensembling, order determines winner!

    - Permutation Ensembling:
        Run 1 aligned: [8.0, 4.0]
        Run 2 aligned: [5.0, 7.0]
        Ensemble mean: [(8.0 + 5.0)/2, (4.0 + 7.0)/2] = [6.5, 5.5]
        Winner: Opt0! Bias completely cancels out, restoring true winner.
    """
    perm1 = [0, 1]
    perm2 = [1, 0]

    # Slot logits under +3.0 slot A bias
    run1_slot = [8.0, 4.0]  # Opt0 at slot 0, Opt1 at slot 1
    run2_slot = [7.0, 5.0]  # Opt1 at slot 0, Opt0 at slot 1

    run1_aligned = align_logits(run1_slot, perm1)
    run2_aligned = align_logits(run2_slot, perm2)

    assert run1_aligned == [8.0, 4.0]
    assert run2_aligned == [5.0, 7.0]

    ensembled = aggregate_logits([run1_aligned, run2_aligned])
    assert ensembled == [6.5, 5.5]
    # Option 0 maintains true semantic superiority
    assert ensembled[0] > ensembled[1]
