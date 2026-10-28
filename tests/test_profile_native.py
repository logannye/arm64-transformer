import pytest

from scripts.profile_native import parse_sample

SAMPLE = """Call graph:
    12 Thread_7: Main Thread
    + 12 _dyld_start (in dyld)
    100 Thread_7 DispatchQueue_1: com.apple.main-thread
      100 start (in dyld)
        70 recursive_parent (in transformer-arm64)
          70 recursive_parent (in transformer-arm64)

Total number in stack (recursive counted multiple, when >=5):
        999 tf_linear_forward (in transformer-arm64)
Sort by top of stack, same collapsed (when >= 5):
        tf_linear_forward_sme  (in transformer-arm64)        30
        tf_linear_backward_sme  (in transformer-arm64)        40
        tf_linear_forward  (in unrelated-library)        5
        tf_attention_backward  (in transformer-arm64)        20
        _dyld_start  (in dyld)        12
Binary images description not available
"""


def test_exclusive_attribution_excludes_loader_and_inclusive_recursion():
    result = parse_sample(SAMPLE)
    assert result["all_thread_samples"] == 112
    assert result["post_loader_samples"] == 100
    assert result["startup_segment_samples_excluded"] == 12
    assert result["dense_linear_exclusive_samples"] == 70
    assert result["dense_linear_percent_of_post_loader_samples"] == 70
    assert len(result["collapsed_leaf_records"]) == 5


def test_requires_identifiable_process_segment():
    with pytest.raises(ValueError, match="post-loader"):
        parse_sample(SAMPLE.replace("DispatchQueue_1", "Unknown"))


def test_rejects_inconsistent_sample_counts():
    with pytest.raises(ValueError, match="exceed"):
        parse_sample(SAMPLE.replace("        40", "        140"))


def test_accepts_legacy_neon_names():
    result = parse_sample(SAMPLE.replace("_sme", ""))
    assert result["dense_linear_exclusive_samples"] == 70


def test_retains_unknown_address_leaves():
    raw = SAMPLE.replace(
        "Binary images description not available",
        "        ???  (in transformer-arm64) load address 0x1000 + 0x45 [0x1045]        7\n"
        "Binary Images:",
    )
    result = parse_sample(raw)
    assert result["unattributed_leaf_samples"] == 7
    assert result["dense_attribution_is_lower_bound"] is True
    assert result["collapsed_leaf_records"][-1]["samples"] == 7
