#include "llama-mmap.h"

#include <cassert>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

static llama_weight_unit_slice slice(
        int64_t row_start,
        int64_t row_count,
        int32_t unit_id,
        uint32_t flags = 0) {
    llama_weight_unit_slice result;
    result.row_start = row_start;
    result.row_count = row_count;
    result.unit_id = unit_id;
    result.flags = flags;
    return result;
}

int main() {
    llama_weight_unit_plan_clear();
    const uint64_t initial_generation =
        llama_weight_unit_plan_generation();

    const std::vector<std::pair<
        std::string, llama_weight_unit_slice>> initial = {
        {"weight.a", slice(0, -1, 40)},
        {"weight.b", slice(0, 128, 81)},
        {"weight.b", slice(128, 128, 82)},
    };
    assert(llama_weight_unit_plan_replace(initial) == 2);
    const uint64_t populated_generation =
        llama_weight_unit_plan_generation();
    assert(populated_generation == initial_generation + 1);

    llama_weight_unit_slice queried[2];
    assert(llama_weight_unit_plan_query(
        "weight.a", queried, 2) == 1);
    assert(queried[0].unit_id == 40);
    assert(llama_weight_unit_plan_query(
        "weight.b", queried, 2) == 2);
    assert(queried[0].row_start == 0);
    assert(queried[1].row_start == 128);

    // Republishing identical bytes is a no-op: it neither changes a logical
    // weight nor advances the generation observed by backend caches.
    assert(llama_weight_unit_plan_replace(initial) == 0);
    assert(llama_weight_unit_plan_generation() ==
           populated_generation);

    // Change one weight while preserving the other. Only the affected
    // logical weight is counted in the runtime delta.
    const std::vector<std::pair<
        std::string, llama_weight_unit_slice>> changed = {
        {"weight.a", slice(0, 64, 41)},
        {"weight.a", slice(64, 64, 42)},
        {"weight.b", slice(0, 128, 81)},
        {"weight.b", slice(128, 128, 82)},
    };
    assert(llama_weight_unit_plan_replace(changed) == 1);
    assert(llama_weight_unit_plan_generation() ==
           populated_generation + 1);

    // Removal is also counted exactly once, and an absent query returns zero.
    const std::vector<std::pair<
        std::string, llama_weight_unit_slice>> remove_b = {
        {"weight.a", slice(0, 64, 41)},
        {"weight.a", slice(64, 64, 42)},
    };
    assert(llama_weight_unit_plan_replace(remove_b) == 1);
    assert(llama_weight_unit_plan_query(
        "weight.b", queried, 2) == 0);

    const std::vector<std::pair<
        std::string, llama_weight_unit_slice>> empty;
    assert(llama_weight_unit_plan_replace(empty) == 1);
    assert(llama_weight_unit_plan_query(
        "weight.a", queried, 2) == 0);
    return 0;
}
