#include "elastic_granularity.h"

#include <cassert>
#include <cstdio>

int main() {
    using elastic::granularity_partition_rows;

    {
        elastic::granularity_mode mode =
            elastic::granularity_mode::TENSOR;
        assert(elastic::granularity_mode_from_string("cut", mode));
        assert(mode == elastic::granularity_mode::CUT);
        assert(!elastic::granularity_mode_from_string("invalid", mode));

        elastic::granularity_config config;
        config.mode = elastic::granularity_mode::MULTI;
        config.multi_tensors = 3;
        config.cut_parts = 2;
        config.explicitly_enabled = true;
        elastic::granularity_runtime_apply(config);
        const auto state = elastic::granularity_runtime_get();
        assert(state.plan_override);
        assert(state.config.mode == elastic::granularity_mode::MULTI);
        assert(state.config.multi_tensors == 3);
        elastic::granularity_runtime_clear();
    }

    {
        const auto p = granularity_partition_rows(512, 2, 16);
        assert(p.size() == 2);
        assert(p[0].row_start == 0 && p[0].row_count == 256);
        assert(p[1].row_start == 256 && p[1].row_count == 256);
    }
    {
        const auto p = granularity_partition_rows(1000, 3, 16);
        assert(p.size() == 3);
        int64_t next = 0;
        int64_t total = 0;
        for (size_t i = 0; i < p.size(); ++i) {
            assert(p[i].row_start == next);
            assert(p[i].row_count > 0);
            if (i + 1 < p.size()) assert(p[i].row_count % 16 == 0);
            next += p[i].row_count;
            total += p[i].row_count;
        }
        assert(total == 1000);
    }
    assert(granularity_partition_rows(16, 2, 16).empty());
    assert(granularity_partition_rows(0, 2, 1).empty());
    assert(granularity_partition_rows(512, 1, 1).empty());

    std::puts("test_elastic_granularity: OK");
    return 0;
}
