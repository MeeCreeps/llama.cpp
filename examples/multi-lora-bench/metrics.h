#pragma once

#include <cstddef>
#include <string>
#include <vector>

// Per-request metric record. Times are seconds since experiment start (t0).
//
// CSV schema is the spec target for plot_results.py / summarize.py
// (see IMPLEMENTATION_GUIDE.md section 4 M4):
//   id, adapter_id, arrival, first_token, finish,
//   n_input_tokens, n_output_tokens, cache_hit, acquire_ms
//
// `acquire_ms` is the wall time spent inside AdapterPool::acquire
// (zero on cache hit, ~adapter_load_time on miss). The spec acceptance
// for M1 explicitly calls out "cold vs hot acquire time can be split
// from metric" — that is what this column is for.
struct multilora_request_metric {
    std::string id;
    std::string adapter_id;
    double      arrival_time      = 0.0;
    double      first_token_time  = 0.0;
    double      finish_time       = 0.0;
    size_t      n_input_tokens    = 0;
    size_t      n_output_tokens   = 0;
    bool        cache_hit         = false;
    double      acquire_ms        = 0.0;
};

class multilora_metrics {
public:
    void record(multilora_request_metric m);

    // Write CSV (with header row) to path. Returns true on success.
    bool dump_csv(const std::string & path) const;

    size_t size() const { return records_.size(); }
    const std::vector<multilora_request_metric> & records() const { return records_; }

private:
    std::vector<multilora_request_metric> records_;
};
