#include "metrics.h"

#include <cstdio>
#include <fstream>
#include <utility>

void multilora_metrics::record(multilora_request_metric m) {
    records_.push_back(std::move(m));
}

bool multilora_metrics::dump_csv(const std::string & path) const {
    std::ofstream f(path);
    if (!f) {
        std::fprintf(stderr, "multilora_metrics::dump_csv: failed to open '%s'\n", path.c_str());
        return false;
    }
    f << "id,adapter_id,arrival,first_token,finish,"
         "n_input_tokens,n_output_tokens,cache_hit,acquire_ms\n";
    f.setf(std::ios::fixed);
    f.precision(6);
    for (const auto & r : records_) {
        f << r.id << ',' << r.adapter_id << ','
          << r.arrival_time << ',' << r.first_token_time << ',' << r.finish_time << ','
          << r.n_input_tokens << ',' << r.n_output_tokens << ','
          << (r.cache_hit ? 1 : 0) << ',' << r.acquire_ms << '\n';
    }
    return f.good();
}
