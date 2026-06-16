// runtime/elastic_profile_writer.h
//
// Lightweight CSV profile writer for elastic memory experiments.
// Enabled only when GGML_ELASTIC_PROFILE_CSV points to an output path.

#pragma once

#include <cstddef>
#include <cstdint>

namespace elastic {

struct profile_record {
    const char * backend = "";
    const char * kind    = "";
    const char * name    = "";
    const char * op      = "";
    const char * quant   = "";
    int          token   = -1;
    int          op_id   = -1;
    int          weight_id = -1;
    int64_t      ne[4]   = {0, 0, 0, 0};
    size_t       bytes   = 0;
    double       ms      = 0.0;
    int          ok      = 1;
    const char * extra   = "";
};

bool profile_enabled();
void profile_write(const profile_record & rec);

}  // namespace elastic

