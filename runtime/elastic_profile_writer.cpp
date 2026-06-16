#include "elastic_profile_writer.h"

#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <string>
#include <sys/stat.h>

namespace elastic {
namespace {

struct writer_state {
    std::mutex mtx;
    FILE * file = nullptr;
    bool initialized = false;
    bool enabled = false;
};

writer_state & state() {
    static writer_state s;
    return s;
}

bool file_is_empty_or_missing(const char * path) {
    struct stat st;
    if (stat(path, &st) != 0) return true;
    return st.st_size == 0;
}

void csv_put(FILE * f, const char * s) {
    if (!s) s = "";
    std::fputc('"', f);
    for (const char * p = s; *p; ++p) {
        if (*p == '"') std::fputc('"', f);
        std::fputc(*p, f);
    }
    std::fputc('"', f);
}

void init_locked(writer_state & s) {
    if (s.initialized) return;
    s.initialized = true;

    const char * path = std::getenv("GGML_ELASTIC_PROFILE_CSV");
    if (!path || !*path) return;

    const bool need_header = file_is_empty_or_missing(path);
    s.file = std::fopen(path, "a");
    if (!s.file) {
        std::fprintf(stderr, "elastic profile: failed to open GGML_ELASTIC_PROFILE_CSV=%s\n", path);
        return;
    }
    s.enabled = true;
    if (need_header) {
        std::fprintf(s.file,
                     "backend,kind,name,op,quant,token,op_id,weight_id,"
                     "ne0,ne1,ne2,ne3,bytes,ms,ok,extra\n");
        std::fflush(s.file);
    }
}

}  // namespace

bool profile_enabled() {
    writer_state & s = state();
    std::lock_guard<std::mutex> lock(s.mtx);
    init_locked(s);
    return s.enabled;
}

void profile_write(const profile_record & rec) {
    writer_state & s = state();
    std::lock_guard<std::mutex> lock(s.mtx);
    init_locked(s);
    if (!s.enabled || !s.file) return;

    csv_put(s.file, rec.backend); std::fputc(',', s.file);
    csv_put(s.file, rec.kind);    std::fputc(',', s.file);
    csv_put(s.file, rec.name);    std::fputc(',', s.file);
    csv_put(s.file, rec.op);      std::fputc(',', s.file);
    csv_put(s.file, rec.quant);   std::fputc(',', s.file);
    std::fprintf(s.file, ",%d,%d,%d,%lld,%lld,%lld,%lld,%zu,%.6f,%d,",
                 rec.token,
                 rec.op_id,
                 rec.weight_id,
                 (long long) rec.ne[0],
                 (long long) rec.ne[1],
                 (long long) rec.ne[2],
                 (long long) rec.ne[3],
                 rec.bytes,
                 rec.ms,
                 rec.ok);
    csv_put(s.file, rec.extra);
    std::fputc('\n', s.file);
    std::fflush(s.file);
}

}  // namespace elastic

