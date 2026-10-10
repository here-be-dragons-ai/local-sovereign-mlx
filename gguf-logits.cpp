// Full next-token logits from a GGUF build, for measure-quality.py `gguf` (sovereign-models#18).
//
// llama-perplexity --kl-divergence compares two llama.cpp runs on its own
// chunking of a text file. We need the logits on exactly our token sequences,
// against our MLX reference, so this reads token ids on stdin and writes the
// raw float32 logits to stdout; measure-quality.py computes KL itself.
//
// Protocol (little-endian int32):
//   out: n_vocab once at start
//   in:  n, mode, n token ids        (n == 0 ends)
//        mode 0: logits for positions 0 .. n-2  ((n-1) x n_vocab float32)
//        mode 1: logits for the last position   (n_vocab float32)
//        mode 2: no forward; each token's text   (int32 length + bytes per token)
//
// Build against a llama.cpp build tree (here the Kolibri branch, which also
// runs Apertus):
//   L=~/src/llama.cpp-kolibri
//   clang++ -std=c++17 -O2 gguf-logits.cpp -I$L/include -I$L/ggml/include \
//     -L$L/build/bin -lllama -lggml -lggml-base -Wl,-rpath,$L/build/bin -o ~/src/mlx/bin/gguf-logits
//
//   gguf-logits <model.gguf> <n_ctx>

#include "llama.h"

#include <cstdio>
#include <cstdlib>
#include <vector>

static bool read_i32(int32_t * v, size_t n) { return fread(v, sizeof(int32_t), n, stdin) == n; }

int main(int argc, char ** argv) {
    if (argc != 3) {
        fprintf(stderr, "usage: %s <model.gguf> <n_ctx>\n", argv[0]);
        return 1;
    }
    llama_backend_init();
    llama_log_set([](ggml_log_level level, const char * text, void *) {
        if (level >= GGML_LOG_LEVEL_WARN) fputs(text, stderr);
    }, nullptr);

    auto mp = llama_model_default_params();
    mp.n_gpu_layers = -1;
    llama_model * model = llama_model_load_from_file(argv[1], mp);
    if (!model) return 1;

    const int n_batch = 512;
    auto cp = llama_context_default_params();
    cp.n_ctx = atoi(argv[2]);
    cp.n_batch = n_batch;
    cp.n_ubatch = n_batch;
    cp.n_seq_max = 1;
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) return 1;

    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int32_t n_vocab = llama_vocab_n_tokens(vocab);
    fwrite(&n_vocab, sizeof n_vocab, 1, stdout);
    fflush(stdout);

    llama_batch batch = llama_batch_init(n_batch, 0, 1);
    std::vector<int32_t> toks;
    int32_t head[2];
    char piece[1024];
    while (read_i32(head, 2) && head[0] > 0) {
        const int32_t n = head[0], mode = head[1];
        toks.resize(n);
        if (!read_i32(toks.data(), n)) return 1;
        if (mode == 2) {
            for (int32_t t : toks) {
                int32_t len = llama_token_to_piece(vocab, t, piece, sizeof piece, 0, true);
                if (len < 0) len = 0;
                fwrite(&len, sizeof len, 1, stdout);
                fwrite(piece, 1, len, stdout);
            }
            fflush(stdout);
            continue;
        }
        if ((uint32_t) n > llama_n_ctx(ctx)) {
            fprintf(stderr, "sequence of %d tokens exceeds n_ctx %u\n", n, llama_n_ctx(ctx));
            return 1;
        }
        llama_memory_clear(llama_get_memory(ctx), true);
        for (int32_t a = 0; a < n; a += n_batch) {
            const int32_t b = a + n_batch < n ? a + n_batch : n;
            batch.n_tokens = b - a;
            for (int32_t i = a; i < b; i++) {
                const int32_t k = i - a;
                batch.token[k] = toks[i];
                batch.pos[k] = i;
                batch.n_seq_id[k] = 1;
                batch.seq_id[k][0] = 0;
                batch.logits[k] = mode == 0 ? i < n - 1 : i == n - 1;
            }
            if (llama_decode(ctx, batch) != 0) {
                fprintf(stderr, "llama_decode failed at position %d\n", a);
                return 1;
            }
            for (int32_t k = 0; k < b - a; k++) {
                if (batch.logits[k]) fwrite(llama_get_logits_ith(ctx, k), sizeof(float), n_vocab, stdout);
            }
        }
        fflush(stdout);
    }
    llama_batch_free(batch);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
