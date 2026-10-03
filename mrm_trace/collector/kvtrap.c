/*
 * kvtrap — capture the engine's KV cache allocation so it can be identified
 * by address rather than guessed.
 *
 * Why this exists: llama.cpp carves the KV cache out of a larger ggml backend
 * arena, so it has no mapping of its own and /proc/self/maps cannot isolate
 * it.  Measured on Llama-3.2-1B at ctx 2048 (expected KV 64 MiB): the new
 * anonymous mappings are 768 MiB, 512 MiB and 250 MiB — nothing to match.
 * The allocation itself, however, is exactly the size the GGUF geometry
 * implies, so interposing the allocator captures its base address exactly:
 *
 *     ctx  512  ->  posix_memalign 16,777,216
 *     ctx 1024  ->  mmap           33,554,432
 *     ctx 2048  ->  posix_memalign 67,108,864
 *     ctx 4096  ->  posix_memalign 134,217,728
 *
 * Both posix_memalign and mmap are needed: glibc routes the same request
 * differently depending on size.
 *
 * Writes "<pid> <fn> <size> <ptr>" per allocation >= MIN_LOG to the file named
 * by MRM_ALLOC_LOG.  Does nothing if that variable is unset, so the library is
 * inert unless a run asks for it.
 *
 * Build: gcc -shared -fPIC -O2 -o libkvtrap.so kvtrap.c -ldl
 */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <unistd.h>

/* Only allocations big enough to be a model buffer are of interest; logging
 * every malloc would perturb the very process being measured. */
#define MIN_LOG (1UL << 20)   /* 1 MiB */

static FILE *trap_log(void)
{
    static FILE *fp = NULL;
    static int tried = 0;
    if (!tried) {
        const char *path = getenv("MRM_ALLOC_LOG");
        if (path && *path) fp = fopen(path, "a");
        tried = 1;
    }
    return fp;
}

static void note(const char *fn, void *ptr, size_t size)
{
    if (size < MIN_LOG || !ptr) return;
    FILE *f = trap_log();
    if (!f) return;
    /* The pid lets the reader drop allocations made by perf itself. */
    fprintf(f, "%d %s %zu %p\n", (int)getpid(), fn, size, ptr);
    fflush(f);
}

void *malloc(size_t size)
{
    static void *(*real)(size_t);
    if (!real) real = dlsym(RTLD_NEXT, "malloc");
    void *p = real(size);
    note("malloc", p, size);
    return p;
}

void *aligned_alloc(size_t alignment, size_t size)
{
    static void *(*real)(size_t, size_t);
    if (!real) real = dlsym(RTLD_NEXT, "aligned_alloc");
    void *p = real(alignment, size);
    note("aligned_alloc", p, size);
    return p;
}

int posix_memalign(void **out, size_t alignment, size_t size)
{
    static int (*real)(void **, size_t, size_t);
    if (!real) real = dlsym(RTLD_NEXT, "posix_memalign");
    int rc = real(out, alignment, size);
    if (rc == 0) note("posix_memalign", *out, size);
    return rc;
}

void *mmap(void *addr, size_t size, int prot, int flags, int fd, off_t offset)
{
    static void *(*real)(void *, size_t, int, int, int, off_t);
    if (!real) real = dlsym(RTLD_NEXT, "mmap");
    void *p = real(addr, size, prot, flags, fd, offset);
    /* Anonymous only: a file mapping is the GGUF weights, already identified. */
    if (p != MAP_FAILED && fd == -1) note("mmap", p, size);
    return p;
}
