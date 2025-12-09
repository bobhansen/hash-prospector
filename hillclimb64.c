#define _POSIX_C_SOURCE 200112L
#define WIN32_LEAN_AND_MEAN
#include <math.h>
#include <ctype.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define HASHN       2    // number of multiplies in hash
#define SHIFT_RANGE 2    // radius of shift search
#define CONST_RANGE 2    // radius of const search
#define QUALITY     18   // 2^N iterations of estimate samples
#define THRESHOLD   1.95 // regenerate anything lower than this estimate
// #define THRESHOLD   0.08 // regenerate anything lower than this estimate

static int optind = 1;
static int opterr = 1;
static int optopt;
static char *optarg;
static int
getopt(int argc, char * const argv[], const char *optstring)
{
    static int optpos = 1;
    const char *arg;
    (void)argc;
    /* Reset? */
    if (optind == 0) {
        optind = 1;
        optpos = 1;
    }
    arg = argv[optind];
    if (arg && strcmp(arg, "--") == 0) {
        optind++;
        return -1;
    } else if (!arg || arg[0] != '-' || !isalnum(arg[1])) {
        return -1;
    } else {
        const char *opt = strchr(optstring, arg[optpos]);
        optopt = arg[optpos];
        if (!opt) {
            if (opterr && *optstring != ':')
                fprintf(stderr, "%s: illegal option: %c\n", argv[0], optopt);
            return '?';
        } else if (opt[1] == ':') {
            if (arg[optpos + 1]) {
                optarg = (char *)arg + optpos + 1;
                optind++;
                optpos = 1;
                return optopt;
            } else if (argv[optind + 1]) {
                optarg = (char *)argv[optind + 1];
                optind += 2;
                optpos = 1;
                return optopt;
            } else {
                if (opterr && *optstring != ':')
                    fprintf(stderr,
                            "%s: option requires an argument: %c\n",
                            argv[0], optopt);
                return *optstring == ':' ? ':' : '?';
            }
        } else {
            if (!arg[++optpos]) {
                optind++;
                optpos = 1;
            }
            return optopt;
        }
    }
}

#if defined(__unix__)
#include <sys/time.h>
uint64_t
uepoch(void)
{
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return 1000000LL * tv.tv_sec + tv.tv_usec;
}
#elif defined(_WIN32)
#include <windows.h>
uint64_t
uepoch(void)
{
    FILETIME ft;
    GetSystemTimeAsFileTime(&ft);
    uint64_t tt = ft.dwHighDateTime;
    tt <<= 32;
    tt |= ft.dwLowDateTime;
    tt /=10;
    tt -= UINT64_C(11644473600000000);
    return tt;
}
#endif

static uint64_t
rand64(uint64_t s[4])
{
    uint64_t x = s[1] * 5;
    uint64_t r = ((x << 7) | (x >> 57)) * 9;
    uint64_t t = s[1] << 17;
    s[2] ^= s[0];
    s[3] ^= s[1];
    s[1] ^= s[2];
    s[0] ^= s[3];
    s[2] ^= t;
    s[3] = (s[3] << 45) | (s[3] >> 19);
    return r;
}

struct hash {
    uint64_t c[HASHN];
    char s[HASHN + 1];
};

static void
hash_gen(struct hash *h, uint64_t rng[4])
{
    for (int i = 0; i < HASHN; i++)
        h->c[i] = rand64(rng) | 1ULL;
    for (int i = 0; i <= HASHN; i++)
        h->s[i] = 32;
}

static int
hash_equal(const struct hash *a, const struct hash *b)
{
    for (int i = 0; i < HASHN; i++) {
        if (a->c[i] != b->c[i])
            return 0;
        if (a->s[i] != b->s[i])
            return 0;
    }
    return a->s[HASHN] == b->s[HASHN];
}

static void
hash_print(const struct hash *h)
{
    putchar('[');
    for (int i = 0; i < HASHN; i++)
        printf("%2d %016llx ", h->s[i], (unsigned long long)h->c[i]);
    printf("%2d]", h->s[HASHN]);
    fflush(stdout);
}

static int
hash_parse(struct hash *h, char *str)
{
    long s;
    unsigned long long c;
    char *end, *tok;
    if (*str != '[')
        return 0;
    str++;
    for (int i = 0; i < HASHN; i++) {
        tok = strtok(i ? 0 : str, " ");
        s = strtol(tok, &end, 10);
        if (s < 1 || s > 63 || !(*end == 0 || *end == ' '))
            return 0;
        h->s[i] = s;
        tok = strtok(0, " ");
        c = strtoull(tok, &end, 16);
        if (c > 0xffffffffffffffffULL || !(*end == 0 || *end == ' '))
            return 0;
        h->c[i] = c;
    }
    tok = strtok(0, "]");
    s = strtol(tok, &end, 10);
    if (s < 1 || s > 63 || *end)
        return 0;
    h->s[HASHN] = s;
    return 1;
}

static uint64_t
hash(const struct hash *h, uint64_t x)
{
    for (int i = 0; i < HASHN; i++) {
        x ^= x >> h->s[i];
        x *= h->c[i];
    }
    x ^= x >> h->s[HASHN];
    return x;
}

static double
estimate_bias64(const struct hash *f, uint64_t rng[4])
{
    long n = 1L << QUALITY;
    long bins[64][64] = {{0}};
    for (long i = 0; i < n; i++) {
        uint64_t x = rand64(rng);
        uint64_t h0 = hash(f, x);
        for (int j = 0; j < 64; j++) {
            uint64_t bit = UINT64_C(1) << j;
            uint64_t h1 = hash(f, x ^ bit);
            uint64_t set = h0 ^ h1;
            for (int k = 0; k < 64; k++)
                bins[j][k] += (set >> k) & 1;
        }
    }
    double mean = 0;
    for (int j = 0; j < 64; j++) {
        for (int k = 0; k < 64; k++) {
            double diff = (bins[j][k] - n / 2) / (n / 2.0);
            mean += (diff * diff) / (64 * 64);
        }
    }
    return sqrt(mean) * 1000.0;
}


/**
 * Use splitmix to ensure that we're not re-exploring the same values while we do
 * a larger estimation check.  Splitmix is a permutation, so there will be no collisions
 * for different inputs.
 */
static uint64_t
splitmix64(uint64_t z)
{
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}


#define DEEPER_SPLIT 32  // must be power of two
#define DEEPER_QUALITY 32

struct deeper_bias_result {
    double rms_deviation;
    double max_bias_95_conf_min;
    double max_bias_95_conf_max;
};

static struct deeper_bias_result
deeper_bias64(const struct hash *f, int quality)
{
    int i; // declare here to work around Visual Studio issue
    long long bins[64][64] = {{0}};
    /**
     * We can't explore all 2^64 inputs, so we'll explore 2^quality of them, parallelizable
     * across DEEPER_SPLIT threads.
     */
    const uint64_t range = (1ULL << (quality)) / DEEPER_SPLIT;
    #pragma omp parallel for
    for (i = 0; i < DEEPER_SPLIT; i++) {
        long long b[64][64] = {{0}};
        uint64_t start = (uint64_t)i * range;
        uint64_t end = start + range - 1;
        for (uint64_t idx = start; idx <= end; idx++) {
            uint64_t x = splitmix64(idx);
            uint64_t h0 = hash(f, x);
            for (int j = 0; j < 64; j++) {
                uint64_t bit = UINT64_C(1) << j;
                uint64_t h1 = hash(f, x ^ bit);
                uint64_t set = h0 ^ h1;
                for (int k = 0; k < 64; k++)
                    b[j][k] += (set >> k) & 1;
            }
        }
        #pragma omp critical
        for (int j = 0; j < 64; j++)
            for (int k = 0; k < 64; k++)
                bins[j][k] += b[j][k];
    }
    const long long expected_value = (1ULL << quality) / 2;
    double deviation_ratio_sq = 0.0;
    double max_deviation = 0.0;
    for (int j = 0; j < 64; j++) {
        for (int k = 0; k < 64; k++) {
            double deviation = bins[j][k] - expected_value;
            /* Max bias statistic */
            max_deviation = fmax(max_deviation, fabs(deviation));
            /* Normalize by expected value first to keep numbers near 1.0 */
            double diff = deviation / (double)expected_value;
            deviation_ratio_sq += (diff * diff) / (64 * 64);
        }
    }
    
    // With 64*64 trials, the 95% confidence interval for the null hypothesis is
    //   for maximum deviation is SD * qnorm((1 + 0.95^(1/4096)) / 2) = SD * 4.3683
    //   SD for unbiased binmonial is sqrt(n) / 2
    const double sd = sqrt(1ULL << quality) / 2.0;
    const double deviation_confidence_interval = 1.96 * sd;

    // Only report the variance confidence interval if the maximum deviation exceeds 
    //    the null hypothesis boundary
    struct deeper_bias_result result;
    if (max_deviation > 4.3683 * sd) {
        result.max_bias_95_conf_min = (max_deviation - deviation_confidence_interval) / expected_value;
        result.max_bias_95_conf_max = (max_deviation + deviation_confidence_interval) / expected_value;
    } else {
        result.max_bias_95_conf_min = 0.0;
        result.max_bias_95_conf_max = (4.3683 * sd) / expected_value;
    }

    result.rms_deviation = sqrt(deviation_ratio_sq)* 1000.0;
    return result;
}

static void 
bias_result_print(const char * prefix, struct deeper_bias_result res, const char *suffix)
{
    printf("%s[%.17g%% - %.17g%%] %.17g %s",
                prefix,
                res.max_bias_95_conf_min * 100.0,
                res.max_bias_95_conf_max * 100.0, 
                res.rms_deviation,
                suffix);
}

static void
hash_gen_strict(struct hash *h, uint64_t rng[4])
{
    double bias = THRESHOLD + 1;
    do {
        hash_gen(h, rng);
        bias = estimate_bias64(h, rng);
        printf("generated bias %.17g\n", bias);
    } while ( bias > THRESHOLD);
}

static uint64_t
load64(const void *buf)
{
    const unsigned char *p = buf;
    return (uint64_t)p[0] <<  0 |
           (uint64_t)p[1] <<  8 |
           (uint64_t)p[2] << 16 |
           (uint64_t)p[3] << 24 |
           (uint64_t)p[4] << 32 |
           (uint64_t)p[5] << 40 |
           (uint64_t)p[6] << 48 |
           (uint64_t)p[7] << 56;
}

static uint64_t
mix64(uint64_t x, uint64_t y)
{
    uint64_t r = 0x2b8a130976726633 * x - 0xb28cbd28446adb17 * y;
    r ^= r >> 32;
    return r;
}

static uint64_t
hash64(uint64_t x, uint64_t m)
{
    x *= m;
    x ^= x >> 32;
    return x;
}

static void
mix64x4(uint64_t x[4])
{
    uint64_t i = 0xf81db9ba6dabee4e;
    uint64_t m = 0xb1d9e3fbc08321db;
    x[0] = hash64(x[0] + 0x347534cdcf0982b6, m);
    x[1] = hash64(x[1] + 0x975e2ee8f0f23aa8, m += i);
    x[2] = hash64(x[2] + 0x7baf736c6c769a0b, m += i);
    x[3] = hash64(x[3] + 0x884afc96accb90d9, m += i);
    #define ROUND64(a, b, c, d) \
        x[b] = mix64(hash64(x[a], m += i), x[b]); \
        x[c] = mix64(hash64(x[a], m += i), x[c]); \
        x[d] = mix64(hash64(x[a], m += i), x[d])
    ROUND64(0, 1, 2, 3);
    ROUND64(1, 0, 2, 3);
    ROUND64(2, 0, 1, 3);
    ROUND64(3, 0, 1, 3);
    #undef ROUND64
}

static void
rng_init(uint64_t rng[4])
{
    void *p = malloc(1024L * 1024);
    rng[0] = uepoch();
    rng[1] = (uint64_t)rng_init;
    rng[2] = (uint64_t)rng;
    rng[3] = (uint64_t)p;
    free(p);
    mix64x4(rng);
}

/* Modular multiplicative inverse (64-bit) */
static uint64_t
modinv64(uint64_t x)
{
    uint64_t a = x;
    x += x - a * x * x;
    x += x - a * x * x;
    x += x - a * x * x;
    x += x - a * x * x;
    x += x - a * x * x;
    x += x - a * x * x;  // 6 iterations for 64-bit
    return x;
}

static void
usage(FILE *f)
{
    fprintf(f, "usage: hillclimb [-EhIqs] [-p INIT] [-x SEED]\n");
    fprintf(f, "  -E       Evaluate given pattern (-p)\n");
    fprintf(f, "  -h       Print this message and exit\n");
    fprintf(f, "  -I       Invert given pattern (-p) an quit\n");
    fprintf(f, "  -p INIT  Provide an initial hash function\n");
    fprintf(f, "  -q       Print less information (quiet)\n");
    fprintf(f, "  -s       Quit after finding a local minima\n");
    fprintf(f, "  -x SEED  Seed PRNG from a string (up to 32 bytes)\n");
}

int
main(int argc, char **argv)
{
    int seeded = 0;
    uint64_t rng[4];
    struct hash cur, last = {0};
    int generate = 1;
    int one_shot = 0;
    int quiet = 0;
    int invert = 0;
    int evaluate = 0;
    struct deeper_bias_result cur_score = {-1.0, 0.0, 0.0};

    int option;
    while ((option = getopt(argc, argv, "EhIp:qsx:")) != -1) {
        switch (option) {
            case 'E': {
                evaluate = 1;
            } break;
            case 'h': {
                usage(stdout);
                exit(EXIT_SUCCESS);
            } break;
            case 'I': {
                invert = 1;
            } break;
            case 'p': {
                if (!hash_parse(&cur, optarg)) {
                    fprintf(stderr, "hillclimb: invalid pattern: %s\n", optarg);
                    exit(EXIT_FAILURE);
                }
                generate = 0;
            } break;
            case 'q': {
                quiet++;
            } break;
            case 's': {
                one_shot = 1;
            } break;
            case 'x': {
                unsigned char buf[32] = {0};
                size_t len = strlen(optarg);
                if (len > sizeof(buf)) {
                    fprintf(stderr, "hillclimb: seed too long (> 32 bytes)\n");
                    exit(EXIT_FAILURE);
                }
                memcpy(buf, optarg, len);
                rng[0] = load64(buf +  0);
                rng[1] = load64(buf +  8);
                rng[2] = load64(buf + 16);
                rng[3] = load64(buf + 24);
                mix64x4(rng);
                seeded = 1;
            } break;
            default:
                usage(stderr);
                exit(EXIT_FAILURE);
        }
    }

    if (invert) {
        if (generate) {
            fprintf(stderr, "hillclimb: -I requires -p\n");
            exit(EXIT_FAILURE);
        }
        printf("uint64_t\nhash_r(uint64_t x)\n{\n");
        for (int i = 0; i < HASHN * 2 + 1; i++) {
            switch (i & 1) {
                case 0: {
                    int s = HASHN - i / 2;
                    printf("    x ^=");
                    for (int i = cur.s[s]; i < 64; i += cur.s[s])
                        printf(" %sx >> %d", i == cur.s[s] ? "" : "^ ", i);
                    printf(";\n");
                } break;
                case 1: {
                    int c = HASHN - (i + 1) / 2;
                    unsigned long long inv = modinv64(cur.c[c]);
                    printf("    x *= 0x%016llx;\n", inv);
                } break;
            }
        }
        printf("    return x;\n}\n");
        exit(EXIT_SUCCESS);
    }

    if (evaluate) {
        if (generate) {
            fprintf(stderr, "hillclimb: -E requires -p\n");
            exit(EXIT_FAILURE);
        }
        hash_print(&cur);
        bias_result_print("", deeper_bias64(&cur, DEEPER_QUALITY), "\n");
        exit(EXIT_SUCCESS);
    }

    if (!seeded)
        rng_init(rng);

    if (generate)
        hash_gen_strict(&cur, rng);

    for (;;) {
        int found = 0;
        struct hash best;
        struct deeper_bias_result best_score;

        if (quiet < 2)
            hash_print(&cur);
        if (cur_score.rms_deviation < 0)
            cur_score = deeper_bias64(&cur, DEEPER_QUALITY);
        if (quiet < 2) {
            bias_result_print(" = ", cur_score, "\n");
        }

        best = cur;
        best_score = cur_score;

        /* Explore around shifts */
        for (int i = 0; i <= HASHN; i++) {
            /* In theory the shift could drift above 63 or below 1, but
             * in practice it would never get this far since these would
             * be terrible hashes.
             */
            for (int d = -SHIFT_RANGE; d <= +SHIFT_RANGE; d++) {
                if (d == 0) continue;
                struct hash tmp = cur;
                tmp.s[i] += d;
                if (hash_equal(&tmp, &last)) continue;
                if (quiet <= 0) {
                    printf("  ");
                    hash_print(&tmp);
                }
                struct deeper_bias_result score = deeper_bias64(&tmp, DEEPER_QUALITY);
                if (quiet <= 0) {
                    bias_result_print(" = ", score, "\n");
                }
                if (score.rms_deviation < best_score.rms_deviation) {
                    best_score = score;
                    best = tmp;
                    found = 1;
                }
            }
        }

        /* Explore around constants */
        for (int i = 0; i < HASHN; i++) {
            for (int d = -CONST_RANGE; d <= +CONST_RANGE; d += 2) {
                if (d == 0) continue;
                struct hash tmp = cur;
                tmp.c[i] += d;
                if (hash_equal(&tmp, &last)) continue;
                if (quiet <= 0) {
                    printf("  ");
                    hash_print(&tmp);
                }
                struct deeper_bias_result score = deeper_bias64(&tmp, DEEPER_QUALITY);
                if (quiet <= 0) {
                    bias_result_print(" = ", score, "\n");
                }
                if (score.rms_deviation < best_score.rms_deviation) {
                    best_score = score;
                    best = tmp;
                    found = 1;
                }
            }
        }

        if (found) {
            /* Move to the lowest item found */
            if (quiet < 1)
                puts("CLIMB");
            last = cur;
            cur = best;
            cur_score = best_score;
        } else if (one_shot) {
            /* Hit local minima, exit */
            if (quiet < 1)
                puts("DONE");
            hash_print(&cur);
            bias_result_print(" = ", cur_score, "\n");
            break;
        } else {
            /* Hit local minima, reset */
            if (quiet < 1)
                puts("RESET");
            hash_print(&cur);
            bias_result_print(" = ", cur_score, "\n");
            last.s[0] = 0; // set to invalid
            hash_gen_strict(&cur, rng);
            cur_score = (struct deeper_bias_result){-1.0, 0.0, 0.0};
        }
    }
}