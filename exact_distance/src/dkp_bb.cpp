// dkp_bb.cpp - certified lower bound on the minimum weight of a logical
// operator of a CSS code by irreducible-cluster branch and bound.
//
//   d(H, G) = min { wt(x) : H x = 0 (mod 2), x not in rowspan(G) }
//
// A search state is a support S, a forbidden set F and the syndrome s(S).
// While the syndrome is nonzero the search picks the unsatisfied check with
// the fewest available variables (N(c) minus S and F) and branches over
// them; each tried variable is added to F for the later siblings. A
// zero-syndrome vector is reduced against a row-echelon basis of G: a nonzero
// remainder is a logical operator of weight |S| <= W, a zero remainder ends
// the branch. Two bounds prune: b1 = ceil(|s|/Delta) with Delta the largest
// column weight of H, and b2 = the size of a greedy set of unsatisfied checks
// whose available sets are pairwise disjoint.
//
// Roots: either every variable once as the least element of the support
// (initial F = all smaller variables), or orbit representatives of a
// verified code automorphism group.
//
// Input (text on stdin, written by exact_bb.py):
//   n mH mG W threads
//   mH lines: k v1 ... vk       variables of each row of H
//   mG lines: k v1 ... vk       variables of each row of G
//   R                           number of roots
//   R lines: root mode          mode 1 = forbid all variables < root, 0 = none,
//                               mode 2 = "root 2 k f1 ... fk": forbid exactly
//                               the k listed variables (orbit roots)
// Output: one line "FOUND w v1 ... vw" or "NONE W" (certified d(H,G) > W) or
// "TIMEOUT done/total" (the time limit stopped the search), plus a
// "STATS nodes secs" line. An "INFO ..." line on stdout records n, mH,
// rank(G), Delta, W, R, depth, the whole-frontier size and the slice.
// Malformed input exits with code 2; the time limit also covers the frontier
// construction.
//
//   g++ -O3 -march=x86-64-v3 -fopenmp -o dkp_bb dkp_bb.cpp
//   ./dkp_bb [time_limit_seconds] [slice slices [depth]] < problem.txt
// With "slice slices" given, only frontier subtrees with index % slices ==
// slice are searched, and NONE certifies this slice only.
// depth = how many levels the roots are expanded before the parallel loop
// (default: 4 for <= 32 roots, 3 for <= 400, else 2).
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif

static const int NW = 16;  // 1024-bit sets: enough for n, mH <= 1024

struct BS {
    uint64_t w[NW];
    void clear() { memset(w, 0, sizeof(w)); }
    bool test(int i) const { return (w[i >> 6] >> (i & 63)) & 1ULL; }
    void set(int i) { w[i >> 6] |= 1ULL << (i & 63); }
    bool any() const { for (int i = 0; i < NW; i++) if (w[i]) return true; return false; }
    int count() const { int c = 0; for (int i = 0; i < NW; i++) c += __builtin_popcountll(w[i]); return c; }
    void xor_with(const BS& o) { for (int i = 0; i < NW; i++) w[i] ^= o.w[i]; }
    void or_with(const BS& o) { for (int i = 0; i < NW; i++) w[i] |= o.w[i]; }
    void andnot_with(const BS& o) { for (int i = 0; i < NW; i++) w[i] &= ~o.w[i]; }
    bool disjoint(const BS& o) const { for (int i = 0; i < NW; i++) if (w[i] & o.w[i]) return false; return true; }
    int highest() const { for (int i = NW - 1; i >= 0; i--) if (w[i]) return i * 64 + 63 - __builtin_clzll(w[i]); return -1; }
    template <class F> void for_each(F f) const {
        for (int i = 0; i < NW; i++) { uint64_t x = w[i]; while (x) { int b = __builtin_ctzll(x); f(i * 64 + b); x &= x - 1; } }
    }
};

struct Code {
    int n, mH, mG, W, delta;
    std::vector<BS> Hrow;   // check -> variables
    std::vector<BS> Hcol;   // variable -> checks
    std::vector<BS> Gbasis; // echelon rows of G, sorted by pivot (highest bit) descending
    std::vector<int> Gpivot;
    // x reduced against the G basis; true iff x is outside rowspan(G)
    bool outside_G(BS x) const {
        for (size_t i = 0; i < Gbasis.size(); i++)
            if (x.test(Gpivot[i])) x.xor_with(Gbasis[i]);
        return x.any();
    }
};

struct Search {
    const Code& C;
    std::atomic<bool> stop{false};
    std::atomic<bool> found{false};
    std::atomic<bool> timed_out{false};   // the time limit fired: the search is incomplete
    std::atomic<long long> nodes{0};
    BS solution;
    double tlimit;
    std::chrono::steady_clock::time_point t0;
    Search(const Code& c, double tl) : C(c), tlimit(tl), t0(std::chrono::steady_clock::now()) {}
    double elapsed() const { return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(); }

    // returns true if a logical of weight <= W was found in this subtree
    bool dfs(const BS& S, const BS& F, const BS& syn, int size, long long& local_nodes) {
        if (stop.load(std::memory_order_relaxed)) return false;
        if ((++local_nodes & 0xFFFF) == 0) {   // every 65536 nodes: time check
            if (elapsed() > tlimit) { timed_out.store(true); stop.store(true); }
        }
        // syndrome is nonzero here (zero syndromes are resolved by the caller)
        // choose the unsatisfied check with the fewest available variables; bounds
        BS blocked = S; blocked.or_with(F);
        int best_c = -1, best_cnt = 1 << 30;
        int unsat = 0;
        BS used; used.clear();
        int b2 = 0;
        bool dead = false;
        syn.for_each([&](int c) {
            if (dead) return;
            unsat++;
            BS av = C.Hrow[c]; av.andnot_with(blocked);
            int cnt = av.count();
            if (cnt == 0) { dead = true; return; }
            if (cnt < best_cnt) { best_cnt = cnt; best_c = c; }
            if (av.disjoint(used)) { b2++; used.or_with(av); }
        });
        if (dead) return false;
        int b1 = (unsat + C.delta - 1) / C.delta;
        int need = b1 > b2 ? b1 : b2;
        if (size + need > C.W) return false;
        // branch over the available variables of best_c, ascending; each tried
        // variable is forbidden for the later siblings
        BS av = C.Hrow[best_c]; av.andnot_with(blocked);
        BS Fl = F;
        std::vector<int> vars; vars.reserve(16);
        av.for_each([&](int v) { vars.push_back(v); });
        for (int v : vars) {
            if (stop.load(std::memory_order_relaxed)) return false;
            BS S2 = S; S2.set(v);
            BS syn2 = syn; syn2.xor_with(C.Hcol[v]);
            if (!syn2.any()) {
                if (C.outside_G(S2)) {   // logical operator of weight size+1 <= W
                    bool exp = false;
                    if (found.compare_exchange_strong(exp, true)) { solution = S2; }
                    stop.store(true);
                    return true;
                }
                // stabilizer: the branch ends (a minimum logical has no zero-syndrome proper subset)
            } else if (size + 1 < C.W) {
                if (dfs(S2, Fl, syn2, size + 1, local_nodes)) return true;
            }
            Fl.set(v);
        }
        return false;
    }
};

static int bad_input(const char* what) { fprintf(stderr, "bad input: %s\n", what); return 2; }

int main(int argc, char** argv) {
    double tlimit = argc > 1 ? atof(argv[1]) : 1e18;
    if (!(tlimit > 0)) return bad_input("time limit must be positive");
    if (argc == 3) return bad_input("slice given without the number of slices");
    long long slice = argc > 3 ? atoll(argv[2]) : 0, slices = argc > 3 ? atoll(argv[3]) : 1;
    if (slices < 1 || slice < 0 || slice >= slices) { fprintf(stderr, "bad slice\n"); return 2; }
    int depth_arg = argc > 4 ? atoi(argv[4]) : 0;
    if (depth_arg < 0) return bad_input("negative depth");
    Code C;
    int threads = 1;
    if (scanf("%d %d %d %d %d", &C.n, &C.mH, &C.mG, &C.W, &threads) != 5) { fprintf(stderr, "bad header\n"); return 2; }
    if (C.n < 1 || C.mH < 0 || C.mG < 0 || C.W < 1 || threads < 1) return bad_input("header values out of range");
    if (C.n > NW * 64 || C.mH > NW * 64) { fprintf(stderr, "too large for NW=%d\n", NW); return 2; }
    C.Hrow.assign(C.mH, BS()); C.Hcol.assign(C.n, BS());
    for (auto& b : C.Hrow) b.clear();
    for (auto& b : C.Hcol) b.clear();
    for (int r = 0; r < C.mH; r++) {
        int k; if (scanf("%d", &k) != 1) return 2;
        if (k < 0 || k > C.n) return bad_input("row length of H");
        for (int j = 0; j < k; j++) {
            int v; if (scanf("%d", &v) != 1) return 2;
            if (v < 0 || v >= C.n) return bad_input("column index of H");
            if (C.Hrow[r].test(v)) return bad_input("repeated column index in a row of H");
            C.Hrow[r].set(v); C.Hcol[v].set(r);
        }
    }
    C.delta = 0;
    for (int v = 0; v < C.n; v++) C.delta = std::max(C.delta, C.Hcol[v].count());
    if (C.delta == 0) C.delta = 1;   // H without checks: b1 is 0 for an empty syndrome anyway
    // echelon basis of G (pivot = highest set bit), rows kept sorted by pivot descending
    for (int r = 0; r < C.mG; r++) {
        int k; if (scanf("%d", &k) != 1) return 2;
        if (k < 0 || k > C.n) return bad_input("row length of G");
        BS x; x.clear();
        for (int j = 0; j < k; j++) {
            int v; if (scanf("%d", &v) != 1) return 2;
            if (v < 0 || v >= C.n) return bad_input("column index of G");
            if (x.test(v)) return bad_input("repeated column index in a row of G");
            x.set(v);
        }
        for (size_t i = 0; i < C.Gbasis.size(); i++)
            if (x.test(C.Gpivot[i])) x.xor_with(C.Gbasis[i]);
        if (!x.any()) continue;
        int p = x.highest();
        size_t pos = 0;
        while (pos < C.Gbasis.size() && C.Gpivot[pos] > p) pos++;
        C.Gbasis.insert(C.Gbasis.begin() + pos, x);
        C.Gpivot.insert(C.Gpivot.begin() + pos, p);
    }
    int R; if (scanf("%d", &R) != 1) return 2;
    if (R < 1 || R > C.n) return bad_input("number of roots");
    std::vector<std::pair<int, int>> roots(R);
    std::vector<BS> root_forbid(R);
    for (int i = 0; i < R; i++) {
        if (scanf("%d %d", &roots[i].first, &roots[i].second) != 2) return 2;
        if (roots[i].first < 0 || roots[i].first >= C.n) return bad_input("root index");
        if (roots[i].second < 0 || roots[i].second > 2) return bad_input("root mode");
        root_forbid[i].clear();
        if (roots[i].second == 2) {
            int k; if (scanf("%d", &k) != 1) return 2;
            if (k < 0 || k >= C.n) return bad_input("forbidden-set size");
            for (int j = 0; j < k; j++) {
                int v; if (scanf("%d", &v) != 1) return 2;
                if (v < 0 || v >= C.n) return bad_input("forbidden index");
                root_forbid[i].set(v);
            }
            if (root_forbid[i].test(roots[i].first)) { fprintf(stderr, "root %d is in its own forbidden set\n", roots[i].first); return 2; }
        }
    }
    int extra; if (scanf("%d", &extra) == 1) return bad_input("trailing data after the roots");
#ifdef _OPENMP
    omp_set_num_threads(threads);
#endif
    Search Sr(C, tlimit);
    // frontier: expand each root `depth` levels so the parallel loop has many independent subtrees
    int depth = depth_arg > 0 ? depth_arg : (R <= 32 ? 4 : (R <= 400 ? 3 : 2));
    struct Node { BS S, F, syn; int size; };
    std::vector<Node> frontier;
    long long pre_nodes = 0;
    for (int ri = 0; ri < R; ri++) {
        if (Sr.elapsed() > tlimit) { Sr.timed_out.store(true); Sr.stop.store(true); break; }   // the limit covers the frontier build
        auto& rt = roots[ri];
        BS S; S.clear(); S.set(rt.first);
        BS F; F.clear();
        if (rt.second == 1) for (int v = 0; v < rt.first; v++) F.set(v);
        if (rt.second == 2) F = root_forbid[ri];
        BS syn = C.Hcol[rt.first];
        if (!syn.any()) {   // isolated variable: weight-1 logical or a stabilizer
            if (C.outside_G(S)) { Sr.found = true; Sr.solution = S; break; }
            continue;
        }
        // expand `depth` levels with the same ordered branching as dfs
        std::vector<Node> level = {Node{S, F, syn, 1}};
        for (int lv = 0; lv < depth && !Sr.found; lv++) {
            std::vector<Node> next;
            for (auto& nd : level) {
                pre_nodes++;
                if ((pre_nodes & 0xFFF) == 0 && Sr.elapsed() > tlimit) { Sr.timed_out.store(true); Sr.stop.store(true); break; }
                if (nd.size >= C.W) continue;   // no variable can be added within the weight limit
                BS blocked = nd.S; blocked.or_with(nd.F);
                int best_c = -1, best_cnt = 1 << 30; bool dead = false;
                nd.syn.for_each([&](int c) {
                    if (dead) return;
                    BS av = C.Hrow[c]; av.andnot_with(blocked);
                    int cnt = av.count();
                    if (cnt == 0) { dead = true; return; }
                    if (cnt < best_cnt) { best_cnt = cnt; best_c = c; }
                });
                if (dead) continue;
                BS av = C.Hrow[best_c]; av.andnot_with(blocked);
                BS Fl = nd.F;
                std::vector<int> vars; av.for_each([&](int v) { vars.push_back(v); });
                for (int v : vars) {
                    BS S2 = nd.S; S2.set(v);
                    BS syn2 = nd.syn; syn2.xor_with(C.Hcol[v]);
                    if (!syn2.any()) {
                        if (C.outside_G(S2)) { Sr.found = true; Sr.solution = S2; break; }
                    } else if (nd.size + 1 < C.W) {
                        next.push_back(Node{S2, Fl, syn2, nd.size + 1});
                    }
                    Fl.set(v);
                }
                if (Sr.found) break;
            }
            level.swap(next);
            if (Sr.timed_out.load()) break;
        }
        if (Sr.found || Sr.timed_out.load()) break;
        frontier.insert(frontier.end(), level.begin(), level.end());
    }
    size_t frontier_total = frontier.size();
    if (slices > 1) {   // keep this job's share of the independent subtrees
        std::vector<Node> mine;
        for (size_t i = 0; i < frontier.size(); i++) if ((long long)(i % slices) == slice) mine.push_back(frontier[i]);
        frontier.swap(mine);
    }
    fprintf(stderr, "n=%d mH=%d rank(G)=%zu delta=%d W=%d roots=%d depth=%d frontier=%zu (slice %lld/%lld of %zu) threads=%d\n",
            C.n, C.mH, C.Gbasis.size(), C.delta, C.W, R, depth, frontier.size(), slice, slices, frontier_total, threads);
    // machine-readable description of the searched problem and of this slice's share
    printf("INFO n=%d mH=%d rankG=%zu delta=%d W=%d roots=%d depth=%d frontier_total=%zu slice=%lld slices=%lld frontier_slice=%zu\n",
           C.n, C.mH, C.Gbasis.size(), C.delta, C.W, R, depth, frontier_total, slice, slices, frontier.size());
    long long done = 0;
    if (!Sr.found && !Sr.timed_out.load()) {
#pragma omp parallel
        {
            long long local = 0;
#pragma omp for schedule(dynamic, 1)
            for (long long i = 0; i < (long long)frontier.size(); i++) {
                if (Sr.stop.load(std::memory_order_relaxed)) continue;
                Sr.dfs(frontier[i].S, frontier[i].F, frontier[i].syn, frontier[i].size, local);
                if (Sr.stop.load(std::memory_order_relaxed)) continue;   // aborted subtree: not done
#pragma omp atomic
                done++;
            }
            Sr.nodes.fetch_add(local);   // std::atomic: no omp atomic clause needed
        }
    }
    double secs = Sr.elapsed();
    if (Sr.found) {
        printf("FOUND %d", Sr.solution.count());
        Sr.solution.for_each([](int v) { printf(" %d", v); });
        printf("\n");
    } else if (Sr.timed_out.load() || done < (long long)frontier.size()) {
        printf("TIMEOUT %lld/%zu\n", done, frontier.size());
    } else {
        printf("NONE %d\n", C.W);
    }
    printf("STATS %lld %.1f\n", (long long)Sr.nodes + pre_nodes, secs);
    return 0;
}
