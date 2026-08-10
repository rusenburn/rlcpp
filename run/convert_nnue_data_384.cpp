// Upgrades an NNUE training set from the 256-feature layout to the
// 384-feature one by deriving the two piline channels offline.
//
//   convert_nnue_data_384 <in.bin> <out.bin>
//
// The piline sets - the empty squares a player may not play on, because doing
// so would build an unbroken line of more than four - are a pure function of
// that player's own pieces, which channels 0-3 already record in full. The
// rule is colour-symmetric, so the stm-relative labelling in the file is
// enough; and it is invariant under the dihedral group, so the seven
// symmetric orientations the generator writes alongside each position convert
// correctly on their own terms. None of the 800-simulations-per-move MCTS
// self-play has to be re-run.
//
// The derivation calls the same mgbb::compute_runs the engine calls at search
// time, so training and inference cannot drift apart.
//
// Record format, unchanged apart from the wider feature ids:
//   float32 score, int16 count, int16 feature_id[count]
//
// Record ORDER is preserved byte for byte. clustered_split() in
// scripts/train_nnue_v2.py groups consecutive records in chunks of 8 because
// the generator writes each position's 8 orientations back to back; reordering
// or dropping records would silently let symmetric siblings straddle the
// train/validation boundary.

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include <games/migoyugo_bb.hpp>

using namespace rl::games::mgbb;

int main(int argc, char** argv)
{
    if (argc < 3)
    {
        std::fprintf(stderr,
            "usage: %s <in.bin> <out.bin>\n"
            "  in.bin  training data with 256-feature ids (0..255)\n"
            "  out.bin training data with 384-feature ids (0..383)\n", argv[0]);
        return 2;
    }

    const std::string in_path = argv[1];
    const std::string out_path = argv[2];

    FILE* in = std::fopen(in_path.c_str(), "rb");
    if (!in) { std::fprintf(stderr, "cannot open %s\n", in_path.c_str()); return 1; }
    FILE* out = std::fopen(out_path.c_str(), "wb");
    if (!out) { std::fprintf(stderr, "cannot open %s for writing\n", out_path.c_str()); std::fclose(in); return 1; }

    long long records = 0;
    long long piline_cells = 0;
    long long empty_piline_records = 0;
    int max_count = 0;

    std::vector<int16_t> ids(512);

    while (true)
    {
        float score;
        if (std::fread(&score, sizeof(float), 1, in) != 1) break; // clean EOF

        int16_t count;
        if (std::fread(&count, sizeof(int16_t), 1, in) != 1)
        {
            std::fprintf(stderr, "record %lld: truncated before the feature count\n", records);
            return 1;
        }
        if (count < 0 || count > 192)
        {
            std::fprintf(stderr, "record %lld: implausible feature count %d\n", records, count);
            return 1;
        }
        if (std::fread(ids.data(), sizeof(int16_t), count, in) != static_cast<size_t>(count))
        {
            std::fprintf(stderr, "record %lld: truncated feature list\n", records);
            return 1;
        }

        // Channels 0/1 are the side to move's Migos and Yugos, 2/3 the
        // opponent's. Legality only cares about the union.
        uint64_t own = 0, opp = 0;
        for (int i = 0; i < count; ++i)
        {
            const int fid = ids[i];
            if (fid < 0 || fid >= 384)
            {
                std::fprintf(stderr, "record %lld: feature id %d out of range\n", records, fid);
                return 1;
            }
            if (fid >= 256)
            {
                std::fprintf(stderr,
                    "record %lld: id %d is already a piline feature - %s has been converted already\n",
                    records, fid, in_path.c_str());
                return 1;
            }
            const int ch = fid >> 6;
            const int sq = fid & 63;
            if (ch < 2) own |= 1ULL << sq; else opp |= 1ULL << sq;
        }

        const uint64_t empty = ~(own | opp);

        uint64_t illegal_own, illegal_opp, makes4;
        compute_runs(own, illegal_own, makes4);
        compute_runs(opp, illegal_opp, makes4);

        const uint64_t piline_own = illegal_own & empty;
        const uint64_t piline_opp = illegal_opp & empty;

        // A piline cell is by definition empty; if this ever fires the axis
        // masks are wrong.
        if ((piline_own | piline_opp) & (own | opp))
        {
            std::fprintf(stderr, "record %lld: a piline cell landed on an occupied square\n", records);
            return 1;
        }

        int n = count;
        for (uint64_t b = piline_own; b; b &= b - 1) ids[n++] = static_cast<int16_t>(4 * 64 + ctz64(b));
        for (uint64_t b = piline_opp; b; b &= b - 1) ids[n++] = static_cast<int16_t>(5 * 64 + ctz64(b));

        piline_cells += n - count;
        if (n == count) ++empty_piline_records;
        if (n > max_count) max_count = n;

        const int16_t new_count = static_cast<int16_t>(n);
        std::fwrite(&score, sizeof(float), 1, out);
        std::fwrite(&new_count, sizeof(int16_t), 1, out);
        std::fwrite(ids.data(), sizeof(int16_t), n, out);

        ++records;
        if (records % 500000 == 0)
            std::fprintf(stderr, "  %lld records...\n", records);
    }

    std::fclose(in);
    std::fclose(out);

    std::printf("converted %lld records -> %s\n", records, out_path.c_str());
    std::printf("  mean piline features per record: %.3f\n",
        records ? static_cast<double>(piline_cells) / records : 0.0);
    std::printf("  records with no piline cell at all: %lld (%.1f%%)\n",
        empty_piline_records, records ? 100.0 * empty_piline_records / records : 0.0);
    std::printf("  max active features per record: %d\n", max_count);

    if (records && static_cast<double>(piline_cells) / records < 0.2)
        std::printf("\n  WARNING: that piline rate looks too low - expected around 1.3.\n"
            "  Check the axis masks in games/include/games/migoyugo_bb.hpp.\n");

    return 0;
}
