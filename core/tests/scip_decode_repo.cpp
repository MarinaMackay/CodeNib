#include "scip_decode.h"

#include <array>
#include <filesystem>
#include <iostream>
#include <optional>
#include <string_view>

using codeminer::core::CodeGraph;
using codeminer::core::SCIPGraphDecoder;
namespace fs = std::filesystem;

namespace {

struct ProgramOptions {
    fs::path index;
    std::optional<fs::path> project_root;
    fs::path output;
};

constexpr std::array<std::string_view, 4> kNullRootSynonyms = {
    "null", "NULL", "none", "-"
};

void print_usage(std::string_view prog_name) {
    std::cerr << "Usage: " << prog_name << " <index.scip> <project_root> <output.json>\n\n"
              << "Arguments:\n"
              << "  index.scip    - Path to SCIP index file\n"
              << "  project_root  - Project root directory (or 'null' for none)\n"
              << "  output.json   - Output JSON file path\n\n"
              << "Example:\n"
              << "  " << prog_name << " /tmp/index.scip /tmp/project output.json\n"
              << "  " << prog_name << " /tmp/index.scip null output.json\n";
}

bool path_exists(const fs::path& path, std::string_view description) {
    if (fs::exists(path)) {
        return true;
    }

    std::cerr << "Error: " << description << " not found: " << path << "\n";
    return false;
}

bool is_null_root(std::string_view value) {
    for (const auto synonym : kNullRootSynonyms) {
        if (value == synonym) {
            return true;
        }
    }
    return false;
}

std::optional<ProgramOptions> parse_arguments(int argc, char** argv) {
    if (argc != 4) {
        print_usage(argv[0]);
        return std::nullopt;
    }

    ProgramOptions options{
        .index = argv[1],
        .project_root = std::nullopt,
        .output = argv[3],
    };

    if (!path_exists(options.index, "Index file")) {
        return std::nullopt;
    }

    const std::string_view project_root_arg = argv[2];
    if (!is_null_root(project_root_arg)) {
        options.project_root = fs::path{project_root_arg};
        if (!path_exists(*options.project_root, "Project root")) {
            return std::nullopt;
        }
    }

    return options;
}

void print_header() {
    std::cout << "========================================\n"
              << "C++ SCIP Decoder\n"
              << "========================================\n";
}

void print_options(const ProgramOptions& options) {
    std::cout << "Index file:   " << options.index << "\n";
    if (options.project_root) {
        std::cout << "Project root: " << *options.project_root << "\n";
    } else {
        std::cout << "Project root: (none)\n";
    }
    std::cout << "Output file:  " << options.output << "\n\n";
}

void print_success() {
    std::cout << "\n"
              << "========================================\n"
              << "✅ SUCCESS\n"
              << "========================================\n";
}

void print_failure(const std::exception& error) {
    std::cerr << "\n"
              << "========================================\n"
              << "❌ ERROR\n"
              << "========================================\n"
              << error.what() << "\n";
}

}  // namespace

int main(int argc, char** argv) {
    const auto options = parse_arguments(argc, argv);
    if (!options) {
        return 1;
    }

    print_header();
    print_options(*options);

    try {
        SCIPGraphDecoder decoder(options->index.string(), options->project_root);
        std::cout << "Decoding SCIP index...\n";
        CodeGraph graph = decoder.decode();
        std::cout << "✓ Decoding completed\n";

        std::cout << "Saving graph to JSON...\n";
        graph.save_graph(options->output.string());
        std::cout << "✓ Saved to: " << options->output << "\n";

        print_success();
        return 0;

    } catch (const std::exception& error) {
        print_failure(error);
        return 1;
    }
}
