#include "scip_decode.h"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <optional>
#include <regex>
#include <sstream>
#include <stdexcept>

namespace codeminer::core {

namespace {

bool debug_logging_enabled() {
    static const bool enabled = std::getenv("CODEMINER_SCIP_DEBUG") != nullptr;
    return enabled;
}

void log_debug(const std::string& message) {
    if (debug_logging_enabled()) {
        std::cerr << "[SCIPDecode] " << message << '\n';
    }
}

std::vector<int> extract_integers(const std::string& text, const std::regex& pattern) {
    std::vector<int> results;
    auto begin = std::sregex_iterator(text.begin(), text.end(), pattern);
    auto end = std::sregex_iterator();
    for (auto it = begin; it != end; ++it) {
        results.push_back(std::stoi((*it)[1].str()));
    }
    return results;
}

std::string rstrip_periods(std::string value) {
    while (!value.empty() && value.back() == '.') {
        value.pop_back();
    }
    return value;
}

bool starts_with_upper(const std::string& value) {
    if (value.empty()) {
        return false;
    }
    return static_cast<unsigned char>(value.front()) >= 'A' &&
           static_cast<unsigned char>(value.front()) <= 'Z';
}

bool contains_function_parentheses(const std::string& symbol) {
    return symbol.find("()") != std::string::npos || symbol.find('(') != std::string::npos;
}

std::vector<std::string> extract_blocks(const std::string& text, const std::string& keyword) {
    std::vector<std::string> blocks;
    std::size_t search_pos = 0;
    while (true) {
        std::size_t key_pos = text.find(keyword, search_pos);
        if (key_pos == std::string::npos) {
            break;
        }

        if (key_pos > 0) {
            unsigned char prev = static_cast<unsigned char>(text[key_pos - 1]);
            if (std::isalnum(prev) || prev == '_' || prev == '/') {
                search_pos = key_pos + keyword.size();
                continue;
            }
        }

        std::size_t brace_pos = key_pos + keyword.size();
        while (brace_pos < text.size() && std::isspace(static_cast<unsigned char>(text[brace_pos]))) {
            ++brace_pos;
        }
        if (brace_pos >= text.size() || text[brace_pos] != '{') {
            search_pos = key_pos + keyword.size();
            continue;
        }

        std::size_t start = brace_pos + 1;
        int depth = 1;
        bool in_string = false;
        bool escape = false;
        for (std::size_t i = start; i < text.size(); ++i) {
            char ch = text[i];
            if (in_string) {
                if (escape) {
                    escape = false;
                } else if (ch == '\\') {
                    escape = true;
                } else if (ch == '"') {
                    in_string = false;
                }
                continue;
            }

            if (ch == '"') {
                in_string = true;
                continue;
            }
            if (ch == '{') {
                ++depth;
                continue;
            }
            if (ch == '}') {
                --depth;
                if (depth == 0) {
                    blocks.emplace_back(text.substr(start, i - start));
                    search_pos = i + 1;
                    break;
                }
                continue;
            }
        }

        if (depth != 0) {
            log_debug("Unbalanced braces while parsing blocks for keyword '" + keyword + "'");
            break;
        }
    }
    return blocks;
}

}  // namespace

SCIPGraphDecoder::SCIPGraphDecoder(std::string index_file_path,
                                   std::optional<std::string> project_root)
    : index_file_path_(std::move(index_file_path)),
      project_root_(std::move(project_root)),
      code_graph_(project_root_ ? *project_root_ : std::string{}) {}

CodeGraph SCIPGraphDecoder::decode() {
    std::ifstream input(index_file_path_);
    if (!input.is_open()) {
        throw std::runtime_error("Failed to open SCIP index file at " + index_file_path_);
    }
    log_debug("Starting decode for index: " + index_file_path_);

    std::ostringstream buffer;
   buffer << input.rdbuf();
   std::string content = buffer.str();

    code_graph_.add_root_node(ROOT_NODE);
    log_debug("Added root node");

    auto document_blocks = extract_blocks(content, "documents");
    for (const auto& block : document_blocks) {
        log_debug("Processing document block");
        process_document(block);
    }

    log_debug("Finished decode");
    return std::move(code_graph_);
}

void SCIPGraphDecoder::process_document(const std::string& document_block) {
    static const std::regex relative_path_regex(R"regex(relative_path:\s*"([^"]+)")regex");
    std::smatch match;
    if (!std::regex_search(document_block, match, relative_path_regex)) {
        return;
    }

    std::string file_path = match[1].str();
    log_debug("Processing file: " + file_path);
    std::filesystem::path file_fs_path(file_path);

    std::filesystem::path dir_path = file_fs_path.parent_path();
    while (!dir_path.empty() && dir_path != dir_path.parent_path()) {
        std::string dir_str = dir_path.generic_string();
        if (indexed_directories_.insert(dir_str).second) {
            code_graph_.add_directory_node(dir_str);
            std::string parent_str = dir_path.parent_path().generic_string();
            if (parent_str.empty()) {
                parent_str = ROOT_NODE;
            }
            code_graph_.add_edge(parent_str, dir_str, EDGE_TYPE_CONTAIN);
        }
        dir_path = dir_path.parent_path();
    }

    code_graph_.add_file_node(file_path);
    log_debug("Added file node: " + file_path);

    std::string parent_str = file_fs_path.parent_path().generic_string();
    if (parent_str.empty()) {
        parent_str = ROOT_NODE;
    }
    code_graph_.add_edge(parent_str, file_path, EDGE_TYPE_CONTAIN);
    log_debug("Added file containment edge from " + parent_str + " to " + file_path);

    auto occurrence_blocks = extract_blocks(document_block, "occurrences");
    for (const auto& occurrence : occurrence_blocks) {
        process_occurrence(occurrence);
    }
}

void SCIPGraphDecoder::process_occurrence(const std::string& occurrence_block) {
    if (occurrence_block.find("local") != std::string::npos) {
        log_debug("Skipping local occurrence");
        return;
    }
    if (occurrence_block.find("python-stdlib") != std::string::npos) {
        log_debug("Skipping stdlib occurrence");
        return;
    }

    static const std::regex range_regex(R"regex(range:\s*(\d+))regex");
    static const std::regex symbol_regex(R"regex(symbol:\s*"([^"]+)")regex");
    static const std::regex symbol_roles_regex(R"regex(symbol_roles:\s*(\d+))regex");
    static const std::regex enclosing_range_regex(R"regex(enclosing_range:\s*(\d+))regex");

    auto ranges = extract_integers(occurrence_block, range_regex);
    if (ranges.size() < 3) {
        return;
    }
    int line = ranges[0];

    std::smatch match;
    if (!std::regex_search(occurrence_block, match, symbol_regex)) {
        log_debug("Occurrence missing symbol");
        return;
    }
    std::string symbol = match[1].str();

    if (!std::regex_search(occurrence_block, match, symbol_roles_regex)) {
        log_debug("Occurrence missing symbol_roles");
        return;
    }
    int symbol_roles = std::stoi(match[1].str());

    auto enclosing_ranges = extract_integers(occurrence_block, enclosing_range_regex);
    log_debug("Processing symbol '" + symbol + "' at line " + std::to_string(line) +
              " roles=" + std::to_string(symbol_roles));
    process_symbol(symbol, line, symbol_roles, enclosing_ranges);
}

std::string SCIPGraphDecoder::unify_symbol_name(const std::string& symbol) const {
    std::string clean_symbol = symbol;
    clean_symbol.erase(std::remove(clean_symbol.begin(), clean_symbol.end(), '`'), clean_symbol.end());

    auto slash_pos = clean_symbol.find('/');
    if (slash_pos != std::string::npos) {
        std::string module_path = clean_symbol.substr(0, slash_pos);
        std::replace(module_path.begin(), module_path.end(), '.', '/');
        module_path += ".py";

        std::string symbol_part;
        if (slash_pos + 1 < clean_symbol.size()) {
            symbol_part = clean_symbol.substr(slash_pos + 1);
        }

        if (symbol_part.empty()) {
            return module_path;
        }

        if (auto hash_pos = symbol_part.find('#'); hash_pos != std::string::npos) {
            std::string class_name = symbol_part.substr(0, hash_pos);
            std::string remainder = symbol_part.substr(hash_pos + 1);
            remainder = rstrip_periods(remainder);

            if (!remainder.empty()) {
                return module_path + ":" + class_name + "." + remainder;
            }
            return module_path + ":" + class_name;
        }

        symbol_part = rstrip_periods(symbol_part);
        return module_path + ":" + symbol_part;
    }

    return rstrip_periods(clean_symbol);
}

std::string SCIPGraphDecoder::classify_symbol_type(const std::string& unified_symbol,
                                                   const std::string& original_symbol) const {
    auto colon_pos = unified_symbol.find(':');
    if (colon_pos != std::string::npos) {
        std::string symbol_part = unified_symbol.substr(colon_pos + 1);
        if (symbol_part.find('.') != std::string::npos) {
            if (contains_function_parentheses(original_symbol)) {
                return NODE_TYPE_METHOD;
            }
            return NODE_TYPE_FIELD;
        }
        if (starts_with_upper(symbol_part)) {
            return NODE_TYPE_CLASS;
        }
        return NODE_TYPE_FUNCTION;
    }

    return NODE_TYPE_FUNCTION;
}

void SCIPGraphDecoder::process_symbol(const std::string& symbol,
                                      int line,
                                      int symbol_roles,
                                      const std::vector<int>& enclosing_ranges) {
    static const std::regex arg_regex(R"(\.\([^)]+\)$)");
    if (std::regex_search(symbol, arg_regex)) {
        return;
    }

    code_graph_.exit_scopes_by_line(line);

    static const std::regex module_regex(R"regex(`?([^`]+)`?/([^.]+)(?:\.|\(|#))regex");
    std::smatch module_match;
    if (!std::regex_search(symbol, module_match, module_regex)) {
        log_debug("Failed to match module for symbol: " + symbol);
        return;
    }
    std::string module_path = module_match[1].str();

    std::string cleaned_symbol = symbol;
    if (auto last_space = cleaned_symbol.find_last_of(' '); last_space != std::string::npos) {
        cleaned_symbol = cleaned_symbol.substr(last_space + 1);
    }
    cleaned_symbol.erase(std::remove(cleaned_symbol.begin(), cleaned_symbol.end(), '`'), cleaned_symbol.end());

    std::string unified_symbol = unify_symbol_name(cleaned_symbol);
    std::string symbol_type = classify_symbol_type(unified_symbol, cleaned_symbol);

    const bool is_definition = symbol_roles == 1;
    const bool is_reference = symbol_roles == 8;

    if (unified_symbol.find("/__init__") != std::string::npos) {
        std::regex init_regex(R"regex((.+)/(?:__init__))regex");
        std::smatch init_match;
        if (std::regex_search(unified_symbol, init_match, init_regex)) {
            std::string module_dir = init_match[1].str();
            std::replace(module_dir.begin(), module_dir.end(), '.', '/');
            std::string file_path = module_dir + ".py";
            if (is_reference) {
                code_graph_.add_edge(code_graph_.current_scope(), file_path, EDGE_TYPE_REFERENCE);
            }
        }
        return;
    }

    if (is_definition && enclosing_ranges.size() >= 4) {
        int scope_start_line = enclosing_ranges[0];
        int scope_end_line = enclosing_ranges[2];
        log_debug("Adding definition with scope: " + unified_symbol +
                  " [" + std::to_string(scope_start_line) + ", " + std::to_string(scope_end_line) + "]");
        code_graph_.add_symbol_node(unified_symbol,
                                    line,
                                    scope_start_line,
                                    scope_end_line,
                                    symbol_type);
        code_graph_.add_containment_edge(unified_symbol);

        if (symbol_type == NODE_TYPE_CLASS || symbol_type == NODE_TYPE_FUNCTION || symbol_type == NODE_TYPE_METHOD) {
            code_graph_.update_current_scope(unified_symbol, scope_start_line, scope_end_line);
        }
    } else if (is_definition) {
        log_debug("Adding definition without scope: " + unified_symbol);
        code_graph_.add_symbol_node(unified_symbol, line, std::nullopt, std::nullopt, symbol_type);
        code_graph_.add_edge(code_graph_.current_scope(), unified_symbol, EDGE_TYPE_CONTAIN);
    } else if (is_reference) {
        log_debug("Adding reference: " + unified_symbol + " from module " + module_path);
        code_graph_.add_symbol_reference(unified_symbol, module_path, symbol_type);
    }
}

}  // namespace codeminer::core
