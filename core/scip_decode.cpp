#include "scip_decode.h"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <future>
#include <iostream>
#include <optional>
#include <re2/re2.h>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <unordered_map>
#include <unordered_set>

namespace codeminer::core {

namespace {

bool debug_logging_enabled() {
  static const bool enabled = std::getenv("CODEMINER_SCIP_DEBUG") != nullptr;
  return enabled;
}

void log_debug(const std::string &message) {
  if (debug_logging_enabled()) {
    std::cerr << "[SCIPDecode] " << message << '\n';
  }
}

std::vector<int> extract_integers(const std::string &text,
                                  const re2::RE2 &pattern) {
  std::vector<int> results;
  re2::StringPiece input(text);
  int value = 0;
  while (re2::RE2::FindAndConsume(&input, pattern, &value)) {
    results.push_back(value);
  }
  return results;
}

std::string rstrip_periods(std::string value) {
  while (!value.empty() && value.back() == '.') {
    value.pop_back();
  }
  return value;
}

bool starts_with_upper(const std::string &value) {
  if (value.empty()) {
    return false;
  }
  return static_cast<unsigned char>(value.front()) >= 'A' &&
         static_cast<unsigned char>(value.front()) <= 'Z';
}

bool contains_function_parentheses(const std::string &symbol) {
  return symbol.find("()") != std::string::npos ||
         symbol.find('(') != std::string::npos;
}

std::vector<std::string> extract_blocks(const std::string &text,
                                        const std::string &keyword) {
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
    while (brace_pos < text.size() &&
           std::isspace(static_cast<unsigned char>(text[brace_pos]))) {
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
      log_debug("Unbalanced braces while parsing blocks for keyword '" +
                keyword + "'");
      break;
    }
  }
  return blocks;
}

} // namespace

struct SCIPGraphDecoder::Subgraph {
  struct Edge {
    std::string source;
    std::string target;
    std::string type;
  };

  struct Node {
    CodeGraph::VertexData data;
    bool is_definition{true}; // True for definitions, false for references
  };

  std::unordered_map<std::string, Node> nodes;
  std::vector<Edge> edges;
};

class SCIPGraphDecoder::SubgraphBuilder {
public:
  SubgraphBuilder() = default;

  void add_directory_node(const std::string &dir_path) {
    Subgraph::Node &node = ensure_node(dir_path);
    apply_update(node, std::make_optional<std::string>(NODE_TYPE_DIRECTORY),
                 std::nullopt, std::nullopt, std::nullopt);
  }

  void add_file_node(const std::string &file_path) {
    current_file_ = file_path;
    Subgraph::Node &node = ensure_node(file_path);
    apply_update(node, std::make_optional<std::string>(NODE_TYPE_FILE),
                 std::nullopt, std::nullopt, std::nullopt);
    reset_scope_to_file(file_path);
  }

  void add_symbol_node(const std::string &symbol, int line,
                       std::optional<int> scope_start_line,
                       std::optional<int> scope_end_line,
                       const std::string &symbol_type) {
    Subgraph::Node &node = ensure_node(symbol);
    apply_update(node, std::make_optional<std::string>(symbol_type),
                 std::make_optional<std::string>(current_file_),
                 std::make_optional<int>(line), std::make_optional<int>(line));

    if (scope_start_line.has_value() && scope_end_line.has_value()) {
      apply_update(node, std::make_optional<std::string>(symbol_type),
                   std::make_optional<std::string>(current_file_),
                   scope_start_line, scope_end_line);
    }

    // Always mark as definition (override if it was previously marked as
    // reference)
    node.is_definition = true;
  }

  void add_symbol_reference(const std::string &symbol,
                            const std::optional<std::string> &module_path,
                            const std::string &symbol_type) {
    const bool already_exists =
        subgraph_.nodes.find(symbol) != subgraph_.nodes.end();
    Subgraph::Node &node = ensure_node(symbol);
    std::optional<std::string> file_attr =
        (!already_exists && module_path.has_value())
            ? std::make_optional<std::string>(*module_path)
            : std::nullopt;

    apply_update(node, std::make_optional<std::string>(symbol_type), file_attr,
                 std::nullopt, std::nullopt);

    // Only mark as reference if it wasn't already added as a definition
    if (!already_exists) {
      node.is_definition = false;
    }

    add_edge(current_scope_, symbol, EDGE_TYPE_REFERENCE);
  }

  void add_containment_edge(const std::string &target_symbol) {
    add_edge(current_scope_, target_symbol, EDGE_TYPE_CONTAIN);
  }

  void add_edge(const std::string &source, const std::string &target,
                const std::string &edge_type) {
    subgraph_.edges.push_back(Subgraph::Edge{source, target, edge_type});
  }

  void update_current_scope(const std::string &symbol,
                            std::optional<int> start_line = std::nullopt,
                            std::optional<int> end_line = std::nullopt) {
    current_scope_ = symbol;
    Range range;
    if (start_line.has_value() && end_line.has_value()) {
      range.has_range = true;
      range.start_line = *start_line;
      range.end_line = *end_line;
    }
    scope_stack_.push_back(ScopeEntry{symbol, range});
  }

  void exit_scopes_by_line(int current_line) {
    while (scope_stack_.size() > 1) {
      const ScopeEntry &top = scope_stack_.back();
      if (!top.range.has_range) {
        break;
      }
      if (current_line > top.range.end_line) {
        scope_stack_.pop_back();
      } else {
        break;
      }
    }

    if (!scope_stack_.empty()) {
      current_scope_ = scope_stack_.back().symbol;
    } else if (!current_file_.empty()) {
      reset_scope_to_file(current_file_);
    }
  }

  bool add_directory_if_needed(const std::string &dir_path) {
    if (dir_path.empty()) {
      return false;
    }
    if (indexed_directories_.insert(dir_path).second) {
      add_directory_node(dir_path);
      return true;
    }
    return false;
  }

  const std::string &current_scope() const { return current_scope_; }

  Subgraph build() { return std::move(subgraph_); }

private:
  struct Range {
    bool has_range{false};
    int start_line{0};
    int end_line{0};
  };

  struct ScopeEntry {
    std::string symbol;
    Range range;
  };

  void reset_scope_to_file(const std::string &file_symbol) {
    scope_stack_.clear();
    scope_stack_.push_back(ScopeEntry{file_symbol, Range{false, 0, 0}});
    current_scope_ = file_symbol;
  }

  Subgraph::Node &ensure_node(const std::string &name) {
    auto it = subgraph_.nodes.find(name);
    if (it != subgraph_.nodes.end()) {
      return it->second;
    }

    Subgraph::Node record;
    record.data.name = name;
    auto inserted = subgraph_.nodes.emplace(name, std::move(record));
    return inserted.first->second;
  }

  void apply_update(Subgraph::Node &node,
                    const std::optional<std::string> &type,
                    const std::optional<std::string> &file,
                    const std::optional<int> &start_line,
                    const std::optional<int> &end_line) {
    if (type.has_value()) {
      node.data.type = *type;
    }
    if (file.has_value()) {
      if (file->empty()) {
        node.data.file.reset();
      } else {
        node.data.file = *file;
      }
    }
    if (start_line.has_value()) {
      node.data.start_line = *start_line;
    }
    if (end_line.has_value()) {
      node.data.end_line = *end_line;
    }
  }

  Subgraph subgraph_;
  std::unordered_set<std::string> indexed_directories_;
  std::vector<ScopeEntry> scope_stack_;
  std::string current_file_;
  std::string current_scope_;
};

SCIPGraphDecoder::SCIPGraphDecoder(std::string index_file_path,
                                   std::optional<std::string> project_root)
    : index_file_path_(std::move(index_file_path)),
      project_root_(std::move(project_root)),
      code_graph_(project_root_ ? *project_root_ : std::string{}) {}

CodeGraph SCIPGraphDecoder::decode() {
  auto start = std::chrono::high_resolution_clock::now();
  std::ifstream input(index_file_path_);
  if (!input.is_open()) {
    throw std::runtime_error("Failed to open SCIP index file at " +
                             index_file_path_);
  }

  log_debug("Starting decode for index: " + index_file_path_);

  std::ostringstream buffer;
  buffer << input.rdbuf();
  std::string content = buffer.str();

  auto end = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::seconds>(end - start);
  std::cout << "Duration of reading: " << duration.count() << "s\n";

  start = std::chrono::high_resolution_clock::now();
  auto document_blocks = extract_blocks(content, "documents");
  log_debug("Submitting " + std::to_string(document_blocks.size()) +
            " document blocks for parallel decode");

  // Use a thread pool approach: limit concurrent threads to hardware
  // concurrency
  const size_t max_threads = std::thread::hardware_concurrency();
  const size_t num_blocks = document_blocks.size();
  std::vector<Subgraph> subgraphs;
  subgraphs.resize(num_blocks);

  // Process blocks in batches
  for (size_t batch_start = 0; batch_start < num_blocks;
       batch_start += max_threads) {
    const size_t batch_end = std::min(batch_start + max_threads, num_blocks);
    std::vector<std::future<Subgraph>> futures;
    futures.reserve(batch_end - batch_start);

    for (size_t i = batch_start; i < batch_end; ++i) {
      futures.emplace_back(
          std::async(std::launch::async, [this, &document_blocks, i]() {
            return process_document(document_blocks[i]);
          }));
    }

    // Wait for this batch to complete
    for (size_t i = 0; i < futures.size(); ++i) {
      subgraphs[batch_start + i] = futures[i].get();
    }
  }

  code_graph_.add_root_node(ROOT_NODE);
  log_debug("Added root node");
  merge_subgraphs(subgraphs);

  end = std::chrono::high_resolution_clock::now();
  duration = std::chrono::duration_cast<std::chrono::seconds>(end - start);
  std::cout << "Duration of decoding: " << duration.count() << "s\n";

  log_debug("Finished decode");
  return std::move(code_graph_);
}

SCIPGraphDecoder::Subgraph
SCIPGraphDecoder::process_document(const std::string &document_block) const {
  static const re2::RE2 relative_path_regex(
      R"re2(relative_path:\s*"([^"]+)")re2");
  std::string file_path;
  if (!re2::RE2::PartialMatch(document_block, relative_path_regex,
                              &file_path)) {
    return Subgraph{};
  }
  log_debug("Processing file: " + file_path);
  std::filesystem::path file_fs_path(file_path);
  SubgraphBuilder builder;

  std::filesystem::path dir_path = file_fs_path.parent_path();
  while (!dir_path.empty() && dir_path != dir_path.parent_path()) {
    std::string dir_str = dir_path.generic_string();
    if (builder.add_directory_if_needed(dir_str)) {
      std::string parent_str = dir_path.parent_path().generic_string();
      if (parent_str.empty()) {
        parent_str = ROOT_NODE;
      }
      builder.add_edge(parent_str, dir_str, EDGE_TYPE_CONTAIN);
    }
    dir_path = dir_path.parent_path();
  }

  builder.add_file_node(file_path);
  log_debug("Added file node: " + file_path);

  std::string parent_str = file_fs_path.parent_path().generic_string();
  if (parent_str.empty()) {
    parent_str = ROOT_NODE;
  }
  builder.add_edge(parent_str, file_path, EDGE_TYPE_CONTAIN);
  log_debug("Added file containment edge from " + parent_str + " to " +
            file_path);

  auto occurrence_blocks = extract_blocks(document_block, "occurrences");
  for (const auto &occurrence : occurrence_blocks) {
    process_occurrence(occurrence, builder);
  }

  return builder.build();
}

void SCIPGraphDecoder::process_occurrence(const std::string &occurrence_block,
                                          SubgraphBuilder &builder) const {
  // Skip stdlib symbols
  if (occurrence_block.find("python-stdlib") != std::string::npos) {
    log_debug("Skipping stdlib occurrence");
    return;
  }

  static const re2::RE2 range_regex(R"re2(range:\s*(\d+))re2");
  static const re2::RE2 symbol_regex(R"re2(symbol:\s*"([^"]+)")re2");
  static const re2::RE2 symbol_roles_regex(R"re2(symbol_roles:\s*(\d+))re2");
  static const re2::RE2 enclosing_range_regex(
      R"re2(enclosing_range:\s*(\d+))re2");

  auto ranges = extract_integers(occurrence_block, range_regex);
  if (ranges.size() < 3) {
    return;
  }
  int line = ranges[0];

  std::string symbol;
  if (!re2::RE2::PartialMatch(occurrence_block, symbol_regex, &symbol)) {
    log_debug("Occurrence missing symbol");
    return;
  }

  // Skip local symbols (scip represents them as 'local <id>')
  if (symbol.rfind("local ", 0) == 0) {
    log_debug("Skipping local symbol");
    return;
  }

  int symbol_roles = 0;
  if (!re2::RE2::PartialMatch(occurrence_block, symbol_roles_regex,
                              &symbol_roles)) {
    log_debug("Occurrence missing symbol_roles");
    return;
  }

  auto enclosing_ranges =
      extract_integers(occurrence_block, enclosing_range_regex);
  log_debug("Processing symbol '" + symbol + "' at line " +
            std::to_string(line) + " roles=" + std::to_string(symbol_roles));
  process_symbol(symbol, line, symbol_roles, enclosing_ranges, builder);
}

std::string
SCIPGraphDecoder::unify_symbol_name(const std::string &symbol) const {
  std::string clean_symbol = symbol;
  clean_symbol.erase(std::remove(clean_symbol.begin(), clean_symbol.end(), '`'),
                     clean_symbol.end());

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

std::string SCIPGraphDecoder::classify_symbol_type(
    const std::string &unified_symbol,
    const std::string &original_symbol) const {
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

void SCIPGraphDecoder::process_symbol(const std::string &symbol, int line,
                                      int symbol_roles,
                                      const std::vector<int> &enclosing_ranges,
                                      SubgraphBuilder &builder) const {
  static const re2::RE2 arg_regex(R"re2(\.\([^)]+\)$)re2");
  if (re2::RE2::PartialMatch(symbol, arg_regex)) {
    return;
  }

  builder.exit_scopes_by_line(line);

  static const re2::RE2 module_regex(R"re2(`?([^`]+)`?/[^.]+(?:\.|\(|#))re2");
  std::string module_path;
  if (!re2::RE2::PartialMatch(symbol, module_regex, &module_path)) {
    log_debug("Failed to match module for symbol: " + symbol);
    return;
  }

  std::string cleaned_symbol = symbol;
  if (auto last_space = cleaned_symbol.find_last_of(' ');
      last_space != std::string::npos) {
    cleaned_symbol = cleaned_symbol.substr(last_space + 1);
  }
  cleaned_symbol.erase(
      std::remove(cleaned_symbol.begin(), cleaned_symbol.end(), '`'),
      cleaned_symbol.end());

  std::string unified_symbol = unify_symbol_name(cleaned_symbol);
  std::string symbol_type =
      classify_symbol_type(unified_symbol, cleaned_symbol);

  const bool is_definition = symbol_roles == 1;
  const bool is_reference = symbol_roles == 8;

  if (unified_symbol.find("/__init__") != std::string::npos) {
    re2::RE2 init_regex(R"re2((.+)/(?:__init__))re2");
    std::string module_dir;
    if (re2::RE2::PartialMatch(unified_symbol, init_regex, &module_dir)) {
      std::replace(module_dir.begin(), module_dir.end(), '.', '/');
      std::string file_path = module_dir + ".py";
      if (is_reference) {
        builder.add_edge(builder.current_scope(), file_path,
                         EDGE_TYPE_REFERENCE);
      }
    }
    return;
  }

  if (is_definition && enclosing_ranges.size() >= 4) {
    int scope_start_line = enclosing_ranges[0];
    int scope_end_line = enclosing_ranges[2];
    log_debug("Adding definition with scope: " + unified_symbol + " [" +
              std::to_string(scope_start_line) + ", " +
              std::to_string(scope_end_line) + "]");
    builder.add_symbol_node(unified_symbol, line, scope_start_line,
                            scope_end_line, symbol_type);
    builder.add_containment_edge(unified_symbol);

    if (symbol_type == NODE_TYPE_CLASS || symbol_type == NODE_TYPE_FUNCTION ||
        symbol_type == NODE_TYPE_METHOD) {
      builder.update_current_scope(unified_symbol, scope_start_line,
                                   scope_end_line);
    }
  } else if (is_definition) {
    log_debug("Adding definition without scope: " + unified_symbol);
    builder.add_symbol_node(unified_symbol, line, std::nullopt, std::nullopt,
                            symbol_type);
    builder.add_edge(builder.current_scope(), unified_symbol,
                     EDGE_TYPE_CONTAIN);
  } else if (is_reference) {
    log_debug("Adding reference: " + unified_symbol + " from module " +
              module_path);
    builder.add_symbol_reference(unified_symbol, module_path, symbol_type);
  }
}

void SCIPGraphDecoder::merge_subgraphs(const std::vector<Subgraph> &subgraphs) {
  auto start = std::chrono::high_resolution_clock::now();

  // Collect nodes and edges from subgraphs
  std::vector<CodeGraph::VertexData> definition_nodes;
  std::vector<CodeGraph::VertexData> reference_nodes;
  std::vector<std::tuple<std::string, std::string, std::string>> all_edges;

  // Estimate total size
  size_t estimated_total = 0;
  for (const auto &subgraph : subgraphs) {
    estimated_total += subgraph.nodes.size();
  }

  definition_nodes.reserve(estimated_total);
  reference_nodes.reserve(estimated_total /
                          10); // Rough estimate: ~10% are references
  all_edges.reserve(estimated_total * 3);

  // Collect nodes and edges, separating definitions from references
  for (const auto &subgraph : subgraphs) {
    for (const auto &[name, node] : subgraph.nodes) {
      if (node.is_definition) {
        definition_nodes.emplace_back(node.data);
      } else {
        reference_nodes.emplace_back(node.data);
      }
    }

    for (const auto &edge : subgraph.edges) {
      all_edges.emplace_back(edge.source, edge.target, edge.type);
    }
  }

  auto collect_end = std::chrono::high_resolution_clock::now();
  auto collect_duration = std::chrono::duration_cast<std::chrono::milliseconds>(
      collect_end - start);
  log_debug("Collection phase: " + std::to_string(collect_duration.count()) +
            "ms");
  log_debug("Collected " + std::to_string(definition_nodes.size()) +
            " definition nodes, " + std::to_string(reference_nodes.size()) +
            " reference nodes, " + std::to_string(all_edges.size()) + " edges");

  // First pass: Insert all definition nodes (including directories, files, and
  // defined symbols)
  auto def_start = std::chrono::high_resolution_clock::now();
  code_graph_.batch_upsert_nodes(definition_nodes);
  auto def_end = std::chrono::high_resolution_clock::now();
  auto def_duration = std::chrono::duration_cast<std::chrono::milliseconds>(
      def_end - def_start);
  log_debug("Definition nodes batch insert: " +
            std::to_string(def_duration.count()) + "ms");

  // Second pass: Insert reference nodes (symbols referenced from external
  // modules)
  auto ref_start = std::chrono::high_resolution_clock::now();
  code_graph_.batch_upsert_nodes(reference_nodes);
  auto ref_end = std::chrono::high_resolution_clock::now();
  auto ref_duration = std::chrono::duration_cast<std::chrono::milliseconds>(
      ref_end - ref_start);
  log_debug("Reference nodes batch insert: " +
            std::to_string(ref_duration.count()) + "ms");

  // Batch insert all edges
  auto edge_start = std::chrono::high_resolution_clock::now();
  code_graph_.batch_add_edges(all_edges);
  auto edge_end = std::chrono::high_resolution_clock::now();
  auto edge_duration = std::chrono::duration_cast<std::chrono::milliseconds>(
      edge_end - edge_start);
  log_debug("Edge batch insert: " + std::to_string(edge_duration.count()) +
            "ms");

  auto end = std::chrono::high_resolution_clock::now();
  auto total_duration =
      std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
  log_debug("Total merge time: " + std::to_string(total_duration.count()) +
            "ms");
}

} // namespace codeminer::core
