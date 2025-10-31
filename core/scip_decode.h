#pragma once

#include "code_graph.h"

#include <string>
#include <unordered_set>
#include <vector>
#include <optional>

namespace codeminer::core {

class SCIPGraphDecoder {
  public:
    explicit SCIPGraphDecoder(std::string index_file_path,
                              std::optional<std::string> project_root = std::nullopt);

    CodeGraph decode();

  private:
    void process_document(const std::string& document_block);
    void process_occurrence(const std::string& occurrence_block);
    void process_symbol(const std::string& symbol,
                        int line,
                        int symbol_roles,
                        const std::vector<int>& enclosing_ranges);

    std::string unify_symbol_name(const std::string& symbol) const;
    std::string classify_symbol_type(const std::string& unified_symbol,
                                     const std::string& original_symbol) const;

    std::string index_file_path_;
    std::optional<std::string> project_root_;

    CodeGraph code_graph_;
    std::unordered_set<std::string> indexed_directories_;
};

}  // namespace codeminer::core
