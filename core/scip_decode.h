#pragma once

#include "code_graph.h"

#include <string>
#include <vector>
#include <optional>

namespace codeminer::core {

class SCIPGraphDecoder {
  public:
    explicit SCIPGraphDecoder(std::string index_file_path,
                              std::optional<std::string> project_root = std::nullopt);

    CodeGraph decode();

  private:
    struct Subgraph;
    class SubgraphBuilder;

    Subgraph process_document(const std::string& document_block) const;
    void process_occurrence(const std::string& occurrence_block, SubgraphBuilder& builder) const;
    void process_symbol(const std::string& symbol,
                        int line,
                        int symbol_roles,
                        const std::vector<int>& enclosing_ranges,
                        SubgraphBuilder& builder) const;
    void merge_subgraphs(const std::vector<Subgraph>& subgraphs);

    std::string unify_symbol_name(const std::string& symbol) const;
    std::string classify_symbol_type(const std::string& unified_symbol,
                                     const std::string& original_symbol) const;

    std::string index_file_path_;
    std::optional<std::string> project_root_;

    CodeGraph code_graph_;
};

}  // namespace codeminer::core
