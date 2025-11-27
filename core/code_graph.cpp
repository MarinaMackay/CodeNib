#include "code_graph.h"
#include <igraph/igraph_attributes.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace codeminer::core {

namespace {

bool initialize_attribute_table() {
    igraph_i_set_attribute_table(&igraph_cattribute_table);
    return true;
}

[[maybe_unused]] const bool ATTRIBUTE_TABLE_READY = initialize_attribute_table();

bool debug_enabled() {
    static const bool enabled = std::getenv("CODEMINER_SCIP_DEBUG") != nullptr;
    return enabled;
}

void log_debug(const std::string& message) {
    if (debug_enabled()) {
        std::cerr << "[CodeGraph] " << message << '\n';
    }
}

std::string escape_json(const std::string& input) {
    std::ostringstream oss;
    oss << '"';
    for (char ch : input) {
        switch (ch) {
            case '\\':
                oss << "\\\\";
                break;
            case '"':
                oss << "\\\"";
                break;
            case '\n':
                oss << "\\n";
                break;
            case '\r':
                oss << "\\r";
                break;
            case '\t':
                oss << "\\t";
                break;
            default:
                if (static_cast<unsigned char>(ch) < 0x20) {
                    oss << "\\u" << std::uppercase << std::hex << std::setw(4) << std::setfill('0')
                        << static_cast<int>(static_cast<unsigned char>(ch)) << std::nouppercase << std::dec;
                    oss << std::setfill(' ');
                } else {
                    oss << ch;
                }
                break;
        }
    }
    oss << '"';
    return oss.str();
}

std::optional<std::string> read_file(const std::string& path) {
    std::ifstream input(path);
    if (!input.is_open()) {
        return std::nullopt;
    }

    std::ostringstream buffer;
    buffer << input.rdbuf();
    return buffer.str();
}

}  // namespace

CodeGraph::CodeGraph() : CodeGraph(std::string{}) {}

CodeGraph::CodeGraph(std::string project_root) : project_root_(std::move(project_root)) {
    if (igraph_empty(&graph_, /*n=*/0, /*directed=*/1) != IGRAPH_SUCCESS) {
        throw std::runtime_error("Failed to initialize igraph instance");
    }
}

CodeGraph::~CodeGraph() {
    igraph_destroy(&graph_);
}

CodeGraph::CodeGraph(CodeGraph&& other) noexcept : CodeGraph() {
    std::swap(graph_, other.graph_);
    name_to_vertex_ = std::move(other.name_to_vertex_);
    symbol_ranges_ = std::move(other.symbol_ranges_);
    scope_stack_ = std::move(other.scope_stack_);
    project_root_ = std::move(other.project_root_);
    current_file_ = std::move(other.current_file_);
    current_scope_ = std::move(other.current_scope_);
}

CodeGraph& CodeGraph::operator=(CodeGraph&& other) noexcept {
    if (this == &other) {
        return *this;
    }

    CodeGraph tmp(std::move(other));
    std::swap(graph_, tmp.graph_);
    name_to_vertex_ = std::move(tmp.name_to_vertex_);
    symbol_ranges_ = std::move(tmp.symbol_ranges_);
    scope_stack_ = std::move(tmp.scope_stack_);
    project_root_ = std::move(tmp.project_root_);
    current_file_ = std::move(tmp.current_file_);
    current_scope_ = std::move(tmp.current_scope_);
    return *this;
}

void CodeGraph::reset_scope_to_file(const std::string& file_symbol) {
    scope_stack_.clear();
    scope_stack_.push_back(ScopeEntry{file_symbol, Range{false, 0, 0}});
    current_scope_ = file_symbol;
}

CodeGraph::VertexId CodeGraph::ensure_vertex(const std::string& name) {
    auto it = name_to_vertex_.find(name);
    if (it != name_to_vertex_.end()) {
        log_debug("Reusing vertex for '" + name + "' with id " + std::to_string(it->second));
        return it->second;
    }

    if (igraph_add_vertices(&graph_, 1, nullptr) != IGRAPH_SUCCESS) {
        throw std::runtime_error("Failed to add vertex to graph");
    }

    VertexId vertex_id = static_cast<VertexId>(igraph_vcount(&graph_) - 1);
    name_to_vertex_.emplace(name, vertex_id);
    log_debug("Created vertex '" + name + "' with id " + std::to_string(vertex_id));

    set_vertex_string(vertex_id, ATTR_VERTEX_NAME, name);
    set_vertex_string(vertex_id, ATTR_VERTEX_TYPE, std::string{});
    set_vertex_string(vertex_id, ATTR_VERTEX_FILE, std::nullopt);
    set_vertex_numeric(vertex_id, ATTR_VERTEX_START_LINE, std::nullopt);
    set_vertex_numeric(vertex_id, ATTR_VERTEX_END_LINE, std::nullopt);

    return vertex_id;
}

void CodeGraph::apply_vertex_update(VertexId vertex_id,
                                    const std::optional<std::string>& type,
                                    const std::optional<std::string>& file,
                                    const std::optional<int>& start_line,
                                    const std::optional<int>& end_line) {
    if (type.has_value()) {
        set_vertex_string(vertex_id, ATTR_VERTEX_TYPE, *type);
    }
    if (file.has_value()) {
        set_vertex_string(vertex_id, ATTR_VERTEX_FILE, file);
    }
    if (start_line.has_value()) {
        set_vertex_numeric(vertex_id, ATTR_VERTEX_START_LINE, start_line);
    }
    if (end_line.has_value()) {
        set_vertex_numeric(vertex_id, ATTR_VERTEX_END_LINE, end_line);
    }
}

void CodeGraph::set_vertex_string(VertexId vertex_id, const char* attribute, const std::string& value) {
    if (vertex_id < 0 || vertex_id >= igraph_vcount(&graph_)) {
        throw std::out_of_range("Vertex id out of range when setting attribute '" + std::string(attribute) + "'");
    }
    if (igraph_cattribute_VAS_set(&graph_, attribute, vertex_id, value.c_str()) != IGRAPH_SUCCESS) {
        throw std::runtime_error("Failed to set vertex attribute '" + std::string(attribute) + "'");
    }
}

void CodeGraph::set_vertex_string(VertexId vertex_id,
                                  const char* attribute,
                                  const std::optional<std::string>& value) {
    if (value.has_value()) {
        set_vertex_string(vertex_id, attribute, *value);
    } else {
        set_vertex_string(vertex_id, attribute, std::string{});
    }
}

void CodeGraph::set_vertex_numeric(VertexId vertex_id,
                                   const char* attribute,
                                   std::optional<int> value) {
    if (vertex_id < 0 || vertex_id >= igraph_vcount(&graph_)) {
        throw std::out_of_range("Vertex id out of range when setting numeric attribute '" +
                                std::string(attribute) + "'");
    }
    igraph_real_t attr_value =
        value.has_value() ? static_cast<igraph_real_t>(*value) : std::numeric_limits<igraph_real_t>::quiet_NaN();
    if (igraph_cattribute_VAN_set(&graph_, attribute, vertex_id, attr_value) != IGRAPH_SUCCESS) {
        throw std::runtime_error("Failed to set vertex numeric attribute '" + std::string(attribute) + "'");
    }
}

std::optional<std::string> CodeGraph::get_vertex_string(VertexId vertex_id, const char* attribute) const {
    if (vertex_id < 0 || vertex_id >= igraph_vcount(&graph_)) {
        return std::nullopt;
    }
    const char* value = igraph_cattribute_VAS(&graph_, attribute, vertex_id);
    if (value == nullptr || value[0] == '\0') {
        return std::nullopt;
    }
    return std::string(value);
}

std::optional<int> CodeGraph::get_vertex_int(VertexId vertex_id, const char* attribute) const {
    if (vertex_id < 0 || vertex_id >= igraph_vcount(&graph_)) {
        return std::nullopt;
    }
    igraph_real_t numeric = igraph_cattribute_VAN(&graph_, attribute, vertex_id);
    if (std::isnan(numeric)) {
        return std::nullopt;
    }
    return static_cast<int>(numeric);
}

void CodeGraph::add_root_node(const std::string& root_name) {
    VertexId id = ensure_vertex(root_name);
    apply_vertex_update(id, std::string("root"), std::nullopt, std::nullopt, std::nullopt);
}

void CodeGraph::add_directory_node(const std::string& dir_path) {
    VertexId id = ensure_vertex(dir_path);
    apply_vertex_update(id, NODE_TYPE_DIRECTORY, std::nullopt, std::nullopt, std::nullopt);
}

void CodeGraph::add_file_node(const std::string& file_path) {
    current_file_ = file_path;
    VertexId id = ensure_vertex(file_path);
    apply_vertex_update(id, NODE_TYPE_FILE, std::nullopt, std::nullopt, std::nullopt);
    reset_scope_to_file(file_path);
}

void CodeGraph::add_symbol_node(const std::string& symbol,
                                int line,
                                std::optional<int> scope_start_line,
                                std::optional<int> scope_end_line,
                                const std::string& symbol_type) {
    VertexId id = ensure_vertex(symbol);
    apply_vertex_update(id, symbol_type, current_file_, line, line);

    if (scope_start_line.has_value() && scope_end_line.has_value()) {
        apply_vertex_update(id, symbol_type, current_file_, scope_start_line, scope_end_line);
        symbol_ranges_[symbol] = Range{true, *scope_start_line, *scope_end_line};
    }
}

void CodeGraph::add_symbol_reference(const std::string& symbol,
                                     const std::optional<std::string>& module_path,
                                     const std::string& symbol_type) {
    const bool already_exists = name_to_vertex_.find(symbol) != name_to_vertex_.end();
    VertexId id = ensure_vertex(symbol);
    std::optional<std::string> file_attr = already_exists ? std::nullopt : module_path;
    apply_vertex_update(id, symbol_type, file_attr, std::nullopt, std::nullopt);
    add_edge(current_scope_, symbol, EDGE_TYPE_REFERENCE);
}

void CodeGraph::add_containment_edge(const std::string& target_symbol) {
    add_edge(current_scope_, target_symbol, EDGE_TYPE_CONTAIN);
}

void CodeGraph::update_current_scope(const std::string& symbol,
                                     std::optional<int> start_line,
                                     std::optional<int> end_line) {
    current_scope_ = symbol;
    Range range;
    if (start_line.has_value() && end_line.has_value()) {
        range.has_range = true;
        range.start_line = *start_line;
        range.end_line = *end_line;
    }
    scope_stack_.push_back(ScopeEntry{symbol, range});
}

void CodeGraph::exit_scopes_by_line(int current_line) {
    while (scope_stack_.size() > 1) {
        const ScopeEntry& top = scope_stack_.back();
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

igraph_integer_t CodeGraph::add_edge(const std::string& source,
                                     const std::string& target,
                                     const std::string& edge_type) {
    log_debug("Adding edge from '" + source + "' to '" + target + "' type '" + edge_type + "'");
    VertexId source_id = ensure_vertex(source);
    VertexId target_id = ensure_vertex(target);

    igraph_integer_t eid = -1;
    if (igraph_get_eid(&graph_, &eid, source_id, target_id, /*directed=*/1, /*error=*/0) == IGRAPH_SUCCESS &&
        eid >= 0) {
        log_debug("Edge already exists; returning eid " + std::to_string(eid) + " without updating type");
        return eid;
    }

    log_debug("Edge not present; creating new edge");
    if (igraph_add_edge(&graph_, source_id, target_id) != IGRAPH_SUCCESS) {
        throw std::runtime_error("Failed to add edge to graph");
    }

    igraph_integer_t new_eid = static_cast<igraph_integer_t>(igraph_ecount(&graph_) - 1);
    if (igraph_cattribute_EAS_set(&graph_, ATTR_EDGE_TYPE, new_eid, edge_type.c_str()) != IGRAPH_SUCCESS) {
        throw std::runtime_error("Failed to set edge attribute 'type'");
    }
    return new_eid;
}

std::optional<CodeGraph::VertexData> CodeGraph::get_node_info_by_name(const std::string& node_name) const {
    auto it = name_to_vertex_.find(node_name);
    if (it == name_to_vertex_.end()) {
        return std::nullopt;
    }
    return get_node_info_by_id(it->second);
}

std::optional<CodeGraph::VertexData> CodeGraph::get_node_info_by_id(VertexId vertex_id) const {
    if (vertex_id < 0 || vertex_id >= igraph_vcount(&graph_)) {
        return std::nullopt;
    }
    VertexData data{};
    data.name = get_vertex_string(vertex_id, ATTR_VERTEX_NAME).value_or(std::string{});
    data.type = get_vertex_string(vertex_id, ATTR_VERTEX_TYPE).value_or(std::string{});
    data.file = get_vertex_string(vertex_id, ATTR_VERTEX_FILE);
    data.start_line = get_vertex_int(vertex_id, ATTR_VERTEX_START_LINE);
    data.end_line = get_vertex_int(vertex_id, ATTR_VERTEX_END_LINE);
    return data;
}

std::vector<CodeGraph::VertexId> CodeGraph::get_neighbors(const std::string& node_name) const {
    std::vector<VertexId> results;
    auto it = name_to_vertex_.find(node_name);
    if (it == name_to_vertex_.end()) {
        return results;
    }

    igraph_vector_int_t neighbors;
    igraph_vector_int_init(&neighbors, 0);
    igraph_neighbors(const_cast<igraph_t*>(&graph_), &neighbors, it->second, IGRAPH_OUT);

    igraph_integer_t count = neighbors.stor_end - neighbors.stor_begin;
    for (igraph_integer_t i = 0; i < count; ++i) {
        results.push_back(static_cast<VertexId>(neighbors.stor_begin[i]));
    }
    igraph_vector_int_destroy(&neighbors);
    return results;
}

std::string CodeGraph::get_node_content(VertexId vertex_id) const {
    auto vertex_opt = get_node_info_by_id(vertex_id);
    if (!vertex_opt.has_value()) {
        return {};
    }
    const VertexData& data = vertex_opt.value();
    if (data.type == NODE_TYPE_FILE) {
        std::string file_path = data.name;
        if (!project_root_.empty()) {
            file_path = std::filesystem::path(project_root_) / file_path;
        }
        auto file_content = read_file(file_path);
        if (file_content.has_value()) {
            return *file_content;
        }
        return {};
    }

    if ((data.type == NODE_TYPE_CLASS || data.type == NODE_TYPE_FUNCTION || data.type == NODE_TYPE_METHOD ||
         data.type == NODE_TYPE_FIELD || data.type == NODE_TYPE_SYMBOL) &&
        data.file.has_value() && data.start_line.has_value() && data.end_line.has_value()) {
        std::string file_path = *data.file;
        if (!project_root_.empty()) {
            file_path = std::filesystem::path(project_root_) / file_path;
        }

        std::ifstream input(file_path);
        if (!input.is_open()) {
            return {};
        }

        std::vector<std::string> lines;
        std::string line;
        while (std::getline(input, line)) {
            lines.push_back(line);
        }

        int start_index = std::max(0, data.start_line.value() - 1);
        int end_index = std::min(static_cast<int>(lines.size()), data.end_line.value());
        if (start_index >= end_index) {
            return {};
        }

        std::ostringstream snippet;
        for (int i = start_index; i < end_index; ++i) {
            snippet << lines[i] << '\n';
        }
        return snippet.str();
    }
    return {};
}

void CodeGraph::save_graph(const std::string& output_path) const {
    std::ofstream out(output_path);
    if (!out.is_open()) {
        throw std::runtime_error("Failed to open output file for writing graph");
    }

    out << "{\n";
    out << "  \"project_root\": " << escape_json(project_root_) << ",\n";
    igraph_integer_t vertex_count = igraph_vcount(&graph_);
    out << "  \"vertices\": [\n";
    for (igraph_integer_t vid = 0; vid < vertex_count; ++vid) {
        auto info_opt = get_node_info_by_id(vid);
        VertexData data = info_opt.value_or(VertexData{});
        out << "    {\n";
        out << "      \"id\": " << vid << ",\n";
        out << "      \"name\": " << escape_json(data.name) << ",\n";
        out << "      \"type\": " << escape_json(data.type) << ",\n";
        out << "      \"file\": ";
        if (data.file.has_value()) {
            out << escape_json(*data.file);
        } else {
            out << "null";
        }
        out << ",\n";
        out << "      \"start_line\": ";
        if (data.start_line.has_value()) {
            out << *data.start_line;
        } else {
            out << "null";
        }
        out << ",\n";
        out << "      \"end_line\": ";
        if (data.end_line.has_value()) {
            out << *data.end_line;
        } else {
            out << "null";
        }
        out << "\n";
        out << "    }";
        if (vid + 1 < vertex_count) {
            out << ",";
        }
        out << "\n";
    }
    out << "  ],\n";

    out << "  \"edges\": [\n";
    igraph_integer_t edge_count = igraph_ecount(&graph_);
    for (igraph_integer_t eid = 0; eid < edge_count; ++eid) {
        igraph_integer_t source = 0;
        igraph_integer_t target = 0;
        igraph_edge(&graph_, eid, &source, &target);
        const char* type_attr = igraph_cattribute_EAS(&graph_, ATTR_EDGE_TYPE, eid);
        std::string type_str = type_attr != nullptr ? type_attr : "";
        out << "    {\n";
        out << "      \"id\": " << eid << ",\n";
        out << "      \"source\": " << source << ",\n";
        out << "      \"target\": " << target << ",\n";
        out << "      \"type\": " << escape_json(type_str) << "\n";
        out << "    }";
        if (eid + 1 < edge_count) {
            out << ",";
        }
        out << "\n";
    }
    out << "  ],\n";

    out << "  \"symbol_ranges\": {\n";
    std::size_t counter = 0;
    for (const auto& [symbol, range] : symbol_ranges_) {
        out << "    " << escape_json(symbol) << ": ";
        if (range.has_range) {
            out << "[" << range.start_line << ", " << range.end_line << "]";
        } else {
            out << "null";
        }
        if (++counter < symbol_ranges_.size()) {
            out << ",";
        }
        out << "\n";
    }
    out << "  }\n";
    out << "}\n";
}

CodeGraph CodeGraph::load_graph(const std::string& /*input_path*/) {
    throw std::runtime_error("CodeGraph::load_graph is not implemented yet for the C++ core");
}

}  // namespace codeminer::core
