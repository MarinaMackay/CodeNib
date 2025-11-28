#pragma once

#include <igraph/igraph.h>

#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace codeminer::core {

constexpr const char* NODE_TYPE_DIRECTORY = "directory";
constexpr const char* NODE_TYPE_FILE = "file";
constexpr const char* NODE_TYPE_SYMBOL = "symbol";
constexpr const char* NODE_TYPE_CLASS = "class";
constexpr const char* NODE_TYPE_FUNCTION = "function";
constexpr const char* NODE_TYPE_METHOD = "method";
constexpr const char* NODE_TYPE_FIELD = "field";
constexpr const char* EDGE_TYPE_CONTAIN = "contain";
constexpr const char* EDGE_TYPE_REFERENCE = "reference";
constexpr const char* ROOT_NODE = ".";

constexpr const char* ATTR_VERTEX_NAME = "name";
constexpr const char* ATTR_VERTEX_TYPE = "type";
constexpr const char* ATTR_VERTEX_FILE = "file";
constexpr const char* ATTR_VERTEX_START_LINE = "start_line";
constexpr const char* ATTR_VERTEX_END_LINE = "end_line";
constexpr const char* ATTR_EDGE_TYPE = "type";

class CodeGraph {
  public:
    using VertexId = igraph_integer_t;

    struct VertexData {
        std::string name;
        std::string type;
        std::optional<std::string> file;
        std::optional<int> start_line;
        std::optional<int> end_line;
    };

    CodeGraph();
    explicit CodeGraph(std::string project_root);
    ~CodeGraph();

    CodeGraph(const CodeGraph&) = delete;
    CodeGraph& operator=(const CodeGraph&) = delete;
    CodeGraph(CodeGraph&& other) noexcept;
    CodeGraph& operator=(CodeGraph&& other) noexcept;

    void add_root_node(const std::string& root_name);
    void add_directory_node(const std::string& dir_path);
    void add_file_node(const std::string& file_path);

    void add_symbol_node(const std::string& symbol,
                         int line,
                         std::optional<int> scope_start_line = std::nullopt,
                         std::optional<int> scope_end_line = std::nullopt,
                         const std::string& symbol_type = NODE_TYPE_SYMBOL);

    void add_symbol_reference(const std::string& symbol,
                              const std::optional<std::string>& module_path = std::nullopt,
                              const std::string& symbol_type = NODE_TYPE_SYMBOL);

    void add_containment_edge(const std::string& target_symbol);
    void update_current_scope(const std::string& symbol,
                              std::optional<int> start_line = std::nullopt,
                              std::optional<int> end_line = std::nullopt);
    void exit_scopes_by_line(int current_line);

    igraph_integer_t add_edge(const std::string& source,
                              const std::string& target,
                              const std::string& edge_type);

    void batch_upsert_nodes(const std::vector<VertexData>& nodes);
    void batch_add_edges(const std::vector<std::tuple<std::string, std::string, std::string>>& edges);

    std::optional<VertexData> get_node_info_by_name(const std::string& node_name) const;
    std::optional<VertexData> get_node_info_by_id(VertexId vertex_id) const;

    std::vector<VertexId> get_neighbors(const std::string& node_name) const;
    std::string get_node_content(VertexId vertex_id) const;

    void save_graph(const std::string& output_path) const;
    static CodeGraph load_graph(const std::string& input_path);

    const igraph_t& graph() const { return graph_; }
    igraph_t& graph() { return graph_; }

    const std::string& project_root() const { return project_root_; }
    void set_project_root(std::string project_root) { project_root_ = std::move(project_root); }

    const std::string& current_scope() const { return current_scope_; }

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

    VertexId ensure_vertex(const std::string& name);
    void apply_vertex_update(VertexId vertex_id,
                             const std::optional<std::string>& type,
                             const std::optional<std::string>& file,
                             const std::optional<int>& start_line,
                             const std::optional<int>& end_line);
    void set_vertex_string(VertexId vertex_id, const char* attribute, const std::string& value);
    void set_vertex_string(VertexId vertex_id, const char* attribute, const std::optional<std::string>& value);
    void set_vertex_numeric(VertexId vertex_id, const char* attribute, std::optional<int> value);
    std::optional<std::string> get_vertex_string(VertexId vertex_id, const char* attribute) const;
    std::optional<int> get_vertex_int(VertexId vertex_id, const char* attribute) const;

    void reset_scope_to_file(const std::string& file_symbol);

    igraph_t graph_{};
    std::unordered_map<std::string, VertexId> name_to_vertex_;
    std::unordered_map<std::string, Range> symbol_ranges_;
    std::vector<ScopeEntry> scope_stack_;

    std::string project_root_;
    std::string current_file_;
    std::string current_scope_;
};

}  // namespace codeminer::core
