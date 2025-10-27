"""
SQLite-based in-memory index for CodeGraph nodes.
Provides fast SQL query capabilities.
"""

import sqlite3
from typing import List, Optional, Tuple, Dict, Any

from ..graph.code_graph import CodeGraph
from ..log_utils import get_logger

logger = get_logger(__name__)

MAX_CONTENT_CHARS = 8000  # Content truncation to avoid memory overflow


class SqliteNodeIndex:
    """
    In-memory SQLite index for CodeGraph nodes.
    Supports arbitrary SQL queries including GLOB pattern matching.
    """

    def __init__(self, code_graph: CodeGraph):
        """
        Initialize SqliteNodeIndex and build index from CodeGraph.

        Args:
            code_graph: CodeGraph instance containing nodes to index
        """
        # Initialize in-memory database (shared cache mode)
        self.con = sqlite3.connect(
            "file:scip_nodes?mode=memory&cache=shared", uri=True
        ) 
        self.con.execute("PRAGMA journal_mode=MEMORY;") # Memory-only journal mode
        self.con.execute("PRAGMA synchronous=OFF;") # Asynchronous commit mode
        self.con.execute("PRAGMA temp_store=MEMORY;") # Memory-only temporary store
        self.con.execute("PRAGMA cache_size=-80000;")  # ~80MB page cache

        self.code_graph = code_graph
        self._init_schema()
        self._build_index()

        logger.info(
            f"SqliteNodeIndex initialized with {self.con.execute('SELECT COUNT(*) FROM nodes').fetchone()[0]} nodes"
        )

    def _init_schema(self):
        """Initialize database schema."""
        cur = self.con.cursor()
        cur.executescript(
            """
CREATE TABLE IF NOT EXISTS nodes (
  vertex_id    INTEGER PRIMARY KEY,
  node_name    TEXT NOT NULL,
  type         TEXT,
  file         TEXT,
  start_line   INTEGER,
  end_line     INTEGER,
  content      TEXT
);

CREATE INDEX IF NOT EXISTS idx_node_name ON nodes(node_name);
"""
        )
        self.con.commit()

    def _build_index(self):
        """Build index from CodeGraph."""
        cur = self.con.cursor()
        cur.execute("DELETE FROM nodes;") # Delete all nodes from the table
        self.con.commit()
        
        vs = self.code_graph.get_graph().vs # Get all vertices from the graph
        for v in vs:
            vid = v.index
            attrs = v.attributes()
            node_name = v["name"]
            node_type = attrs.get("type") or ""
            file = attrs.get("file")
            start_line = attrs.get("start_line")
            end_line = attrs.get("end_line")

            # Get content from the node
            content: Optional[str] = None
            try:
                content = self.code_graph.get_node_content(vid)
                if content:
                    content = content[:MAX_CONTENT_CHARS]
            except Exception as e:
                logger.debug(f"Failed to get content for node {vid}: {e}")
                content = None

            cur.execute(
                "INSERT INTO nodes(vertex_id, node_name, type, file, start_line, end_line, content) "
                "VALUES(?, ?, ?, ?, ?, ?, ?);",
                (vid, node_name, node_type, file, start_line, end_line, content),
            )

        self.con.commit()
        logger.info(f"Built index with {len(vs)} nodes")

    def query(self, sql: str, params: Tuple = ()) -> List[Tuple]:
        """
        Execute arbitrary SQL query and return results.

        Args:
            sql: SQL query string (supports GLOB, LIKE, WHERE, etc.)
            params: Query parameters (for parameterized queries)

        Returns:
            List of tuples containing query results

        Examples:
            >>> idx.query("SELECT * FROM nodes WHERE type='file' LIMIT 10")
            >>> idx.query("SELECT * FROM nodes WHERE file GLOB '*calculator*'")
            >>> idx.query("SELECT type, COUNT(*) FROM nodes GROUP BY type")
            >>> idx.query("SELECT * FROM nodes WHERE vertex_id = ?", (42,))
        """
        try:
            return self.con.execute(sql, params).fetchall()
        except sqlite3.Error as e:
            logger.error(f"SQL query failed: {e}")
            raise

    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the indexed nodes.

        Returns:
            Dictionary with statistics including:
            - total_nodes: Total number of nodes
            - nodes_by_type: Count of nodes per type
            - files_indexed: Number of file nodes
        """
        stats = {}

        # Total nodes
        stats["total_nodes"] = self.query("SELECT COUNT(*) FROM nodes")[0][0]

        # Nodes by type
        type_counts = self.query("SELECT type, COUNT(*) FROM nodes GROUP BY type")
        stats["nodes_by_type"] = {type_name: count for type_name, count in type_counts}

        # Files indexed
        stats["files_indexed"] = self.query(
            "SELECT COUNT(*) FROM nodes WHERE type='file'"
        )[0][0]

        return stats