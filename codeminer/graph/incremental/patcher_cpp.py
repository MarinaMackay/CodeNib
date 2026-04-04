"""C/C++-specific incremental patcher.

Uses clangd .idx files instead of LSP queries for incremental updates.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from pathlib import Path

from ...log_utils import get_logger
from ...types import (
    EDGE_TYPE_CONTAIN,
    EDGE_TYPE_REFERENCE,
    NODE_TYPE_FUNCTION,
    NODE_TYPE_METHOD,
)
from .patcher_base import PatcherBase

logger = get_logger(__name__)


class PatcherCpp(PatcherBase):
    """C/C++ incremental patcher using clangd .idx files."""

    def get_lsp_command(self):
        return ["clangd"]

    def _language_id(self):
        return "cpp"

    def _get_crossfile_token_types(self):
        return {
            "type",
            "class",
            "struct",
            "enum",
            "function",
            "method",
            "namespace",
            "macro",
        }

    def _build_unified_name(self, file_path, name, parent_unified_part, kind):
        node_type = self._classify_symbol_type(kind)
        clean_name = name.replace("::", ".")

        if parent_unified_part:
            display = f"{parent_unified_part}.{clean_name}"
        else:
            display = clean_name

        if node_type in (NODE_TYPE_FUNCTION, NODE_TYPE_METHOD):
            if not display.endswith("()"):
                display = f"{display}()"

        return f"{file_path}:{display}"

    def flatten_symbols(self, file_path, lsp_symbols):
        return self._flatten_symbols_default(file_path, lsp_symbols)

    # ═══════════════════════════════════════════════════════════
    # Override patch_files: C++ uses .idx path, not LSP
    # ═══════════════════════════════════════════════════════════

    def patch_files(self, changed_files, **kwargs):
        """C++ incremental: delete old subgraphs, reindex via .idx, rebuild."""
        total_stats = {
            "files_deleted": 0,
            "files_modified": 0,
            "files_added": 0,
            "files_renamed": 0,
            "vertices_deleted": 0,
            "vertices_created": 0,
            "refs_incoming": 0,
            "refs_outgoing": 0,
            "refs_remapped": 0,
            "refs_unmatched": 0,
        }

        # Rebuild indexes for delete_file_subgraph
        self.build_indexes()

        nodes_before = self.code_graph.graph.vcount()
        edges_before = self.code_graph.graph.ecount()

        # Collect all changed file paths
        all_changed = (
            changed_files.get("modified", [])
            + changed_files.get("added", [])
            + [new for _, new in changed_files.get("renamed", [])]
        )

        # Delete old subgraphs
        for path in changed_files.get("deleted", []):
            with self.profiler.section("delete_subgraph"):
                self.delete_file_subgraph(path)
            total_stats["files_deleted"] += 1

        for old_path, _ in changed_files.get("renamed", []):
            with self.profiler.section("delete_subgraph"):
                self.delete_file_subgraph(old_path)

        for path in changed_files.get("modified", []):
            with self.profiler.section("delete_subgraph"):
                self.delete_file_subgraph(path)

        for path in changed_files.get("added", []):
            with self.profiler.section("delete_subgraph"):
                self.delete_file_subgraph(path)

        # Reindex and rebuild via .idx
        if all_changed:
            stats = self._incremental_update_idx(all_changed)
            total_stats["vertices_created"] = stats["vertices_created"]
            total_stats["refs_outgoing"] = stats["refs_added"]

        total_stats["files_modified"] = len(changed_files.get("modified", []))
        total_stats["files_added"] = len(changed_files.get("added", []))
        total_stats["files_renamed"] = len(changed_files.get("renamed", []))

        nodes_after = self.code_graph.graph.vcount()
        edges_after = self.code_graph.graph.ecount()
        total_stats["nodes_before"] = nodes_before
        total_stats["nodes_after"] = nodes_after
        total_stats["edges_before"] = edges_before
        total_stats["edges_after"] = edges_after

        earlier = kwargs.get("earlier_commit", "")
        later = kwargs.get("later_commit", "")
        commit_info = ""
        if earlier and later:
            total_stats["commit_earlier"] = earlier
            total_stats["commit_later"] = later
            commit_info = f"commits {earlier[:12]}..{later[:12]}, "

        total_changed = sum(
            len(changed_files.get(k, []))
            for k in ("deleted", "added", "renamed", "modified")
        )
        logger.info(
            f"Incremental patch summary: {commit_info}"
            f"{total_changed} files changed, "
            f"nodes {nodes_before}→{nodes_after} "
            f"(Δ{nodes_after - nodes_before:+d}), "
            f"edges {edges_before}→{edges_after} "
            f"(Δ{edges_after - edges_before:+d})"
        )

        report = self.profiler.report(reset=True)
        if report:
            total_time = sum(s.total for _, s in report)
            lines = [
                f"[graph_patcher_cpp] profiler summary: "
                f"{total_time:.3f}s total across {len(report)} section(s)"
            ]
            for label, s in report:
                avg = s.total / s.count if s.count else 0
                lines.append(
                    f"  {label:<40s} total={s.total:>7.3f}s "
                    f"count={s.count:>3d} avg={avg:>7.3f}s "
                    f"min={s.min_duration:>7.3f}s max={s.max_duration:>7.3f}s"
                )
            logger.info("\n".join(lines))

        return total_stats

    # ═══════════════════════════════════════════════════════════
    # .idx-based incremental update
    # ═══════════════════════════════════════════════════════════

    def _incremental_update_idx(self, changed_files: list[str]) -> dict:
        """Reindex via clangd .idx, then rebuild graph for changed files."""
        stats = {"vertices_created": 0, "refs_added": 0}

        with self.profiler.section("cpp.reindex"):
            idx_dir = self._reindex_changed_files(changed_files)
        if idx_dir is None:
            logger.warning("clangd incremental indexing failed")
            return stats

        from codeminer.ls_index.clangd_decode import ClangdGraphDecoder

        with self.profiler.section("cpp.parse_idx"):
            decoder = ClangdGraphDecoder(
                idx_directory=str(idx_dir),
                project_root=str(self.project_root),
            )
            decoder._collect_all_idx()
            decoder._build_id_to_display()

        decoder.code_graph = self.code_graph

        with self.profiler.section("cpp.rebuild_graph"):
            v_count, r_count = self._apply_idx_data(decoder, changed_files)
            stats["vertices_created"] = v_count
            stats["refs_added"] = r_count

        logger.info(
            f"C++ .idx incremental: {len(changed_files)} files, "
            f"+{v_count} vertices, {r_count} refs"
        )
        return stats

    def _reindex_changed_files(self, changed_files: list[str]) -> Path | None:
        """Run clangd background-index, preserving .idx cache."""
        clangd = shutil.which("clangd")
        if not clangd:
            return None

        comp_db = None
        for candidate in [
            Path(self.project_root) / "compile_commands.json",
            Path(self.project_root) / "build" / "compile_commands.json",
        ]:
            # Skip empty/trivial files (bare "[]" is ~3 bytes)
            if candidate.exists() and candidate.stat().st_size > 4:
                try:
                    data = json.loads(candidate.read_text())
                    if isinstance(data, list) and len(data) > 0:
                        comp_db = candidate
                        break
                except Exception as exc:
                    logger.debug(f"Failed to parse {candidate}: {exc}")
                    continue
        if comp_db is None:
            logger.warning("No valid compile_commands.json found")
            return None

        candidates = [
            Path(self.project_root) / ".cache" / "clangd" / "index",
            comp_db.parent / ".cache" / "clangd" / "index",
        ]
        idx_dir = None
        for d in candidates:
            if d.exists() and any(d.glob("*.idx")):
                idx_dir = d
                break

        if idx_dir is None:
            logger.info("No .idx cache, running full clangd index")
            from codeminer.ls_index.clangd_indexer import ClangdIndexer

            indexer = ClangdIndexer(project_root=str(self.project_root))
            success = indexer.generate_index(compdb_path=str(comp_db))
            return indexer.idx_directory if success else None

        pre_mtime = max(
            (f.stat().st_mtime for f in idx_dir.glob("*.idx")),
            default=0,
        )
        pre_count = len(list(idx_dir.glob("*.idx")))
        logger.info(
            f"C++ incremental: {pre_count} .idx in {idx_dir}, "
            f"didOpen {len(changed_files)} files"
        )

        cmd = [
            clangd,
            "--background-index",
            f"--compile-commands-dir={comp_db.parent}",
            "--background-index-priority=normal",
            "--log=error",
        ]
        process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=str(self.project_root),
        )

        def lsp_send(msg):
            content = json.dumps(msg).encode()
            header = f"Content-Length: {len(content)}\r\n\r\n".encode()
            process.stdin.write(header + content)
            process.stdin.flush()

        try:
            lsp_send(
                {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "initialize",
                    "params": {
                        "processId": os.getpid(),
                        "rootUri": Path(self.project_root).as_uri(),
                        "capabilities": {},
                    },
                }
            )
            time.sleep(1)
            lsp_send(
                {
                    "jsonrpc": "2.0",
                    "method": "initialized",
                    "params": {},
                }
            )
            time.sleep(0.5)

            for fpath in changed_files:
                abs_path = Path(self.project_root) / fpath
                if not abs_path.is_file():
                    continue
                try:
                    text = abs_path.read_text(errors="replace")
                except Exception:
                    continue
                lsp_send(
                    {
                        "jsonrpc": "2.0",
                        "method": "textDocument/didOpen",
                        "params": {
                            "textDocument": {
                                "uri": abs_path.as_uri(),
                                "languageId": "cpp",
                                "version": 1,
                                "text": text,
                            }
                        },
                    }
                )

            stable = 0
            last_mtime = pre_mtime
            for _ in range(120):
                time.sleep(1)
                cur_mtime = max(
                    (f.stat().st_mtime for f in idx_dir.glob("*.idx")),
                    default=0,
                )
                if cur_mtime > last_mtime:
                    last_mtime = cur_mtime
                    stable = 0
                else:
                    stable += 1
                    if stable >= 5 and cur_mtime > pre_mtime:
                        break
                    if stable >= 30:
                        break

            post_count = len(list(idx_dir.glob("*.idx")))
            logger.info(f"C++ incremental done: {pre_count}→{post_count} .idx files")

        finally:
            try:
                lsp_send(
                    {
                        "jsonrpc": "2.0",
                        "id": 99,
                        "method": "shutdown",
                        "params": None,
                    }
                )
                time.sleep(0.5)
                lsp_send(
                    {
                        "jsonrpc": "2.0",
                        "method": "exit",
                        "params": None,
                    }
                )
                process.wait(timeout=5)
            except Exception:
                process.kill()
                process.wait()

        return idx_dir

    def _apply_idx_data(
        self,
        decoder,
        changed_files: list[str],
    ) -> tuple[int, int]:
        """Add symbols and edges from .idx data for changed files."""
        from codeminer.ls_index.clangd_decode import (
            KIND_MACRO,
            REF_KIND_DEFINITION,
            REF_KIND_REFERENCE,
            ZERO_SYMBOL_ID,
        )

        changed_set = set(changed_files)
        new_vertices = 0
        new_refs = 0

        changed_sym_ids = set()
        for sym_id, sym in decoder._symbols.items():
            if sym.get("kind", 0) == KIND_MACRO:
                continue
            def_info = decoder._resolve_definition(sym_id)
            if not def_info["file"]:
                continue
            rel_file = decoder._file_uri_to_relative(def_info["file"])
            if rel_file in changed_set:
                changed_sym_ids.add(sym_id)

        for sym_id in changed_sym_ids:
            sym = decoder._symbols[sym_id]
            def_info = decoder._resolve_definition(sym_id)
            rel_file = decoder._file_uri_to_relative(def_info["file"])
            decoder._ensure_file_hierarchy(rel_file)

            node_type = decoder._kind_to_node_type(sym.get("kind", 0))
            qualified_name = decoder._sym_id_to_display_name(sym_id)
            line = def_info["line"]
            scope_start, scope_end = decoder._find_range(rel_file, line)

            self.code_graph.add_symbol_node(
                sym_id,
                line,
                scope_start_line=scope_start,
                scope_end_line=scope_end,
                symbol_type=node_type,
            )

            if sym_id in self.code_graph.name_to_vertex:
                vid = self.code_graph.name_to_vertex[sym_id]
                unified_display = qualified_name.replace("::", ".")
                if node_type in (NODE_TYPE_FUNCTION, NODE_TYPE_METHOD):
                    unified_display = f"{unified_display}()"
                self.code_graph.graph.vs[vid][
                    "unified_name"
                ] = f"{rel_file}:{unified_display}"
                self.code_graph.graph.vs[vid]["file"] = rel_file
                new_vertices += 1

        contained = set()
        for sym_id in changed_sym_ids:
            if sym_id not in self.code_graph.name_to_vertex:
                continue
            for ref in decoder._refs.get(sym_id, []):
                if ref["kind"] & REF_KIND_DEFINITION:
                    container_id = ref.get("container", "")
                    if (
                        container_id
                        and container_id != ZERO_SYMBOL_ID
                        and container_id in self.code_graph.name_to_vertex
                    ):
                        self.code_graph._add_edge(
                            container_id, sym_id, EDGE_TYPE_CONTAIN
                        )
                        contained.add(sym_id)
                        break

        for sym_id in changed_sym_ids:
            if sym_id in contained:
                continue
            if sym_id not in self.code_graph.name_to_vertex:
                continue
            vid = self.code_graph.name_to_vertex[sym_id]
            file_path = self.code_graph.graph.vs[vid].attributes().get("file")
            if file_path and file_path in self.code_graph.name_to_vertex:
                self.code_graph._add_edge(file_path, sym_id, EDGE_TYPE_CONTAIN)

        for sym_id, ref_list in decoder._refs.items():
            if sym_id not in self.code_graph.name_to_vertex:
                continue
            for ref in ref_list:
                if not (ref["kind"] & REF_KIND_REFERENCE):
                    continue
                container_id = ref.get("container", "")
                if not container_id or container_id == ZERO_SYMBOL_ID:
                    continue
                if container_id not in self.code_graph.name_to_vertex:
                    continue
                if sym_id in changed_sym_ids or container_id in changed_sym_ids:
                    self.code_graph._add_edge(container_id, sym_id, EDGE_TYPE_REFERENCE)
                    new_refs += 1

        return new_vertices, new_refs
