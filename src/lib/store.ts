import Database from 'better-sqlite3';
import * as sqliteVec from 'sqlite-vec';
import type { ParsedNode, ParsedEdge, SearchResult } from './types.js';

export class Store {
  db: Database.Database;

  constructor(dbPath: string) {
    this.db = new Database(dbPath);
    this.db.pragma('journal_mode = WAL');
    sqliteVec.load(this.db);
    this.createSchema();
  }

  private createSchema(): void {
    this.db.exec(`
      CREATE TABLE IF NOT EXISTS nodes (
        id TEXT PRIMARY KEY,
        title TEXT NOT NULL,
        content TEXT,
        frontmatter TEXT NOT NULL DEFAULT '{}'
      );

      CREATE TABLE IF NOT EXISTS edges (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        source_id TEXT NOT NULL,
        target_id TEXT NOT NULL,
        context TEXT NOT NULL DEFAULT ''
      );

      CREATE INDEX IF NOT EXISTS idx_edges_source ON edges(source_id);
      CREATE INDEX IF NOT EXISTS idx_edges_target ON edges(target_id);

      CREATE TABLE IF NOT EXISTS communities (
        id INTEGER PRIMARY KEY,
        label TEXT NOT NULL,
        summary TEXT NOT NULL DEFAULT '',
        node_ids TEXT NOT NULL DEFAULT '[]'
      );

      CREATE TABLE IF NOT EXISTS sync (
        path TEXT PRIMARY KEY,
        mtime INTEGER NOT NULL,
        indexed_at INTEGER NOT NULL
      );

      CREATE VIRTUAL TABLE IF NOT EXISTS nodes_fts
        USING fts5(title, content, content='nodes', content_rowid='rowid');

      -- One row per content chunk (not per node): a note's body is split into
      -- overlapping ~200-token windows, each embedded separately, so a single
      -- node can have many vectors. node_id/chunk_index/chunk_text are
      -- auxiliary vec0 columns (stored, not ANN-indexed) used to trace a
      -- match back to its node and to show the exact matching passage.
      CREATE VIRTUAL TABLE IF NOT EXISTS node_chunks_vec
        USING vec0(embedding float[384], +node_id TEXT, +chunk_index INTEGER, +chunk_text TEXT);
    `);
  }

  upsertNode(node: ParsedNode): void {
    // FTS5 content-sync tables require manual delete-before-reinsert.
    // We must fetch the ACTUAL old values for the FTS5 delete command.
    const existing = this.db.prepare(
      'SELECT rowid, title, content FROM nodes WHERE id = ?'
    ).get(node.id) as { rowid: number; title: string; content: string } | undefined;

    if (existing) {
      this.db.prepare(
        "INSERT INTO nodes_fts(nodes_fts, rowid, title, content) VALUES('delete', ?, ?, ?)"
      ).run(existing.rowid, existing.title, existing.content);
    }

    this.db.prepare(`
      INSERT INTO nodes (id, title, content, frontmatter)
      VALUES (?, ?, ?, ?)
      ON CONFLICT(id) DO UPDATE SET
        title = excluded.title,
        content = excluded.content,
        frontmatter = excluded.frontmatter
    `).run(node.id, node.title, node.content, JSON.stringify(node.frontmatter));

    const row = this.db.prepare(
      'SELECT rowid FROM nodes WHERE id = ?'
    ).get(node.id) as { rowid: number };

    this.db.prepare(
      'INSERT INTO nodes_fts(rowid, title, content) VALUES(?, ?, ?)'
    ).run(row.rowid, node.title, node.content);
  }

  getNode(id: string): (ParsedNode & { rowid: number }) | undefined {
    const row = this.db.prepare(
      'SELECT rowid, id, title, content, frontmatter FROM nodes WHERE id = ?'
    ).get(id) as any;
    if (!row) return undefined;
    return {
      id: row.id,
      title: row.title,
      content: row.content,
      frontmatter: JSON.parse(row.frontmatter),
      rowid: row.rowid,
    };
  }

  allNodeIds(): string[] {
    return this.db.prepare('SELECT id FROM nodes').all().map((r: any) => r.id);
  }

  insertEdge(edge: ParsedEdge): void {
    this.db.prepare(
      'INSERT INTO edges (source_id, target_id, context) VALUES (?, ?, ?)'
    ).run(edge.sourceId, edge.targetId, edge.context);
  }

  getEdgesFrom(nodeId: string): Array<ParsedEdge & { id: number }> {
    return this.db.prepare(
      'SELECT id, source_id, target_id, context FROM edges WHERE source_id = ?'
    ).all(nodeId).map((r: any) => ({
      id: r.id,
      sourceId: r.source_id,
      targetId: r.target_id,
      context: r.context,
    }));
  }

  getEdgesTo(nodeId: string): Array<ParsedEdge & { id: number }> {
    return this.db.prepare(
      'SELECT id, source_id, target_id, context FROM edges WHERE target_id = ?'
    ).all(nodeId).map((r: any) => ({
      id: r.id,
      sourceId: r.source_id,
      targetId: r.target_id,
      context: r.context,
    }));
  }

  countEdgesFrom(nodeId: string): number {
    const row = this.db.prepare(
      'SELECT COUNT(*) as cnt FROM edges WHERE source_id = ?'
    ).get(nodeId) as { cnt: number };
    return row.cnt;
  }

  countEdgesTo(nodeId: string): number {
    const row = this.db.prepare(
      'SELECT COUNT(*) as cnt FROM edges WHERE target_id = ?'
    ).get(nodeId) as { cnt: number };
    return row.cnt;
  }

  getEdgeSummariesFrom(nodeId: string): Array<{ nodeId: string; title: string }> {
    return this.db.prepare(`
      SELECT DISTINCT e.target_id, n.title
      FROM edges e
      LEFT JOIN nodes n ON n.id = e.target_id
      WHERE e.source_id = ?
    `).all(nodeId).map((r: any) => ({
      nodeId: r.target_id,
      title: r.title ?? r.target_id,
    }));
  }

  getEdgeSummariesTo(nodeId: string): Array<{ nodeId: string; title: string }> {
    return this.db.prepare(`
      SELECT DISTINCT e.source_id, n.title
      FROM edges e
      LEFT JOIN nodes n ON n.id = e.source_id
      WHERE e.target_id = ?
    `).all(nodeId).map((r: any) => ({
      nodeId: r.source_id,
      title: r.title ?? r.source_id,
    }));
  }

  deleteNode(id: string): void {
    // FTS5 delete requires actual old values, not empty strings
    const row = this.db.prepare(
      'SELECT rowid, title, content FROM nodes WHERE id = ?'
    ).get(id) as { rowid: number; title: string; content: string } | undefined;

    if (row) {
      this.db.prepare(
        "INSERT INTO nodes_fts(nodes_fts, rowid, title, content) VALUES('delete', ?, ?, ?)"
      ).run(row.rowid, row.title, row.content);
    }
    this.db.prepare('DELETE FROM node_chunks_vec WHERE node_id = ?').run(id);

    this.db.prepare('DELETE FROM nodes WHERE id = ?').run(id);
    this.db.prepare('DELETE FROM edges WHERE source_id = ? OR target_id = ?').run(id, id);
    this.db.prepare('DELETE FROM sync WHERE path = ?').run(id);
  }

  deleteAllEdgesFrom(nodeId: string): void {
    this.db.prepare('DELETE FROM edges WHERE source_id = ?').run(nodeId);
  }

  searchFullText(query: string): SearchResult[] {
    return this.db.prepare(`
      SELECT n.id, n.title, rank,
        snippet(nodes_fts, 1, '>>>', '<<<', '...', 40) as excerpt
      FROM nodes_fts f
      JOIN nodes n ON n.rowid = f.rowid
      WHERE nodes_fts MATCH ?
      ORDER BY rank
      LIMIT 20
    `).all(query).map((r: any) => ({
      nodeId: r.id,
      title: r.title,
      score: -r.rank,
      excerpt: r.excerpt ?? '',
    }));
  }

  /**
   * Replaces all chunk embeddings for a node. Unlike the old one-vector-per-node
   * scheme (keyed by nodes.rowid), chunk rows are keyed by the node_id auxiliary
   * column, so a node can own any number of chunk vectors.
   */
  upsertEmbeddings(nodeId: string, chunks: Array<{ index: number; text: string; embedding: Float32Array }>): void {
    this.db.prepare('DELETE FROM node_chunks_vec WHERE node_id = ?').run(nodeId);
    const insert = this.db.prepare(
      'INSERT INTO node_chunks_vec(embedding, node_id, chunk_index, chunk_text) VALUES (?, ?, ?, ?)'
    );
    for (const chunk of chunks) {
      // sqlite-vec aux INTEGER columns require BigInt binding, or better-sqlite3
      // sends plain numbers as REAL and vec0 rejects the type mismatch.
      insert.run(Buffer.from(chunk.embedding.buffer), nodeId, BigInt(chunk.index), chunk.text);
    }
  }

  /**
   * KNN over chunk vectors, then collapsed to one result per node using its
   * best (highest-scoring) chunk. candidateK over-fetches chunks so that
   * enough distinct nodes surface before truncating to `limit`.
   */
  searchVector(embedding: Float32Array, limit = 20): SearchResult[] {
    const candidateK = Math.max(limit * 10, 200);
    const rows = this.db.prepare(`
      SELECT node_id, chunk_text, distance
      FROM node_chunks_vec
      WHERE embedding MATCH ? AND k = ?
      ORDER BY distance
    `).all(Buffer.from(embedding.buffer), candidateK) as Array<{
      node_id: string; chunk_text: string; distance: number;
    }>;

    const bestByNode = new Map<string, { distance: number; chunkText: string }>();
    for (const row of rows) {
      const existing = bestByNode.get(row.node_id);
      if (!existing || row.distance < existing.distance) {
        bestByNode.set(row.node_id, { distance: row.distance, chunkText: row.chunk_text });
      }
    }

    const nodeIds = [...bestByNode.keys()];
    if (nodeIds.length === 0) return [];

    const titleRows = this.db.prepare(
      `SELECT id, title FROM nodes WHERE id IN (${nodeIds.map(() => '?').join(',')})`
    ).all(...nodeIds) as Array<{ id: string; title: string }>;
    const titleById = new Map(titleRows.map(r => [r.id, r.title]));

    return nodeIds
      .map(nodeId => {
        const best = bestByNode.get(nodeId)!;
        return {
          nodeId,
          title: titleById.get(nodeId) ?? nodeId,
          score: 1 - best.distance,
          excerpt: best.chunkText,
        };
      })
      .sort((a, b) => b.score - a.score)
      .slice(0, limit);
  }

  upsertSync(path: string, mtime: number): void {
    this.db.prepare(`
      INSERT INTO sync (path, mtime, indexed_at) VALUES (?, ?, ?)
      ON CONFLICT(path) DO UPDATE SET mtime = excluded.mtime, indexed_at = excluded.indexed_at
    `).run(path, mtime, Date.now());
  }

  getSyncMtime(path: string): number | undefined {
    const row = this.db.prepare(
      'SELECT mtime FROM sync WHERE path = ?'
    ).get(path) as { mtime: number } | undefined;
    return row?.mtime;
  }

  getAllSyncPaths(): Set<string> {
    return new Set(
      this.db.prepare('SELECT path FROM sync').all().map((r: any) => r.path)
    );
  }

  upsertCommunity(community: { id: number; label: string; summary: string; nodeIds: string[] }): void {
    this.db.prepare(`
      INSERT INTO communities (id, label, summary, node_ids) VALUES (?, ?, ?, ?)
      ON CONFLICT(id) DO UPDATE SET
        label = excluded.label,
        summary = excluded.summary,
        node_ids = excluded.node_ids
    `).run(community.id, community.label, community.summary, JSON.stringify(community.nodeIds));
  }

  clearCommunities(): void {
    this.db.prepare('DELETE FROM communities').run();
  }

  getAllCommunities(): Array<{ id: number; label: string; summary: string; nodeIds: string[] }> {
    return this.db.prepare('SELECT * FROM communities').all().map((r: any) => ({
      id: r.id,
      label: r.label,
      summary: r.summary,
      nodeIds: JSON.parse(r.node_ids),
    }));
  }

  close(): void {
    this.db.close();
  }
}
